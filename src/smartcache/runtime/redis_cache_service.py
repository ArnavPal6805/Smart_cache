from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns
from typing import Any, Callable, Mapping

from smartcache.core.interfaces import CacheEntry, EvictionPolicy
from smartcache.runtime.redis_store import RedisSharedCache


@dataclass(frozen=True, slots=True)
class RedisCacheResult:
    payload: bytes
    hit: bool
    estimated_miss_cost_ms: float
    backend_latency_ms: float
    policy_time_ms: float
    evicted_keys: tuple[str, ...]
    used_bytes: int
    bypassed: bool


class RedisByteCacheRuntime:
    """Byte cache runtime coordinating admission through Redis."""

    def __init__(
        self,
        client: RedisSharedCache,
        policy: EvictionPolicy,
        origin_specs: Mapping[str, Any],
        origin_loader: Callable[[str], bytes],
        capacity_bytes: int = 24_576,
        max_objects: int = 16,
        frequency_window: int = 100,
        recency_window: int = 1000,
        cost_scale_ms: float = 250.0,
        size_scale_bytes: float = 32_768.0,
    ) -> None:
        self.store = client
        self.policy = policy
        self.origin_specs = origin_specs
        self.origin_loader = origin_loader
        self.capacity_bytes = capacity_bytes
        self.max_objects = max_objects
        self.frequency_window = frequency_window
        self.recency_window = recency_window
        self.cost_scale_ms = cost_scale_ms
        self.size_scale_bytes = size_scale_bytes
        self.position = 0
        self.metrics = {
            "requests": 0, "hits": 0, "misses": 0, "bypasses": 0,
            "evictions": 0, "policy_time_ms": 0.0, "backend_latency_ms": 0.0,
        }

    def get(self, key: str) -> RedisCacheResult:
        self.metrics["requests"] += 1
        self.position += 1
        spec = self.origin_specs[key]
        payload = self.store.get(key)
        if payload is not None:
            with self.store.transaction_lock():
                payload = self.store.get(key)
                if payload is not None:
                    self.store.touch(key, self.position)
            if payload is None:
                return self.get(key)
            self.metrics["hits"] += 1
            return RedisCacheResult(
                payload, True, 0.0, 0.0, 0.0, (), self.store.stats()["used_bytes"], False
            )

        started = perf_counter_ns()
        loaded = self.origin_loader(key)
        backend_ms = (perf_counter_ns() - started) / 1_000_000
        self.metrics["misses"] += 1
        if len(loaded) > self.capacity_bytes:
            self.metrics["bypasses"] += 1
            self.metrics["backend_latency_ms"] += backend_ms
            return RedisCacheResult(loaded, False, spec.retrieval_cost_ms, backend_ms, 0.0, (), self.store.stats()["used_bytes"], True)

        with self.store.transaction_lock():
            payload = self.store.get(key)
            if payload is not None:
                self.store.touch(key, self.position)
                self.metrics["hits"] += 1
                self.metrics["misses"] -= 1
                return RedisCacheResult(payload, True, 0.0, backend_ms, 0.0, (), self.store.stats()["used_bytes"], False)

            entries = [
                CacheEntry(
                    key=name,
                    retrieval_cost_ms=float(meta["retrieval_cost_ms"]),
                    size_bytes=int(meta["size_bytes"]),
                    frequency=int(meta.get("frequency", 0)),
                    last_access=int(meta.get("last_access", 0)),
                )
                for name in self.store.keys()
                if (meta := self.store.metadata(name)) is not None
            ]
            catalog = {
                name: {"cost": value.retrieval_cost_ms, "size": value.size_bytes}
                for name, value in self.origin_specs.items()
            }
            evicted: list[str] = []
            policy_ms = 0.0
            used = self.store.stats()["used_bytes"]
            while used + len(loaded) > self.capacity_bytes or len(entries) >= self.max_objects:
                if not entries:
                    raise RuntimeError("Redis cache cannot admit object with no eviction candidate")
                context = {
                    "catalog": catalog,
                    "frequency_counts": {entry.key: entry.frequency for entry in entries},
                    "frequency_window": self.frequency_window,
                    "recency_window": self.recency_window,
                    "cost_scale_ms": self.cost_scale_ms,
                    "size_scale_bytes": self.size_scale_bytes,
                    "capacity_bytes": self.capacity_bytes,
                    "max_objects": self.max_objects,
                    "used_bytes": used,
                    "pending_cost_charged": bool(evicted),
                }
                started_policy = perf_counter_ns()
                victim_key = self.policy.choose_victim(
                    entries,
                    {"key": key, "cost": spec.retrieval_cost_ms, "size": len(loaded)},
                    self.position,
                    context=context,
                )
                policy_ms += (perf_counter_ns() - started_policy) / 1_000_000
                victim = next(entry for entry in entries if entry.key == victim_key)
                self.store.remove(victim_key)
                entries.remove(victim)
                used -= victim.size_bytes
                evicted.append(victim_key)
                self.metrics["evictions"] += 1
            self.store.admit(key, loaded, spec.retrieval_cost_ms, [], self.position)
            self.metrics["policy_time_ms"] += policy_ms
            self.metrics["backend_latency_ms"] += backend_ms
            return RedisCacheResult(loaded, False, spec.retrieval_cost_ms, backend_ms, policy_ms, tuple(evicted), self.store.stats()["used_bytes"], False)

    def snapshot_metrics(self) -> dict[str, Any]:
        storage = self.store.stats()
        return {
            **self.metrics,
            **storage,
            "objects": storage["object_count"],
            "policy_fallback_errors": int(
                getattr(self.policy, "fallback_errors", 0)
            ),
        }
