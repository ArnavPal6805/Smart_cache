from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from time import perf_counter_ns
from typing import Any, Callable, Mapping

from smartcache.core.interfaces import CacheEntry, EvictionPolicy


@dataclass(frozen=True, slots=True)
class CacheResult:
    payload: bytes
    hit: bool
    estimated_miss_cost_ms: float
    backend_latency_ms: float
    policy_time_ms: float
    evicted_keys: tuple[str, ...]
    used_bytes: int
    bypassed: bool


class ByteCacheRuntime:
    """Thread-safe byte-capacity cache using a pluggable policy."""

    def __init__(
        self,
        policy: EvictionPolicy,
        origin_specs: Mapping[str, Any],
        origin_loader: Callable[[str], bytes],
        capacity_bytes: int = 24576,
        max_objects: int = 16,
        frequency_window: int = 100,
        recency_window: int = 1000,
        cost_scale_ms: float = 250.0,
        size_scale_bytes: float = 32768.0,
    ) -> None:
        if capacity_bytes < 1 or max_objects < 1:
            raise ValueError("Capacity settings must be positive")
        self.policy = policy
        self.origin_specs = origin_specs
        self.origin_loader = origin_loader
        self.capacity_bytes = capacity_bytes
        self.max_objects = max_objects
        self.frequency_window = frequency_window
        self.recency_window = recency_window
        self.cost_scale_ms = cost_scale_ms
        self.size_scale_bytes = size_scale_bytes
        self.entries: list[CacheEntry] = []
        self.payloads: dict[str, bytes] = {}
        self.used_bytes = 0
        self.position = 0
        self.frequency: dict[str, int] = {}
        self.access_history: list[str] = []
        self.lock = RLock()
        self.metrics = {
            "requests": 0,
            "hits": 0,
            "misses": 0,
            "bypasses": 0,
            "evictions": 0,
            "policy_time_ms": 0.0,
            "backend_latency_ms": 0.0,
        }

    def get(self, key: str) -> CacheResult:
        with self.lock:
            self.metrics["requests"] += 1
            entry = next((item for item in self.entries if item.key == key), None)
            self.position += 1
            if entry is not None:
                entry.frequency += 1
                entry.last_access = self.position
                self.metrics["hits"] += 1
                return CacheResult(
                    self.payloads[key], True, 0.0, 0.0, 0.0, (), self.used_bytes, False
                )

            spec = self.origin_specs[key]
            started = perf_counter_ns()
            payload = self.origin_loader(key)
            backend_ms = (perf_counter_ns() - started) / 1_000_000
            self.metrics["misses"] += 1
            evicted: list[str] = []
            policy_ms = 0.0
            if spec.size_bytes > self.capacity_bytes:
                self.metrics["bypasses"] += 1
                return CacheResult(
                    payload, False, spec.retrieval_cost_ms, backend_ms, 0.0, (),
                    self.used_bytes, True
                )

            catalog = {
                name: {"cost": value.retrieval_cost_ms, "size": value.size_bytes}
                for name, value in self.origin_specs.items()
            }
            while (
                self.used_bytes + spec.size_bytes > self.capacity_bytes
                or len(self.entries) >= self.max_objects
            ):
                context = {
                    "catalog": catalog,
                    "frequency_counts": dict(self.frequency),
                    "frequency_window": self.frequency_window,
                    "recency_window": self.recency_window,
                    "cost_scale_ms": self.cost_scale_ms,
                    "size_scale_bytes": self.size_scale_bytes,
                    "capacity_bytes": self.capacity_bytes,
                    "max_objects": self.max_objects,
                    "used_bytes": self.used_bytes,
                    "pending_cost_charged": bool(evicted),
                }
                start = perf_counter_ns()
                victim_key = self.policy.choose_victim(
                    self.entries,
                    {"key": key, "cost": spec.retrieval_cost_ms, "size": spec.size_bytes},
                    self.position,
                    context=context,
                )
                policy_ms += (perf_counter_ns() - start) / 1_000_000
                victim = next(item for item in self.entries if item.key == victim_key)
                self.entries.remove(victim)
                self.payloads.pop(victim.key, None)
                self.used_bytes -= victim.size_bytes
                evicted.append(victim.key)
                self.metrics["evictions"] += 1

            self.entries.append(
                CacheEntry(key, spec.retrieval_cost_ms, spec.size_bytes, 1, self.position)
            )
            self.payloads[key] = payload
            self.used_bytes += spec.size_bytes
            self.metrics["policy_time_ms"] += policy_ms
            self.metrics["backend_latency_ms"] += backend_ms
            self.frequency[key] = self.frequency.get(key, 0) + 1
            return CacheResult(
                payload, False, spec.retrieval_cost_ms, backend_ms, policy_ms,
                tuple(evicted), self.used_bytes, False
            )

    def snapshot_metrics(self) -> dict[str, Any]:
        with self.lock:
            return {**self.metrics, "used_bytes": self.used_bytes, "objects": len(self.entries)}
