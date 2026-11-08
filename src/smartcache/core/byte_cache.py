from __future__ import annotations

from collections import Counter
from math import isfinite
from time import perf_counter_ns
from typing import Any, Mapping, Sequence

from smartcache.core.interfaces import CacheEntry, EvictionPolicy


class ByteCacheSimulator:
    """Cache simulator with a byte-capacity budget."""

    def __init__(
        self,
        capacity_bytes: int,
        max_objects: int = 64,
        frequency_window: int = 100,
        recency_window: int = 1000,
        cost_scale_ms: float = 250.0,
        size_scale_bytes: float = 32768.0,
    ):
        if capacity_bytes < 1:
            raise ValueError("capacity_bytes must be positive")
        if max_objects < 1:
            raise ValueError("max_objects must be positive")
        if frequency_window < 1 or recency_window < 1:
            raise ValueError("Window sizes must be positive")
        if cost_scale_ms <= 0 or size_scale_bytes <= 0:
            raise ValueError("Feature scales must be positive")
        self.capacity_bytes = capacity_bytes
        self.max_objects = max_objects
        self.frequency_window = frequency_window
        self.recency_window = recency_window
        self.cost_scale_ms = cost_scale_ms
        self.size_scale_bytes = size_scale_bytes

    @staticmethod
    def _validate_trace(
        requests: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        if not requests:
            raise ValueError("Request trace must not be empty")

        normalized: list[dict[str, Any]] = []
        catalogue: dict[str, tuple[float, int]] = {}
        for index, request in enumerate(requests):
            if not {"key", "cost", "size"} <= request.keys():
                raise ValueError(f"Request {index} requires key, cost, and size")
            key = str(request["key"])
            cost = float(request["cost"])
            size_value = float(request["size"])
            if not key:
                raise ValueError(f"Request {index} key must not be empty")
            if not isfinite(cost) or cost < 0:
                raise ValueError(
                    f"Request {index} cost must be finite and non-negative"
                )
            if (
                not isfinite(size_value)
                or size_value <= 0
                or not size_value.is_integer()
            ):
                raise ValueError(f"Request {index} size must be a positive integer")
            size = int(size_value)
            metadata = (cost, size)
            if key in catalogue and catalogue[key] != metadata:
                raise ValueError(f"Request {index} metadata differs for key {key!r}")
            catalogue[key] = metadata
            normalized.append({"key": key, "cost": cost, "size": size})
        return normalized

    def run(
        self,
        requests: Sequence[Mapping[str, Any]],
        policy: EvictionPolicy,
    ) -> dict[str, Any]:
        trace = self._validate_trace(requests)
        cache: list[CacheEntry] = []
        hits = 0
        evictions = 0
        total_miss_cost_ms = 0.0
        total_policy_time_ns = 0
        current_bytes = 0
        peak_bytes = 0
        last_access: dict[str, int] = {}
        frequency: Counter[str] = Counter()
        access_history: list[str] = []
        catalogue = {
            request["key"]: {
                "cost": float(request["cost"]),
                "size": float(request["size"]),
            }
            for request in trace
        }

        for timestamp, request in enumerate(trace, start=1):
            key = request["key"]
            size = request["size"]
            cost = request["cost"]
            entry = next((item for item in cache if item.key == key), None)

            if entry is not None:
                hits += 1
                entry.frequency += 1
                entry.last_access = timestamp
            else:
                total_miss_cost_ms += cost
                if size <= self.capacity_bytes:
                    pending_cost_charged = False
                    while (
                        current_bytes + size > self.capacity_bytes
                        or len(cache) >= self.max_objects
                    ):
                        if not cache:
                            raise ValueError(
                                "Cache cannot make room for a fitting object"
                            )
                        context = {
                            "catalog": catalogue,
                            "frequency_counts": dict(frequency),
                            "frequency_window": self.frequency_window,
                            "recency_window": self.recency_window,
                            "cost_scale_ms": self.cost_scale_ms,
                            "size_scale_bytes": self.size_scale_bytes,
                            "capacity_bytes": self.capacity_bytes,
                            "max_objects": self.max_objects,
                            "used_bytes": current_bytes,
                            "pending_cost_charged": pending_cost_charged,
                        }
                        start = perf_counter_ns()
                        victim_key = policy.choose_victim(
                            cache,
                            request,
                            timestamp,
                            context=context,
                        )
                        total_policy_time_ns += perf_counter_ns() - start
                        victim = next(
                            (item for item in cache if item.key == victim_key),
                            None,
                        )
                        if victim is None:
                            raise ValueError(
                                f"Policy {policy.name!r} selected a key not in "
                                f"the cache: {victim_key!r}"
                            )
                        cache.remove(victim)
                        current_bytes -= victim.size_bytes
                        evictions += 1
                        pending_cost_charged = True

                    cache.append(
                        CacheEntry(
                            key=key,
                            retrieval_cost_ms=cost,
                            size_bytes=size,
                            frequency=1,
                            last_access=timestamp,
                        )
                    )
                    current_bytes += size
                    peak_bytes = max(peak_bytes, current_bytes)

            last_access[key] = timestamp
            frequency[key] += 1
            access_history.append(key)
            if len(access_history) > self.frequency_window:
                expired = access_history.pop(0)
                frequency[expired] -= 1
                if frequency[expired] <= 0:
                    del frequency[expired]

        misses = len(trace) - hits
        policy_time_ms = total_policy_time_ns / 1_000_000
        return {
            "requests": len(trace),
            "hits": hits,
            "misses": misses,
            "hit_ratio": hits / len(trace),
            "evictions": evictions,
            "total_miss_cost_ms": total_miss_cost_ms,
            "policy_time_ns": total_policy_time_ns,
            "policy_time_ms": policy_time_ms,
            "estimated_net_cost_ms": total_miss_cost_ms + policy_time_ms,
            "peak_bytes": peak_bytes,
            "final_bytes": current_bytes,
            "final_cache": [entry.key for entry in cache],
        }
