from __future__ import annotations

from collections import Counter
from math import isfinite
from time import perf_counter_ns
from typing import Any, Mapping, Sequence

from smartcache.core.interfaces import CacheEntry, EvictionPolicy


class CacheSimulator:
    """Deterministic request-trace simulator with selectable policies."""

    def __init__(
        self,
        capacity: int = 32,
        frequency_window: int = 100,
        recency_window: int = 1000,
    ):
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        if frequency_window < 1 or recency_window < 1:
            raise ValueError("Window sizes must be positive")
        self.capacity = capacity
        self.frequency_window = frequency_window
        self.recency_window = recency_window

    @staticmethod
    def _validate_trace(
        requests: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        if not requests:
            raise ValueError("Request trace must not be empty")

        normalized: list[dict[str, Any]] = []
        catalogue: dict[str, tuple[float, int]] = {}
        for i, request in enumerate(requests):
            if not {"key", "cost", "size"} <= request.keys():
                raise ValueError(f"Request {i} requires key, cost, and size")
            key = str(request["key"])
            cost = float(request["cost"])
            size_value = float(request["size"])
            if not key:
                raise ValueError(f"Request {i} key must not be empty")
            if not isfinite(cost) or cost < 0:
                raise ValueError(f"Request {i} cost must be finite and non-negative")
            if not isfinite(size_value) or size_value <= 0 or not size_value.is_integer():
                raise ValueError(f"Request {i} size must be a positive integer")
            size = int(size_value)

            previous = catalogue.get(key)
            if previous is not None and previous != (cost, size):
                raise ValueError(f"Request {i} metadata differs for key {key!r}")
            catalogue[key] = (cost, size)
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
        last_access: dict[str, int] = {}
        frequency: Counter[str] = Counter()
        access_history: list[str] = []
        catalog = {
            request["key"]: {
                "cost": float(request["cost"]),
                "size": float(request["size"]),
            }
            for request in trace
        }
        max_cost = max(float(request["cost"]) for request in trace)
        max_size = max(float(request["size"]) for request in trace)

        for timestamp, request in enumerate(trace, start=1):
            key = request["key"]
            matching_entry = next((entry for entry in cache if entry.key == key), None)
            if matching_entry is not None:
                hits += 1
                matching_entry.frequency += 1
                matching_entry.last_access = timestamp
            else:
                total_miss_cost_ms += request["cost"]
                if len(cache) >= self.capacity:
                    context = {
                        "catalog": catalog,
                        "last_access": last_access,
                        "frequency_counts": frequency,
                        "frequency_window": self.frequency_window,
                        "recency_window": self.recency_window,
                        "max_cost": max_cost,
                        "max_size": max_size,
                    }
                    start = perf_counter_ns()
                    victim_key = policy.choose_victim(
                        cache, request, timestamp, context=context
                    )
                    total_policy_time_ns += perf_counter_ns() - start
                    if not any(entry.key == victim_key for entry in cache):
                        raise ValueError(
                            f"Policy {policy.name!r} selected a key not in the cache: "
                            f"{victim_key!r}"
                        )
                    cache.remove(
                        next(entry for entry in cache if entry.key == victim_key)
                    )
                    evictions += 1

                cache.append(
                    CacheEntry(
                        key=key,
                        retrieval_cost_ms=request["cost"],
                        size_bytes=request["size"],
                        frequency=1,
                        last_access=timestamp,
                    )
                )

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
            "final_cache": [entry.key for entry in cache],
        }
