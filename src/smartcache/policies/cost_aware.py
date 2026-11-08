from math import exp, log1p, sqrt
from typing import Any, Mapping, Sequence

from smartcache.core.interfaces import CacheEntry


class CostAwarePolicy:
    name = "CostAware"

    def __init__(self, recency_tau: float = 20.0):
        if recency_tau <= 0:
            raise ValueError("recency_tau must be positive")
        self.recency_tau = recency_tau

    def utility(self, entry: CacheEntry, timestamp: int) -> float:
        age = max(0, timestamp - entry.last_access)
        cost_value = log1p(entry.retrieval_cost_ms)
        frequency_value = log1p(entry.frequency)
        recency_weight = exp(-age / self.recency_tau)
        size_penalty = sqrt(max(1, entry.size_bytes))
        return cost_value * frequency_value * recency_weight / size_penalty

    def choose_victim(
        self,
        cache: Sequence[CacheEntry],
        incoming: Mapping[str, object],
        timestamp: int,
        context: Mapping[str, Any] | None = None,
    ) -> str:
        if not cache:
            raise ValueError("Cannot evict from an empty cache")
        return min(cache, key=lambda entry: self.utility(entry, timestamp)).key
