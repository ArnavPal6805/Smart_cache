from typing import Any, Mapping, Sequence

from smartcache.core.interfaces import CacheEntry


class LRUPolicy:
    name = "LRU"

    def choose_victim(
        self,
        cache: Sequence[CacheEntry],
        incoming: Mapping[str, object],
        timestamp: int,
        context: Mapping[str, Any] | None = None,
    ) -> str:
        if not cache:
            raise ValueError("Cannot evict from an empty cache")
        return min(cache, key=lambda entry: entry.last_access).key
