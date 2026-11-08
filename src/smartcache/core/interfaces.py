from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence


@dataclass(slots=True)
class CacheEntry:
    key: str
    retrieval_cost_ms: float
    size_bytes: int
    frequency: int
    last_access: int


class EvictionPolicy(Protocol):
    name: str

    def choose_victim(
        self,
        cache: Sequence[CacheEntry],
        incoming: Mapping[str, object],
        timestamp: int,
        context: Mapping[str, Any] | None = None,
    ) -> str:
        """Return the key of the object to evict."""
        ...
