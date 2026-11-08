from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

from smartcache.core.interfaces import CacheEntry

logger = logging.getLogger(__name__)


class SafePolicy:
    """Use a deterministic fallback when the primary policy fails."""

    def __init__(self, primary: Any, fallback: Any):
        self.primary = primary
        self.fallback = fallback
        self.name = getattr(primary, "name", type(primary).__name__)
        self.metadata = getattr(primary, "metadata", None)
        self.fallback_errors = 0

    def choose_victim(
        self,
        cache: Sequence[CacheEntry],
        incoming: Mapping[str, object],
        timestamp: int,
        context: Mapping[str, Any] | None = None,
    ) -> str:
        try:
            return self.primary.choose_victim(
                cache, incoming, timestamp, context=context
            )
        except (ValueError, KeyError, RuntimeError) as error:
            self.fallback_errors += 1
            logger.exception("Primary cache policy failed; using fallback: %s", error)
            return self.fallback.choose_victim(
                cache, incoming, timestamp, context=context
            )
