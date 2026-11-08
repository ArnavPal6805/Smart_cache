from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from smartcache.core.interfaces import CacheEntry
from smartcache.environment.byte_state_encoder import encode_byte_state
from smartcache.training.dqn_agent import DQNAgent


class DQNBytePolicy:
    name = "DQN-Byte"

    def __init__(
        self,
        checkpoint_path: str | Path,
        capacity_bytes: int,
        max_objects: int = 16,
        frequency_window: int = 100,
        recency_window: int = 1000,
        device: str = "cpu",
    ):
        self.capacity_bytes = capacity_bytes
        self.max_objects = max_objects
        self.frequency_window = frequency_window
        self.recency_window = recency_window
        self.agent, self.metadata = DQNAgent.load_checkpoint(
            checkpoint_path, device=device
        )
        expected_state_dim = max_objects * 5 + 5
        if self.agent.state_dim != expected_state_dim:
            raise ValueError("Checkpoint state dimension mismatch")
        if self.agent.action_dim != max_objects + 1:
            raise ValueError("Checkpoint action dimension mismatch")
        if self.metadata.get("feature_version") != "byte_cache_v2":
            raise ValueError("Expected a byte_cache_v2 checkpoint")
        if self.metadata.get("max_objects") != max_objects:
            raise ValueError("Checkpoint max_objects mismatch")
        if self.metadata.get("capacity_bytes") != capacity_bytes:
            raise ValueError("Checkpoint capacity mismatch")
        if self.metadata.get("frequency_window") != frequency_window:
            raise ValueError("Checkpoint frequency_window mismatch")
        if self.metadata.get("recency_window") != recency_window:
            raise ValueError("Checkpoint recency_window mismatch")
        if float(self.metadata.get("cost_scale_ms", 0.0)) <= 0:
            raise ValueError("Checkpoint cost scale missing or invalid")
        if float(self.metadata.get("size_scale_bytes", 0.0)) <= 0:
            raise ValueError("Checkpoint size scale missing or invalid")

    def choose_victim(
        self,
        cache: Sequence[CacheEntry],
        incoming: Mapping[str, object],
        timestamp: int,
        context: Mapping[str, Any] | None = None,
    ) -> str:
        if not cache:
            raise ValueError("Cannot evict from an empty cache")
        if len(cache) > self.max_objects:
            raise ValueError("Cache contains too many objects")
        if context is None:
            raise ValueError("DQN-Byte policy requires simulator context")

        entries = sorted(cache, key=lambda item: (item.last_access, item.key))
        state = encode_byte_state(
            entries=entries,
            incoming_key=str(incoming["key"]),
            catalog=context["catalog"],
            frequency_counts=context["frequency_counts"],
            position=timestamp,
            capacity_bytes=self.capacity_bytes,
            used_bytes=int(context["used_bytes"]),
            max_objects=self.max_objects,
            frequency_window=int(context["frequency_window"]),
            recency_window=int(context["recency_window"]),
            cost_scale_ms=float(self.metadata["cost_scale_ms"]),
            size_scale_bytes=float(self.metadata["size_scale_bytes"]),
            pending_cost_charged=bool(context["pending_cost_charged"]),
        )
        valid_actions = [True] * len(entries) + [False] * (
            self.max_objects + 1 - len(entries)
        )
        action = self.agent.act(state, epsilon=0.0, action_mask=valid_actions)
        if not 0 <= action < len(entries):
            raise ValueError(f"Invalid byte eviction action: {action}")
        return entries[action].key
