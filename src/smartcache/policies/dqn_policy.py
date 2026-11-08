from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from smartcache.core.interfaces import CacheEntry
from smartcache.environment.state_encoder import encode_state
from smartcache.training.dqn_agent import DQNAgent


class DQNPolicy:
    name = "DQN"

    def __init__(
        self,
        checkpoint_path: str | Path,
        capacity: int,
        device: str = "cpu",
    ):
        self.capacity = capacity
        self.agent, metadata = DQNAgent.load_checkpoint(
            checkpoint_path, device=device
        )
        self.metadata = metadata
        expected_state_dim = capacity * 5 + 3
        if self.agent.state_dim != expected_state_dim:
            raise ValueError("Checkpoint state dimension mismatch")
        if self.agent.action_dim != capacity + 1:
            raise ValueError("Checkpoint action dimension mismatch")

    def choose_victim(
        self,
        cache: Sequence[CacheEntry],
        incoming: Mapping[str, object],
        timestamp: int,
        context: Mapping[str, Any] | None = None,
    ) -> str:
        if not cache:
            raise ValueError("Cannot evict from an empty cache")
        if len(cache) != self.capacity:
            raise ValueError("DQN decisions require a full cache")
        if context is None:
            raise ValueError("DQN policy requires simulator context")

        slot_keys = [entry.key for entry in cache]
        state = encode_state(
            slot_keys=slot_keys,
            incoming_key=str(incoming["key"]),
            catalog=context["catalog"],
            last_access=context["last_access"],
            frequency_counts=context["frequency_counts"],
            position=timestamp,
            frequency_window=int(context["frequency_window"]),
            recency_window=int(context["recency_window"]),
            max_cost=float(context["max_cost"]),
            max_size=float(context["max_size"]),
        )
        valid_actions = [True] * self.capacity + [False]
        action = self.agent.act(
            state,
            epsilon=0.0,
            action_mask=valid_actions,
        )
        if not 0 <= action < self.capacity:
            raise ValueError(f"Invalid eviction action: {action}")
        return cache[action].key
