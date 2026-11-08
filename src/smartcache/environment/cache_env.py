"""Cost-aware cache eviction environment."""

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from smartcache.environment.state_encoder import encode_state


class CacheEnv(gym.Env[np.ndarray, int]):
    """A fixed-slot cache environment for eviction-policy experiments.

    The observation is a float vector with this layout:

    * current request cost, size, and normalized request position;
    * for each cache slot: normalized recency, recent frequency, normalized
      retrieval cost, normalized object size, and occupancy;
    * normalized incoming-request cost, size, and recent frequency.

    Object sizes are exposed as features only.  The cache capacity is the
    number of slots and is not currently a byte-capacity constraint.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        requests: Sequence[Mapping[str, Any]],
        capacity: int,
        frequency_window: int = 100,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if frequency_window <= 0:
            raise ValueError("frequency_window must be positive")

        self.requests = [self._validate_request(request) for request in requests]
        self.catalog = {
            request["key"]: {
                "cost": float(request["cost"]),
                "size": float(request["size"]),
            }
            for request in self.requests
        }
        self.capacity = capacity
        self.frequency_window = frequency_window
        self.recency_window = max(1, len(self.requests))
        self.max_cost = max(
            (float(request["cost"]) for request in self.requests), default=1.0
        )
        self.max_size = max(
            (float(request["size"]) for request in self.requests), default=1.0
        )

        # Actions 0...capacity-1 select eviction slots.
        # The final action means no eviction is required.
        self.noop_action = capacity
        self.action_space = spaces.Discrete(capacity + 1)
        observation_size = 3 + capacity * 5
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_size,),
            dtype=np.float32,
        )

        self.cache: list[str | None] = [None] * capacity
        self.position = 0
        self.last_access: dict[str, int] = {}
        self.frequency: Counter[str] = Counter()
        self.access_history: list[str] = []

    def action_mask(self) -> np.ndarray:
        """Return the actions valid for the current request."""
        mask = np.zeros(self.capacity + 1, dtype=np.bool_)
        if self.position >= len(self.requests):
            mask[self.noop_action] = True
            return mask

        request = self.requests[self.position]
        key = str(request["key"])
        needs_eviction = key not in self.cache and None not in self.cache
        if needs_eviction:
            mask[: self.capacity] = True
        else:
            mask[self.noop_action] = True
        return mask

    @staticmethod
    def _validate_request(request: Mapping[str, Any]) -> dict[str, Any]:
        if "key" not in request or "cost" not in request or "size" not in request:
            raise ValueError("Each request must contain key, cost, and size")

        key = str(request["key"])
        if not key:
            raise ValueError("Request keys must not be empty")

        cost = float(request["cost"])
        size = float(request["size"])
        if not np.isfinite(cost) or cost < 0:
            raise ValueError("Request cost must be a finite non-negative number")
        if not np.isfinite(size) or size <= 0:
            raise ValueError("Request size must be a finite positive number")

        return {"key": key, "cost": cost, "size": size}

    def _observation(self) -> np.ndarray:
        if self.position >= len(self.requests):
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        request = self.requests[self.position]
        return encode_state(
            slot_keys=self.cache,
            incoming_key=str(request["key"]),
            catalog=self.catalog,
            last_access=self.last_access,
            frequency_counts=self.frequency,
            position=self.position,
            frequency_window=self.frequency_window,
            recency_window=self.recency_window,
            max_cost=self.max_cost,
            max_size=self.max_size,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.cache = [None] * self.capacity
        self.position = 0
        self.last_access.clear()
        self.frequency.clear()
        self.access_history.clear()
        return self._observation(), {"cache": list(self.cache)}

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.position >= len(self.requests):
            raise RuntimeError("Episode finished. Call reset() before step().")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        valid_mask = self.action_mask()
        if not valid_mask[action]:
            raise ValueError(f"Action {action} is invalid for the current state")

        request = self.requests[self.position]
        key = str(request["key"])
        cost = float(request["cost"])
        hit = key in self.cache
        evicted = None
        reward = 0.0 if hit else -cost

        if not hit:
            if None in self.cache:
                slot = self.cache.index(None)
            else:
                slot = int(action)
                evicted = self.cache[slot]
            self.cache[slot] = key

        self.position += 1
        self.last_access[key] = self.position
        self.frequency[key] += 1
        self.access_history.append(key)

        if len(self.access_history) > self.frequency_window:
            expired = self.access_history.pop(0)
            self.frequency[expired] -= 1
            if self.frequency[expired] <= 0:
                del self.frequency[expired]

        terminated = self.position >= len(self.requests)
        truncated = False
        info = {
            "key": key,
            "hit": hit,
            "miss_cost_ms": 0.0 if hit else cost,
            "evicted": evicted,
            "cache": list(self.cache),
        }
        observation = (
            np.zeros(self.observation_space.shape, dtype=np.float32)
            if terminated
            else self._observation()
        )
        return observation, reward, terminated, truncated, info
