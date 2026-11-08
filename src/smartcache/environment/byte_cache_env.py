from __future__ import annotations

from collections import Counter
from math import isfinite
from typing import Any, Mapping, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from smartcache.core.interfaces import CacheEntry
from smartcache.environment.byte_state_encoder import encode_byte_state


class ByteCacheEnv(gym.Env[np.ndarray, int]):
    """Byte-capacity cache replacement environment."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        requests: Sequence[Mapping[str, Any]],
        capacity_bytes: int,
        max_objects: int = 16,
        frequency_window: int = 100,
        recency_window: int = 1000,
        cost_scale_ms: float = 250.0,
        size_scale_bytes: float = 32768.0,
    ) -> None:
        if capacity_bytes < 1 or max_objects < 1:
            raise ValueError("Capacity and max_objects must be positive")
        if frequency_window < 1 or recency_window < 1:
            raise ValueError("Window sizes must be positive")
        if cost_scale_ms <= 0 or size_scale_bytes <= 0:
            raise ValueError("Feature scales must be positive")
        self.requests = self._validate_trace(requests)
        self.capacity_bytes = capacity_bytes
        self.max_objects = max_objects
        self.frequency_window = frequency_window
        self.recency_window = recency_window
        self.cost_scale_ms = cost_scale_ms
        self.size_scale_bytes = size_scale_bytes
        self.catalog = {
            request["key"]: {
                "cost": float(request["cost"]),
                "size": float(request["size"]),
            }
            for request in self.requests
        }
        self.noop_action = max_objects
        self.action_space = spaces.Discrete(max_objects + 1)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(max_objects * 5 + 5,),
            dtype=np.float32,
        )
        self.entries: list[CacheEntry] = []
        self.position = 0
        self.used_bytes = 0
        self.frequency: Counter[str] = Counter()
        self.access_history: list[str] = []
        self.pending_cost_charged = False

    @staticmethod
    def _validate_trace(
        requests: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        if not requests:
            raise ValueError("Request trace must not be empty")
        result: list[dict[str, Any]] = []
        catalogue: dict[str, tuple[float, int]] = {}
        for index, request in enumerate(requests):
            if not {"key", "cost", "size"} <= request.keys():
                raise ValueError(f"Request {index} requires key, cost, and size")
            key = str(request["key"])
            cost = float(request["cost"])
            size_value = float(request["size"])
            if not key or not isfinite(cost) or cost < 0:
                raise ValueError(f"Invalid request {index}")
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
            result.append({"key": key, "cost": cost, "size": size})
        return result

    def _ordered_entries(self) -> list[CacheEntry]:
        return sorted(self.entries, key=lambda entry: (entry.last_access, entry.key))

    def _current_request(self) -> dict[str, Any]:
        return self.requests[self.position]

    def _request_needs_eviction(self) -> bool:
        request = self._current_request()
        return (
            request["key"] not in {entry.key for entry in self.entries}
            and request["size"] <= self.capacity_bytes
            and (
                self.used_bytes + request["size"] > self.capacity_bytes
                or len(self.entries) >= self.max_objects
            )
        )

    def _request_complete_without_eviction(self) -> bool:
        request = self._current_request()
        return (
            request["key"] in {entry.key for entry in self.entries}
            or request["size"] > self.capacity_bytes
            or (
                self.used_bytes + request["size"] <= self.capacity_bytes
                and len(self.entries) < self.max_objects
            )
        )

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_space.n, dtype=np.bool_)
        if self.position >= len(self.requests) or self._request_complete_without_eviction():
            mask[self.noop_action] = True
        else:
            mask[: len(self.entries)] = True
        return mask

    def _observation(self) -> np.ndarray:
        if self.position >= len(self.requests):
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        return encode_byte_state(
            entries=self._ordered_entries(),
            incoming_key=self._current_request()["key"],
            catalog=self.catalog,
            frequency_counts=self.frequency,
            position=self.position,
            capacity_bytes=self.capacity_bytes,
            used_bytes=self.used_bytes,
            max_objects=self.max_objects,
            frequency_window=self.frequency_window,
            recency_window=self.recency_window,
            cost_scale_ms=self.cost_scale_ms,
            size_scale_bytes=self.size_scale_bytes,
            pending_cost_charged=self.pending_cost_charged,
        )

    def _finish_request(self, admit: bool) -> None:
        request = self._current_request()
        if admit:
            entry = CacheEntry(
                key=request["key"],
                retrieval_cost_ms=request["cost"],
                size_bytes=request["size"],
                frequency=1,
                last_access=self.position + 1,
            )
            self.entries.append(entry)
            self.used_bytes += request["size"]
        self.position += 1
        self.frequency[request["key"]] += 1
        self.access_history.append(request["key"])
        if len(self.access_history) > self.frequency_window:
            expired = self.access_history.pop(0)
            self.frequency[expired] -= 1
            if self.frequency[expired] <= 0:
                del self.frequency[expired]
        self.pending_cost_charged = False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.entries.clear()
        self.position = 0
        self.used_bytes = 0
        self.frequency.clear()
        self.access_history.clear()
        self.pending_cost_charged = False
        return self._observation(), {"cache": []}

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.position >= len(self.requests):
            raise RuntimeError("Episode finished. Call reset() before step().")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        mask = self.action_mask()
        if not mask[action]:
            raise ValueError(f"Action {action} is invalid for the current state")

        request = self._current_request()
        hit = request["key"] in {entry.key for entry in self.entries}
        reward = 0.0
        evicted = None
        request_complete = False

        if action == self.noop_action:
            if hit:
                entry = next(item for item in self.entries if item.key == request["key"])
                entry.frequency += 1
                entry.last_access = self.position + 1
                self._finish_request(admit=False)
            elif request["size"] > self.capacity_bytes:
                reward = 0.0 if self.pending_cost_charged else -request["cost"]
                self._finish_request(admit=False)
            else:
                reward = 0.0 if self.pending_cost_charged else -request["cost"]
                self._finish_request(admit=True)
            request_complete = True
        else:
            ordered = self._ordered_entries()
            victim = ordered[action]
            self.entries.remove(victim)
            self.used_bytes -= victim.size_bytes
            evicted = victim.key
            reward = 0.0 if self.pending_cost_charged else -request["cost"]
            self.pending_cost_charged = True
            if not self._request_needs_eviction():
                self._finish_request(admit=True)
                request_complete = True

        terminated = self.position >= len(self.requests)
        info = {
            "key": request["key"],
            "hit": hit,
            "evicted": evicted,
            "request_complete": request_complete,
            "miss_cost_ms": -reward if reward < 0 else 0.0,
            "used_bytes": self.used_bytes,
            "cache": [entry.key for entry in self._ordered_entries()],
        }
        return self._observation(), reward, terminated, False, info
