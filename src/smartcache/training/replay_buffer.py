from __future__ import annotations

import random
from collections import deque

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        if capacity < 1:
            raise ValueError("Replay capacity must be positive")
        self.buffer = deque(maxlen=capacity)

    def add(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        next_action_mask,
    ) -> None:
        self.buffer.append(
            (
                np.array(state, dtype=np.float32, copy=True),
                int(action),
                float(reward),
                np.array(next_state, dtype=np.float32, copy=True),
                float(done),
                np.array(next_action_mask, dtype=np.bool_, copy=True),
            )
        )

    def sample(self, batch_size: int, device: torch.device):
        if batch_size < 1:
            raise ValueError("Batch size must be positive")
        if len(self.buffer) < batch_size:
            raise ValueError("Not enough transitions to sample")
        states, actions, rewards, next_states, dones, next_action_masks = zip(
            *random.sample(list(self.buffer), batch_size)
        )
        return (
            torch.as_tensor(np.asarray(states), dtype=torch.float32, device=device),
            torch.as_tensor(actions, dtype=torch.long, device=device),
            torch.as_tensor(rewards, dtype=torch.float32, device=device),
            torch.as_tensor(
                np.asarray(next_states), dtype=torch.float32, device=device
            ),
            torch.as_tensor(dones, dtype=torch.float32, device=device),
            torch.as_tensor(
                np.asarray(next_action_masks), dtype=torch.bool, device=device
            ),
        )

    def __len__(self) -> int:
        return len(self.buffer)
