from __future__ import annotations

from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.layers(state)


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        target_update_interval: int = 250,
        reward_scale_ms: float = 250.0,
        device: str = "cpu",
    ):
        if state_dim < 1 or action_dim < 1:
            raise ValueError("state_dim and action_dim must be positive")
        if not 0 < gamma <= 1 or reward_scale_ms <= 0:
            raise ValueError("gamma and reward scale are invalid")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.target_update_interval = target_update_interval
        self.reward_scale_ms = reward_scale_ms
        self.device = torch.device(device)
        self.online = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.online_net = self.online
        self.target_net = self.target
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = AdamW(self.online.parameters(), lr=learning_rate)
        self.update_count = 0

    @torch.no_grad()
    def act(self, state, epsilon: float = 0.0, action_mask=None) -> int:
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be between 0 and 1")
        if action_mask is None:
            valid = np.ones(self.action_dim, dtype=np.bool_)
        else:
            valid = np.asarray(action_mask, dtype=np.bool_)
        if valid.shape != (self.action_dim,):
            raise ValueError("Action mask has an invalid shape")
        valid_actions = np.flatnonzero(valid)
        if len(valid_actions) == 0:
            raise ValueError("No valid actions available")
        if random.random() < epsilon:
            return int(random.choice(valid_actions.tolist()))
        state_tensor = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        q_values = self.online_net(state_tensor).squeeze(0)
        valid_tensor = torch.as_tensor(valid, dtype=torch.bool, device=self.device)
        q_values = q_values.masked_fill(~valid_tensor, float("-inf"))
        return int(q_values.argmax().item())

    def learn(self, batch, reward_scale_ms: float = 250.0) -> float:
        if reward_scale_ms <= 0:
            raise ValueError("Reward scale must be positive")
        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            next_action_masks,
        ) = batch
        current = self.online_net(states).gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)
        with torch.no_grad():
            next_online_q = self.online_net(next_states)
            next_online_q = next_online_q.masked_fill(
                ~next_action_masks, float("-inf")
            )
            next_actions = next_online_q.argmax(dim=1)
            next_q = self.target_net(next_states).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)
            scaled_rewards = rewards / reward_scale_ms
            target = scaled_rewards + self.gamma * next_q * (1.0 - dones)
        loss = nn.functional.smooth_l1_loss(current, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        self.update_count += 1
        if self.update_count % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
        return float(loss.item())

    def select_action(self, state, epsilon: float = 0.0, action_mask=None) -> int:
        return self.act(state, epsilon=epsilon, action_mask=action_mask)

    def update(self, batch) -> float:
        return self.learn(batch, reward_scale_ms=self.reward_scale_ms)

    def save_checkpoint(self, path: str | Path, metadata: dict[str, Any]) -> None:
        checkpoint = {
            "model": self.online.state_dict(),
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "gamma": self.gamma,
            "reward_scale_ms": self.reward_scale_ms,
            "metadata": metadata,
        }
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, destination)

    @classmethod
    def load_checkpoint(cls, path: str | Path, device: str = "cpu"):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        agent = cls(
            state_dim=checkpoint["state_dim"],
            action_dim=checkpoint["action_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            gamma=checkpoint["gamma"],
            reward_scale_ms=checkpoint["reward_scale_ms"],
            device=device,
        )
        agent.online.load_state_dict(checkpoint["model"])
        agent.target.load_state_dict(checkpoint["model"])
        return agent, checkpoint.get("metadata", {})
