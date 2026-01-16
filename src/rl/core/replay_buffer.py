# src/rl/core/replay_buffer.py
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


@dataclass
class ReplayBatch:
    obs: list[Any]
    actions: list[Any]
    rewards: list[float]
    next_obs: list[Any]
    dones: list[bool]


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self._data: list[dict[str, Any]] = []
        self._idx = 0

    def add(self, *, obs, action, reward, next_obs, done: bool) -> None:
        item = {
            "obs": obs,
            "action": action,
            "reward": float(reward),
            "next_obs": next_obs,
            "done": bool(done),
        }
        if len(self._data) < self.capacity:
            self._data.append(item)
        else:
            self._data[self._idx] = item
            self._idx = (self._idx + 1) % self.capacity

    def __len__(self) -> int:
        return len(self._data)

    def sample(self, batch_size: int) -> ReplayBatch:
        if len(self._data) == 0:
            raise ValueError("Cannot sample from an empty ReplayBuffer.")
        bs = min(int(batch_size), len(self._data))
        batch = random.sample(self._data, bs)
        return ReplayBatch(
            obs=[b["obs"] for b in batch],
            actions=[b["action"] for b in batch],
            rewards=[b["reward"] for b in batch],
            next_obs=[b["next_obs"] for b in batch],
            dones=[b["done"] for b in batch],
        )
