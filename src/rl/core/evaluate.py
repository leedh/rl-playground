# src/rl/core/evaluate.py
from __future__ import annotations

from pathlib import Path
from typing import Any

from rl.core.seed import seed_everything
from rl.envs.make_env import make_env


class Evaluator:
    def __init__(self, *, cfg: dict[str, Any], run_dir: Path) -> None:
        self.cfg = cfg
        self.run_dir = run_dir

        train_cfg = self.cfg.get("train", {}) or {}
        self.base_seed = int(train_cfg.get("seed", 0)) + 10_000  # separate stream

    def evaluate(self, agent: Any, *, step: int, n_episodes: int) -> dict[str, float]:
        seed_everything(self.base_seed, deterministic_torch=True)
        env = make_env(self.cfg, mode="eval")

        returns = []
        lengths = []

        for ep in range(n_episodes):
            obs, _ = env.reset(seed=self.base_seed + ep)
            done = False
            ep_ret = 0.0
            ep_len = 0

            while not done:
                act_out = agent.act(obs, deterministic=True)
                action = act_out["action"] if isinstance(act_out, dict) else act_out
                obs, reward, terminated, truncated, _ = env.step(action)
                done = bool(terminated or truncated)
                ep_ret += float(reward)
                ep_len += 1

            returns.append(ep_ret)
            lengths.append(ep_len)

        env.close()

        mean_ret = sum(returns) / max(1, len(returns))
        mean_len = sum(lengths) / max(1, len(lengths))
        # cheap std
        var = sum((r - mean_ret) ** 2 for r in returns) / max(1, len(returns))
        std_ret = var**0.5

        return {
            "eval/return_mean": float(mean_ret),
            "eval/return_std": float(std_ret),
            "eval/episode_len_mean": float(mean_len),
        }
