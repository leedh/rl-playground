# src/rl/core/trainer.py
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from rl.core.logger import make_logger
from rl.core.replay_buffer import ReplayBuffer
from rl.core.rollout import collect_trajectory


@dataclass
class TrainerState:
    step: int = 0
    best_eval_return: float = float("-inf")
    start_time: float = 0.0


class Trainer:
    """
    Shared training loop.
    - On-policy: collect trajectories (rollout_steps) -> agent.update(trajectory)
    - Off-policy: step env, push to replay -> periodic updates with sampled batches

    Contract:
      - agent.training_mode in {"on_policy", "off_policy"}
      - agent.act(obs, deterministic=bool) -> action (and optionally extra)
      - agent.update(batch_or_trajectory) -> dict of scalar metrics
      - agent.state_dict() / load_state_dict()
    """

    def __init__(
        self,
        *,
        env: Any,
        agent: Any,
        evaluator: Any,
        logger: Any,
        config: dict[str, Any],
        run_dir: Path,
    ) -> None:
        self.env = env
        self.agent = agent
        self.evaluator = evaluator
        self.cfg = config
        self.run_dir = run_dir
        if logger is not None:
            self.logger = logger
        else:
            self.logger = make_logger(cfg=self.cfg, run_dir=self.run_dir)

        self.train_cfg = self.cfg.get("train", {}) or {}
        self.algo_cfg = self.cfg.get("algo", {}) or {}

        self.total_steps = int(self.train_cfg.get("total_steps", 100_000))
        self.log_every = int(self.train_cfg.get("log_every", 1000))
        self.eval_every = int(self.train_cfg.get("eval_every", 10_000))
        self.eval_episodes = int(self.train_cfg.get("eval_episodes", 10))
        self.save_every = int(self.train_cfg.get("save_every", 50_000))
        self.save_best = bool(self.train_cfg.get("save_best", True))
        self.time_limit_minutes = self.train_cfg.get("time_limit_minutes", None)

        self.state = TrainerState(step=0, best_eval_return=float("-inf"), start_time=time.time())

        # Off-policy components
        self.replay: ReplayBuffer | None = None
        if getattr(self.agent, "training_mode", None) == "off_policy":
            cap = int(self.algo_cfg.get("replay_capacity", 100_000))
            self.replay = ReplayBuffer(capacity=cap)

    def run(self) -> dict[str, Any]:
        self.state.start_time = time.time()

        # Reset env
        obs, _ = self.env.reset()

        while self.state.step < self.total_steps:
            # Time limit (optional)
            if self.time_limit_minutes is not None:
                elapsed_min = (time.time() - self.state.start_time) / 60.0
                if elapsed_min >= float(self.time_limit_minutes):
                    break

            mode = getattr(self.agent, "training_mode", "on_policy")
            if mode == "on_policy":
                obs = self._step_on_policy(obs)
            elif mode == "off_policy":
                obs = self._step_off_policy(obs)
            else:
                raise ValueError(f"Unknown agent.training_mode='{mode}'")

            # periodic eval/log/save hooks based on env steps
            self._maybe_log()
            self._maybe_eval()
            self._maybe_save()

        # Final evaluation
        final_eval = self.evaluator.evaluate(
            self.agent, step=self.state.step, n_episodes=self.eval_episodes
        )
        self.logger.log(final_eval, step=self.state.step)

        return {
            "total_steps": self.state.step,
            "best_eval_return": self.state.best_eval_return,
            "walltime_sec": time.time() - self.state.start_time,
        }

    def _step_on_policy(self, obs):
        rollout_steps = int(self.algo_cfg.get("rollout_steps", 2048))
        traj, last_obs = collect_trajectory(
            env=self.env,
            agent=self.agent,
            start_obs=obs,
            rollout_steps=rollout_steps,
        )
        self.state.step += traj.num_steps
        metrics = self.agent.update(traj)
        self._log_train_metrics(metrics, fps_hint=traj.num_steps)
        return last_obs

    def _step_off_policy(self, obs):
        assert self.replay is not None, "ReplayBuffer must be initialized for off-policy training."

        learning_starts = int(self.algo_cfg.get("learning_starts", 1000))
        update_every = int(self.algo_cfg.get("update_every", 1))
        num_updates = int(self.algo_cfg.get("num_updates", 1))
        batch_size = int(self.algo_cfg.get("batch_size", 64))

        # 1 env step
        act_out = self.agent.act(obs, deterministic=False)
        action = act_out["action"] if isinstance(act_out, dict) else act_out
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        done = bool(terminated or truncated)

        self.replay.add(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done)

        self.state.step += 1

        if done:
            self.agent.on_episode_end(info if isinstance(info, dict) else {})
            next_obs, _ = self.env.reset()

        # updates
        if self.state.step >= learning_starts and (self.state.step % update_every == 0):
            all_metrics: dict[str, float] = {}
            for _ in range(num_updates):
                batch = self.replay.sample(batch_size)
                metrics = self.agent.update(batch)
                for k, v in metrics.items():
                    # mean over updates
                    all_metrics[k] = float(all_metrics.get(k, 0.0)) + float(v) / float(num_updates)
            self._log_train_metrics(all_metrics, fps_hint=1)

        return next_obs

    def _log_train_metrics(self, metrics: dict[str, Any], fps_hint: int) -> None:
        now = time.time()
        elapsed = now - self.state.start_time
        fps = float(self.state.step) / elapsed if elapsed > 0 else 0.0

        namespaced = {}
        for k, v in metrics.items():
            namespaced[f"train/{k}"] = float(v)
        namespaced["time/fps"] = fps
        self.logger.log(namespaced, step=self.state.step)

    def _maybe_log(self) -> None:
        if self.log_every > 0 and (self.state.step % self.log_every == 0):
            self.logger.flush()

    def _maybe_eval(self) -> None:
        if self.eval_every > 0 and (self.state.step % self.eval_every == 0):
            eval_metrics = self.evaluator.evaluate(
                self.agent, step=self.state.step, n_episodes=self.eval_episodes
            )
            self.logger.log(eval_metrics, step=self.state.step)

            mean_ret = float(eval_metrics.get("eval/return_mean", float("-inf")))
            if mean_ret > self.state.best_eval_return:
                self.state.best_eval_return = mean_ret
                if self.save_best:
                    self.save_checkpoint(step=self.state.step, is_best=True)

    def _maybe_save(self) -> None:
        if self.save_every > 0 and (self.state.step % self.save_every == 0):
            self.save_checkpoint(step=self.state.step, is_best=False)

    def save_checkpoint(self, step: int, is_best: bool = False) -> None:
        ckpt = {
            "step": step,
            "agent": self.agent.state_dict(),
            "config": self.cfg,
        }
        ckpt_path = self.run_dir / "checkpoints" / f"ckpt_{step}.pt"
        torch.save(ckpt, ckpt_path)

        # convenience links
        torch.save(ckpt, self.run_dir / "latest.pt")
        if is_best:
            torch.save(ckpt, self.run_dir / "checkpoints" / "best.pt")

    def load_checkpoint(self, path_or_run_dir: str) -> int:
        p = Path(path_or_run_dir)
        if p.is_dir():
            # Prefer latest.pt, else best.pt
            if (p / "latest.pt").exists():
                p = p / "latest.pt"
            elif (p / "checkpoints" / "best.pt").exists():
                p = p / "checkpoints" / "best.pt"
            else:
                raise FileNotFoundError(f"No checkpoint found in run_dir={path_or_run_dir}")

        ckpt = torch.load(p, map_location="cpu")
        self.agent.load_state_dict(ckpt["agent"])
        self.state.step = int(ckpt.get("step", 0))
        return self.state.step
