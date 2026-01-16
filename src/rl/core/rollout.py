# src/rl/core/rollout.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Trajectory:
    obs: list[Any]
    actions: list[Any]
    rewards: list[float]
    dones: list[bool]
    next_obs: list[Any]
    infos: list[dict[str, Any]]

    @property
    def num_steps(self) -> int:
        return len(self.rewards)


def collect_trajectory(
    *,
    env: Any,
    agent: Any,
    start_obs: Any,
    rollout_steps: int,
) -> tuple[Trajectory, Any]:
    obs = start_obs

    obs_list: list[Any] = []
    act_list: list[Any] = []
    rew_list: list[float] = []
    done_list: list[bool] = []
    next_obs_list: list[Any] = []
    info_list: list[dict[str, Any]] = []

    for _ in range(rollout_steps):
        act_out = agent.act(obs, deterministic=False)
        action = act_out["action"] if isinstance(act_out, dict) else act_out

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)

        obs_list.append(obs)
        act_list.append(action)
        rew_list.append(float(reward))
        done_list.append(done)
        next_obs_list.append(next_obs)
        info_list.append(info if isinstance(info, dict) else {})

        obs = next_obs
        if done:
            agent.on_episode_end(info if isinstance(info, dict) else {})
            obs, _ = env.reset()

    traj = Trajectory(
        obs=obs_list,
        actions=act_list,
        rewards=rew_list,
        dones=done_list,
        next_obs=next_obs_list,
        infos=info_list,
    )
    return traj, obs
