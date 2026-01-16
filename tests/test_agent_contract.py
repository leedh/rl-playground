# tests/test_agent_contract.py
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pytest
from rl.algos.registry import make_agent
from rl.core.rollout import collect_trajectory


def _base_cfg(algo_name: str) -> dict[str, Any]:
    return {
        "env": {"id": "CartPole-v1", "wrappers": {"record_video": False}},
        "algo": {"name": algo_name, "gamma": 0.99, "rollout_steps": 32},
        "train": {"seed": 0, "device": "cpu", "deterministic_torch": True},
        "logging": {"dir": "experiments/results"},
        "paths": {"run_dir": None},
    }


@pytest.mark.parametrize("algo_name", ["q_learning", "ppo"])
def test_agent_act_update_state_dict_roundtrip(algo_name: str):
    cfg = _base_cfg(algo_name)
    env = gym.make(cfg["env"]["id"])

    agent = make_agent(cfg, env)

    # act() should return either action or dict with action
    obs, _ = env.reset(seed=0)
    out = agent.act(obs, deterministic=False)
    action = out["action"] if isinstance(out, dict) else out
    # Ensure action is valid
    env.action_space.contains(action)

    # collect a tiny trajectory then update
    traj, _ = collect_trajectory(env=env, agent=agent, start_obs=obs, rollout_steps=32)
    metrics = agent.update(traj)
    assert isinstance(metrics, dict)
    # expect scalar-ish values if present
    for _, v in metrics.items():
        assert np.isfinite(float(v))

    # state_dict roundtrip should not crash
    sd = agent.state_dict()
    agent.load_state_dict(sd)

    env.close()
