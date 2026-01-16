# src/rl/envs/make_env.py
from __future__ import annotations

from typing import Any

import gymnasium as gym

from rl.envs.wrappers import apply_wrappers


def make_env(cfg: dict[str, Any], mode: str = "train"):
    env_cfg = cfg.get("env", {}) or {}
    env_id = env_cfg.get("id", None)
    if env_id is None:
        raise ValueError("config.env.id is required")

    make_kwargs = dict(env_cfg.get("kwargs", {}) or {})
    if "render_mode" not in make_kwargs and env_cfg.get("render_mode") is not None:
        make_kwargs["render_mode"] = env_cfg.get("render_mode")

    # If video recording is enabled for eval, ensure render_mode is compatible
    wrappers_cfg = env_cfg.get("wrappers", {}) or {}
    record_video = bool(wrappers_cfg.get("record_video", False)) and (mode == "eval")
    if record_video and "render_mode" not in make_kwargs:
        make_kwargs["render_mode"] = "rgb_array"

    env = gym.make(env_id, **make_kwargs)

    # Apply wrappers consistently
    env = apply_wrappers(env, cfg=cfg, mode=mode)
    return env
