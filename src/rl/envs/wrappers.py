# src/rl/envs/wrappers.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym


def apply_wrappers(env, *, cfg: dict[str, Any], mode: str):
    env_cfg = cfg.get("env", {}) or {}
    wcfg = (env_cfg.get("wrappers", {}) or {}).copy()

    record_video = bool(wcfg.get("record_video", False)) and (mode == "eval")
    if record_video:
        video_dir = Path(wcfg.get("video_dir", "videos"))
        video_dir.mkdir(parents=True, exist_ok=True)

        video_episodes = int(wcfg.get("video_episodes", 3))
        ep_counter = {"n": 0}

        def trigger(episode_id: int) -> bool:
            # Record first N episodes only
            ep_counter["n"] += 1
            return ep_counter["n"] <= video_episodes

        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(video_dir),
            episode_trigger=trigger,
            name_prefix=f"{env_cfg.get('id', 'env')}_{mode}",
        )

    return env
