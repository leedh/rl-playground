# src/rl/scripts/eval.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml
from rl.algos.registry import make_agent
from rl.core.evaluate import Evaluator
from rl.core.seed import seed_everything
from rl.envs.make_env import make_env


def load_config_from_run_dir(run_dir: Path) -> dict[str, Any]:
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config.yaml in run_dir: {run_dir}")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def resolve_ckpt(run_dir: Path, ckpt: str | None) -> Path:
    if ckpt is None:
        # Prefer latest.pt, else best.pt
        if (run_dir / "latest.pt").exists():
            return run_dir / "latest.pt"
        if (run_dir / "checkpoints" / "best.pt").exists():
            return run_dir / "checkpoints" / "best.pt"
        raise FileNotFoundError(f"No latest.pt or checkpoints/best.pt in {run_dir}")

    p = Path(ckpt)
    if p.is_dir():
        # interpret as run_dir
        return resolve_ckpt(p, None)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    return p


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RL Playground evaluation entrypoint")
    p.add_argument(
        "--run", type=str, required=True, help="Run directory under experiments/results/..."
    )
    p.add_argument(
        "--ckpt", type=str, default=None, help="Checkpoint path or run_dir (default: latest.pt)"
    )
    p.add_argument("--episodes", type=int, default=10, help="Number of eval episodes")
    p.add_argument("--seed", type=int, default=None, help="Override eval seed")
    p.add_argument("--video", action="store_true", help="Record evaluation videos")
    p.add_argument("--video-episodes", type=int, default=3, help="How many episodes to record")
    return p


def main() -> int:
    args = build_argparser().parse_args()
    run_dir = Path(args.run)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    cfg = load_config_from_run_dir(run_dir)

    # Override eval options
    train_cfg = cfg.get("train", {}) or {}
    base_seed = int(train_cfg.get("seed", 0))
    eval_seed = int(args.seed) if args.seed is not None else base_seed + 10_000
    seed_everything(eval_seed, deterministic_torch=True)

    # Set video wrapper options
    cfg = dict(cfg)  # shallow copy
    env_cfg = dict(cfg.get("env", {}) or {})
    wrappers = dict(env_cfg.get("wrappers", {}) or {})
    wrappers["record_video"] = bool(args.video)
    wrappers["video_dir"] = str(run_dir / "videos")
    wrappers["video_episodes"] = int(args.video_episodes)
    env_cfg["wrappers"] = wrappers
    cfg["env"] = env_cfg

    # Build eval env + agent
    env = make_env(cfg, mode="eval")
    agent = make_agent(cfg, env)

    # Load checkpoint
    ckpt_path = resolve_ckpt(run_dir, args.ckpt)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    agent.load_state_dict(ckpt["agent"])

    evaluator = Evaluator(cfg=cfg, run_dir=run_dir)
    metrics = evaluator.evaluate(
        agent, step=int(ckpt.get("step", 0)), n_episodes=int(args.episodes)
    )

    env.close()

    print(f"run_dir: {run_dir}")
    print(f"checkpoint: {ckpt_path}")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
