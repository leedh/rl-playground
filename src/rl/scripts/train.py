# src/rl/scripts/train.py
from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml  # pip install pyyaml
from rl.algos.registry import make_agent as make_agent_from_registry
from rl.core.evaluate import Evaluator
from rl.core.logger import CsvLogger
from rl.core.seed import seed_everything
from rl.core.trainer import Trainer
from rl.envs.make_env import make_env


def _deep_update(d: dict[str, Any], u: dict[str, Any]) -> dict[str, Any]:
    """Recursively update dict d with dict u (in-place)."""
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            _deep_update(d[k], v)
        else:
            d[k] = v
    return d


def _set_by_dotted_key(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    cur = cfg
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def _parse_scalar(s: str) -> Any:
    """Parse override values: JSON if possible, else try int/float/bool/null, else string."""
    # Try JSON first (supports lists/dicts/strings/numbers/bools/null)
    try:
        return json.loads(s)
    except Exception:
        pass

    sl = s.strip().lower()
    if sl in ("true", "false"):
        return sl == "true"
    if sl in ("null", "none"):
        return None

    # int
    try:
        if s.strip().startswith("0") and s.strip() != "0":
            # keep leading-zero strings as strings
            raise ValueError
        return int(s)
    except Exception:
        pass

    # float
    try:
        return float(s)
    except Exception:
        return s


def load_config(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        cfg = {}
    return cfg


def load_config_with_base(path: str) -> dict[str, Any]:
    cfg_path = Path(path)
    cfg = load_config(path)

    base_ref = cfg.get("base")
    if not base_ref:
        return cfg

    base_path = (cfg_path.parent / str(base_ref)).resolve()
    base_cfg = load_config(str(base_path)) if base_path.exists() else {}

    # Merge: base first, then overrides from cfg (excluding base key)
    merged = copy.deepcopy(base_cfg)
    cfg_no_base = dict(cfg)
    cfg_no_base.pop("base", None)
    _deep_update(merged, cfg_no_base)
    return merged


def apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"Invalid override '{ov}'. Expected key=value.")
        k, v = ov.split("=", 1)
        _set_by_dotted_key(cfg, k.strip(), _parse_scalar(v.strip()))
    return cfg


def resolve_run_dir(cfg: dict[str, Any], run_dir_cli: str | None) -> Path:
    # Priority: CLI --run-dir > paths.run_dir > logging.dir + auto-name
    if run_dir_cli:
        return Path(run_dir_cli)

    paths = cfg.get("paths", {}) or {}
    if paths.get("run_dir"):
        return Path(paths["run_dir"])

    logging_cfg = cfg.get("logging", {}) or {}
    base_dir = Path(logging_cfg.get("dir", "experiments/results"))
    env_id = (cfg.get("env", {}) or {}).get("id", "unknown_env")
    algo_name = (cfg.get("algo", {}) or {}).get("name", "unknown_algo")
    run_name = logging_cfg.get("run_name")
    if not run_name:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"{ts}"
    return base_dir / env_id / algo_name / run_name


def ensure_run_dir(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)


# -------- Agent factory (minimal registry) --------
def make_agent(cfg: dict[str, Any], env) -> Any:
    """
    Minimal agent factory by algo.name.
    Replace this with a proper registry later (recommended).
    """
    algo_cfg = cfg.get("algo", {}) or {}
    name = algo_cfg.get("name", None)
    if name is None:
        raise ValueError("config.algo.name is required")

    name = str(name).lower()

    # Example: wire up only what's implemented now. Extend as you implement more agents.
    if name in ("q_learning", "q-learning", "qlearning"):
        from rl.algos.tabular.q_learning import QLearningAgent

        return QLearningAgent(env=env, cfg=cfg)
    if name in ("sarsa",):
        from rl.algos.tabular.sarsa import SARSAgent  # adjust class name if needed

        return SARSAgent(env=env, cfg=cfg)
    if name in ("dqn",):
        from rl.algos.deep_value.dqn import DQNAgent

        return DQNAgent(env=env, cfg=cfg)
    if name in ("reinforce",):
        from rl.algos.policy_grad.reinforce import ReinforceAgent

        return ReinforceAgent(env=env, cfg=cfg)
    if name in ("actor_critic", "actor-critic", "a2c"):
        from rl.algos.policy_grad.actor_critic import ActorCriticAgent

        return ActorCriticAgent(env=env, cfg=cfg)
    if name in ("ppo",):
        from rl.algos.actor_critic.ppo import PPOAgent

        return PPOAgent(env=env, cfg=cfg)

    raise ValueError(f"Unknown algo.name='{name}'. Add it to make_agent().")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RL Playground training entrypoint")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config")
    p.add_argument(
        "--overrides",
        type=str,
        nargs="*",
        default=[],
        help=(
            "Config overrides as key=value (supports dotted keys). "
            "Example: train.seed=0 algo.lr=3e-4"
        ),
    )
    p.add_argument("--run-dir", type=str, default=None, help="Override output run directory")
    p.add_argument(
        "--device", type=str, default=None, help="Override train.device (cpu/cuda/mps/auto)"
    )
    p.add_argument(
        "--resume", type=str, default=None, help="Resume from checkpoint (file path or run_dir)"
    )
    p.add_argument(
        "--dry-run", action="store_true", help="Validate config, create run_dir, then exit"
    )
    p.add_argument("--print-config", action="store_true", help="Print final merged config and exit")
    return p


def main() -> int:
    args = build_argparser().parse_args()

    cfg = load_config_with_base(args.config)
    cfg = apply_overrides(cfg, args.overrides)

    # CLI --device overrides config.train.device
    if args.device is not None:
        cfg = copy.deepcopy(cfg)
        cfg.setdefault("train", {})
        cfg["train"]["device"] = args.device

    if args.print_config:
        print(yaml.safe_dump(cfg, sort_keys=False))
        return 0

    run_dir = resolve_run_dir(cfg, args.run_dir)
    ensure_run_dir(run_dir)

    # Persist final config for reproducibility
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    if args.dry_run:
        print(f"[dry-run] run_dir = {run_dir}")
        return 0

    # Seed everything
    train_cfg = cfg.get("train", {}) or {}
    seed = int(train_cfg.get("seed", 0))
    deterministic_torch = bool(train_cfg.get("deterministic_torch", True))
    seed_everything(seed=seed, deterministic_torch=deterministic_torch)

    # Build env + agent
    env = make_env(cfg, mode="train")
    agent = make_agent_from_registry(cfg, env)
    # agent = make_agent(cfg, env)

    # Logger + evaluator
    logger = CsvLogger(
        run_dir=run_dir, flush_every=int((cfg.get("logging", {}) or {}).get("flush_every", 1000))
    )
    evaluator = Evaluator(cfg=cfg, run_dir=run_dir)

    trainer = Trainer(
        env=env,
        agent=agent,
        evaluator=evaluator,
        logger=logger,
        config=cfg,
        run_dir=run_dir,
    )

    # Resume if requested
    if args.resume:
        trainer.load_checkpoint(args.resume)

    summary = trainer.run()
    logger.log(
        {**{f"summary/{k}": v for k, v in summary.items()}}, step=int(summary.get("total_steps", 0))
    )
    logger.flush()

    print(f"Done. run_dir={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
