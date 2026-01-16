# src/rl/core/logger.py
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def _to_float(v: Any) -> float | None:
    try:
        return float(v)
    except Exception:
        return None


class BaseLogger:
    def log(self, metrics: dict[str, Any], *, step: int) -> None:
        raise NotImplementedError

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


class CompositeLogger(BaseLogger):
    """
    Fan-out logger: writes the same metrics to multiple backends.
    """

    def __init__(self, loggers: list[BaseLogger]) -> None:
        self.loggers = loggers

    def log(self, metrics: dict[str, Any], *, step: int) -> None:
        for lg in self.loggers:
            lg.log(metrics, step=step)

    def flush(self) -> None:
        for lg in self.loggers:
            lg.flush()

    def close(self) -> None:
        for lg in self.loggers:
            lg.close()


class CsvLogger(BaseLogger):
    def __init__(self, *, run_dir: Path, flush_every: int = 1000) -> None:
        self.run_dir = run_dir
        self.flush_every = int(flush_every)

        self.path = run_dir / "metrics.csv"
        self._fp = open(self.path, "a", newline="", encoding="utf-8")
        self._writer: csv.DictWriter | None = None
        self._row_count = 0

    def log(self, metrics: dict[str, Any], *, step: int) -> None:
        row: dict[str, Any] = {"step": int(step)}
        for k, v in metrics.items():
            fv = _to_float(v)
            if fv is not None:
                row[k] = fv

        if self._writer is None:
            fieldnames = list(row.keys())
            self._writer = csv.DictWriter(self._fp, fieldnames=fieldnames)
            if self._fp.tell() == 0:
                self._writer.writeheader()

        # Minimal behavior: ignore new keys after header established
        filtered = {k: row[k] for k in self._writer.fieldnames if k in row}
        self._writer.writerow(filtered)

        self._row_count += 1
        if self.flush_every > 0 and (self._row_count % self.flush_every == 0):
            self.flush()

    def flush(self) -> None:
        self._fp.flush()

    def close(self) -> None:
        try:
            self._fp.flush()
        finally:
            self._fp.close()


class TensorBoardLogger(BaseLogger):
    def __init__(self, *, run_dir: Path) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except Exception as e:
            raise ImportError(
                "TensorBoardLogger requires tensorboard. Install with: pip install -e '.[tb]'"
            ) from e

        self.run_dir = run_dir
        self.tb_dir = run_dir / "tb"
        self.tb_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.tb_dir))

    def log(self, metrics: dict[str, Any], *, step: int) -> None:
        for k, v in metrics.items():
            fv = _to_float(v)
            if fv is not None:
                self.writer.add_scalar(k, fv, global_step=int(step))

    def flush(self) -> None:
        self.writer.flush()

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()


class WandbLogger(BaseLogger):
    def __init__(self, *, run_dir: Path, cfg: dict[str, Any]) -> None:
        try:
            import wandb
        except Exception as e:
            raise ImportError(
                "WandbLogger requires wandb. Install with: pip install -e '.[wandb]'"
            ) from e

        self.run_dir = run_dir
        self.wandb = wandb

        project = (cfg.get("project", {}) or {}).get("name", "rl-playground")
        tags = (cfg.get("project", {}) or {}).get("tags", [])

        logging_cfg = cfg.get("logging", {}) or {}
        run_name = logging_cfg.get("run_name", None) or run_dir.name

        self.wandb.init(
            project=project,
            name=run_name,
            tags=tags if isinstance(tags, list) else None,
            dir=str(run_dir),
            config=cfg,
        )

    def log(self, metrics: dict[str, Any], *, step: int) -> None:
        payload: dict[str, Any] = {"step": int(step)}
        for k, v in metrics.items():
            fv = _to_float(v)
            if fv is not None:
                payload[k] = fv
        self.wandb.log(payload, step=int(step))

    def close(self) -> None:
        try:
            self.wandb.finish()
        except Exception:
            pass


def _make_single_logger(backend: str, *, cfg: dict[str, Any], run_dir: Path) -> BaseLogger:
    backend = backend.lower().strip()
    logging_cfg = cfg.get("logging", {}) or {}

    if backend == "csv":
        flush_every = int(logging_cfg.get("flush_every", 1000))
        return CsvLogger(run_dir=run_dir, flush_every=flush_every)

    if backend in ("tb", "tensorboard"):
        return TensorBoardLogger(run_dir=run_dir)

    if backend == "wandb":
        return WandbLogger(run_dir=run_dir, cfg=cfg)

    raise ValueError(f"Unknown logging backend '{backend}'")


def make_logger(*, cfg: dict[str, Any], run_dir: Path) -> BaseLogger:
    """
    Supports:
      logging.backend: "csv" | "tensorboard" | "wandb"
      logging.backends: ["csv", "tensorboard"]  # composite
    Rule of thumb:
      - If backends provided, use them.
      - Else use backend (single).
    """
    logging_cfg = cfg.get("logging", {}) or {}

    backends = logging_cfg.get("backends", None)
    if backends is not None:
        if not isinstance(backends, list) or len(backends) == 0:
            raise ValueError("logging.backends must be a non-empty list")
        loggers = [_make_single_logger(b, cfg=cfg, run_dir=run_dir) for b in backends]
        return CompositeLogger(loggers)

    backend = str(logging_cfg.get("backend", "csv"))
    return _make_single_logger(backend, cfg=cfg, run_dir=run_dir)
