# src/rl/algos/registry.py
from __future__ import annotations

import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Any

import gymnasium as gym

Factory = Callable[..., Any]
_REGISTRY: dict[str, Factory] = {}

# Required docstring fields (headers)
_REQUIRED_DOC_FIELDS = [
    "Algorithm:",
    "Reference:",
    "Docs:",
    "Training mode:",
    "Key equations:",
    "Config keys:",
]

_ALLOWED_TRAINING_MODES = {"on_policy", "off_policy"}


def _repo_root() -> Path:
    # registry.py is at src/rl/algos/registry.py
    # repo root is 3 parents up: registry.py -> algos -> rl -> src -> repo
    return Path(__file__).resolve().parents[3]


def _extract_doc_field(doc: str, header: str) -> str:
    """
    Extract the text block following `header` until next known header or end.
    This is intentionally strict-ish but simple.
    """
    lines = doc.splitlines()
    start = None
    for i, ln in enumerate(lines):
        if ln.strip() == header:
            start = i + 1
            break
    if start is None:
        return ""

    # end at next header
    end = len(lines)
    headers = set(_REQUIRED_DOC_FIELDS + ["Notes:"])
    for j in range(start, len(lines)):
        if lines[j].strip() in headers:
            end = j
            break

    block = "\n".join(lines[start:end]).strip()
    return block


def _parse_docs_paths(doc: str) -> list[str]:
    """
    Parse 'Docs:' block as one or multiple repo-relative paths.
    Accepts:
      Docs:
          docs/xx.md
      or comma-separated, or bullet list.
    """
    block = _extract_doc_field(doc, "Docs:")
    if not block:
        return []

    # normalize: split lines, remove bullets, split commas
    paths: list[str] = []
    for raw in block.splitlines():
        s = raw.strip()
        if not s:
            continue
        if s.startswith(("-", "*")):
            s = s[1:].strip()
        # allow commas on the same line
        for part in [p.strip() for p in s.split(",")]:
            if part:
                paths.append(part)
    return paths


def _parse_training_mode_from_doc(doc: str) -> str:
    block = _extract_doc_field(doc, "Training mode:")
    # accept first token in the block
    mode = block.split()[0].strip() if block else ""
    return mode


def _validate_docstring(cls: type) -> None:
    doc = inspect.getdoc(cls)
    if not doc:
        raise ValueError(f"{cls.__name__} must have a class docstring.")

    missing = [h for h in _REQUIRED_DOC_FIELDS if h not in doc]
    if missing:
        raise ValueError(
            f"{cls.__name__} docstring missing fields: {missing}\n"
            f"Required fields: {_REQUIRED_DOC_FIELDS}"
        )

    # Validate docs links exist
    root = _repo_root()
    docs_paths = _parse_docs_paths(doc)
    if not docs_paths:
        raise ValueError(f"{cls.__name__}: 'Docs:' must list at least one docs/*.md path.")

    for p in docs_paths:
        # Only allow repo-relative docs paths
        if not p.startswith("docs/"):
            raise ValueError(
                f"{cls.__name__}: Docs path must be repo-relative and start with 'docs/': got '{p}'"
            )
        fp = (root / p).resolve()
        if not fp.exists():
            raise FileNotFoundError(
                f"{cls.__name__}: Docs file not found: '{p}' (expected at {fp})"
            )

    # Validate training mode in docstring is one of allowed
    mode = _parse_training_mode_from_doc(doc)
    if mode not in _ALLOWED_TRAINING_MODES:
        raise ValueError(
            f"{cls.__name__}: Training mode in docstring must be one of "
            f"{sorted(_ALLOWED_TRAINING_MODES)}. Got: '{mode}'"
        )


def _validate_training_mode_matches_runtime(cls: type, factory: Factory) -> None:
    """
    (요구 2) Validate that docstring Training mode matches class attribute training_mode.
    """
    doc = inspect.getdoc(cls) or ""
    doc_mode = _parse_training_mode_from_doc(doc)

    runtime_mode = getattr(cls, "training_mode", None)
    if runtime_mode is None:
        raise ValueError(f"{cls.__name__} must define class attribute 'training_mode'.")

    if str(runtime_mode) != doc_mode:
        raise ValueError(
            f"{cls.__name__}: training_mode mismatch.\n"
            f"  docstring Training mode: {doc_mode}\n"
            f"  class attribute training_mode: {runtime_mode}\n"
            f"Fix one of them so they match."
        )


def register(name: str) -> Callable[[Factory], Factory]:
    key = name.lower().strip()

    def _decorator(factory: Factory) -> Factory:
        # We expect the decorated object to be a class (agent) OR a callable returning an agent.
        cls = factory if inspect.isclass(factory) else factory.__class__

        _validate_docstring(cls)
        _validate_training_mode_matches_runtime(cls, factory)

        if key in _REGISTRY:
            raise ValueError(f"Algorithm '{key}' already registered.")

        _REGISTRY[key] = factory
        return factory

    return _decorator


def registered_algorithms() -> dict[str, Factory]:
    return dict(_REGISTRY)


def _normalize_algo_name(name: str) -> str:
    return name.lower().strip().replace("-", "_")


def make_agent(cfg: dict[str, Any], env: gym.Env) -> Any:
    algo_cfg = cfg.get("algo", {}) or {}
    name = algo_cfg.get("name")
    if not name:
        raise ValueError("config.algo.name is required")

    key = _normalize_algo_name(str(name))
    if key not in _REGISTRY:
        choices = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"Unknown algo.name='{key}'. Registered: [{choices}]")

    factory = _REGISTRY[key]
    if inspect.isclass(factory):
        return factory(env=env, cfg=cfg)
    return factory(env=env, cfg=cfg)
