# tests/test_registry_docstrings.py
from __future__ import annotations

import inspect

import rl.algos  # noqa: F401  (import side-effects: registers algos)
from rl.algos.registry import registered_algorithms


def test_all_registered_algorithms_have_valid_docstrings_and_links():
    reg = registered_algorithms()
    assert (
        len(reg) > 0
    ), "No algorithms registered. Did you forget to import rl.algos in rl/algos/__init__.py?"

    # The registry validation runs at registration time.
    # This test mainly ensures registration indeed occurred for all algorithms.
    # If any docstring/link/mode mismatch exists, import-time registration should already fail.
    for name, factory in reg.items():
        assert factory is not None
        # sanity: decorated object should be a class in our repo
        assert inspect.isclass(
            factory
        ), f"Registry entry '{name}' is not a class. Use @register on the agent class."
