# src/rl/algos/__init__.py
"""
Import this package to register algorithms.
"""

from rl.algos.actor_critic.ppo import PPOAgent  # noqa: F401
from rl.algos.registry import make_agent, register, registered_algorithms  # noqa: F401

# Import algorithm modules so @register decorators execute.
# Keep this lightweight; only import top-level agent definitions.
from rl.algos.tabular.q_learning import QLearningAgent  # noqa: F401
