rl-playground/
├── README.md
├── pyproject.toml
├── .gitignore
├── .pre-commit-config.yaml
│
├── docs/
│   ├── 00_overview.md
│   ├── 01_mdp_rl_basics/
│   │   ├── mdp.md
│   │   ├── bellman_equations.md
│   │   └── policy_value_relationships.md
│   ├── 02_value_based/
│   │   ├── q_learning.md
│   │   ├── sarsa.md
│   │   └── dqn.md
│   ├── 03_policy_based/
│   │   ├── reinforce.md
│   │   └── actor_critic.md
│   ├── 04_actor_critic_modern/
│   │   ├── ppo.md
│   │   └── trpo.md
│   ├── 05_off_policy_continuous/
│   │   ├── ddpg.md
│   │   ├── td3.md
│   │   └── sac.md
│   └── 99_appendix/
│       ├── notation.md
│       ├── derivations.md
│       └── implementation_tips.md
│
├── src/
│   └── rl/
│       ├── __init__.py
│       ├── algos/
│       │   ├── __init__.py
│       │   ├── tabular/
│       │   │   ├── __init__.py
│       │   │   ├── q_learning.py
│       │   │   ├── sarsa.py
│       │   │   └── utils.py
│       │   ├── deep_value/
│       │   │   ├── __init__.py
│       │   │   ├── dqn.py
│       │   │   └── networks.py
│       │   ├── policy_grad/
│       │   │   ├── __init__.py
│       │   │   ├── reinforce.py
│       │   │   └── actor_critic.py
│       │   └── actor_critic/
│       │       ├── __init__.py
│       │       ├── ppo.py
│       │       ├── buffers.py
│       │       └── losses.py
│       │
│       ├── envs/
│       │   ├── __init__.py
│       │   ├── make_env.py
│       │   ├── wrappers.py
│       │   └── registry.py
│       │
│       ├── core/
│       │   ├── __init__.py
│       │   ├── trainer.py
│       │   ├── rollout.py
│       │   ├── replay_buffer.py
│       │   ├── logger.py
│       │   ├── evaluate.py
│       │   └── seed.py
│       │
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── metrics.py
│       │   ├── visualization.py
│       │   └── math_utils.py
│       │
│       ├── configs/
│       │   ├── _base.yaml
│       │   ├── q_learning_cartpole.yaml
│       │   ├── dqn_cartpole.yaml
│       │   ├── ppo_cartpole.yaml
│       │   └── ppo_mujoco.yaml
│       │
│       └── scripts/
│           ├── train.py
│           ├── eval.py
│           └── sweep.py
│
├── experiments/
│   ├── benchmarks/
│   │   ├── cartpole/
│   │   ├── mountaincar/
│   │   └── mujoco_halfcheetah/
│   ├── results/
│   │   ├── plots/
│   │   └── reports/
│   └── notes.md
│
├── notebooks/
│   ├── 01_tabular_debug.ipynb
│   ├── 02_policy_grad_sanity.ipynb
│   ├── 03_ppo_ablation.ipynb
│   ├── 04_algorithms_comparison.ipynb
│   └── 05_hyperparameter_sensitivity.ipynb
│
└── tests/
    ├── test_env_make.py
    ├── test_determinism.py
    └── test_ppo_loss_shapes.py