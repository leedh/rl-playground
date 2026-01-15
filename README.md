# RL Playground

**RL Playground** is a personal research and study repository for reinforcement learning, organized around the workflow:

**theory (math) → implementation → empirical comparison**.

Each algorithm is first documented with its mathematical formulation and intuition, then implemented in a clean and reproducible way, and finally evaluated in Gym/Gymnasium environments under controlled experimental conditions.

This repository is **not** intended to be a production-grade RL library.
Its primary goal is to make the connection between **equations, code, and empirical behavior** explicit and easy to trace.

---

## Features

* Step-by-step coverage from tabular RL to modern actor–critic methods
* One-to-one mapping between theory documents (`docs/`) and implementations (`src/rl/algos/`)
* YAML-based configuration for training, evaluation, and sweeps
* Shared training / rollout / evaluation pipeline
* Environment-level benchmarks and algorithm comparison
* Reproducibility-focused design (seeds, logging, result structure)

---

## Implemented Algorithms

| Category              | Algorithm    | Theory                                  | Code                                       |
| --------------------- | ------------ | --------------------------------------- | ------------------------------------------ |
| Tabular               | Q-learning   | `docs/02_value_based/q_learning.md`     | `src/rl/algos/tabular/q_learning.py`       |
| Tabular               | SARSA        | `docs/02_value_based/sarsa.md`          | `src/rl/algos/tabular/sarsa.py`            |
| Deep Value            | DQN          | `docs/02_value_based/dqn.md`            | `src/rl/algos/deep_value/dqn.py`           |
| Policy Gradient       | REINFORCE    | `docs/03_policy_based/reinforce.md`     | `src/rl/algos/policy_grad/reinforce.py`    |
| Actor–Critic          | Actor–Critic | `docs/03_policy_based/actor_critic.md`  | `src/rl/algos/policy_grad/actor_critic.py` |
| Modern Actor–Critic   | PPO          | `docs/04_actor_critic_modern/ppo.md`    | `src/rl/algos/actor_critic/ppo.py`         |
| Off-policy Continuous | DDPG         | `docs/05_off_policy_continuous/ddpg.md` | (WIP)                                      |
| Off-policy Continuous | TD3          | `docs/05_off_policy_continuous/td3.md`  | (WIP)                                      |
| Off-policy Continuous | SAC          | `docs/05_off_policy_continuous/sac.md`  | (WIP)                                      |

---

## Quickstart

### Clone this repository

```bash
git clone https://github.com/leedh/rl-playground.git
cd rl-playground
pip install -e .
```

---

### Virtual Environment & Dependency

#### Conda environment
```bash
conda create -n rl-playground python=3.10 -y
conda activate rl-playground
# CPU only
conda install pytorch -c pytorch -y
# CUDA (e.g., CUDA 11.8)
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia -y
# Check installation for CUDA
python - << 'EOF'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
EOF
```

#### Install dependency
```bash
# choose one of the two(tb for tensorboard, wandb for wandb)
pip install -e ".[dev,viz,video,tb]"
pip install -e ".[dev,viz,video,wandb]"
# Check installation of packages
python - << 'EOF'
import gymnasium, torch, yaml, numpy
print("OK")
EOF
```
You can install the followings with the code above:
- runtime dependencies
- pytest / pre-commit
- matplotlib
- video recording dependencies

**(Optional) Development setup:**

```bash
pre-commit install

# test
pre-commit run --all-files
```

Sanity check
```bash
pytest
```
Potential failure points:
- Docstring compliance violation
- Invalid documentation link path
- training_mode mismatch

Please fix according to the provided error message.

---

### Run Your First Experiment

#### Q-learning (CartPole)

```bash
python -m rl.scripts.train \
  --config src/rl/configs/q_learning_cartpole.yaml
```
Upon success, the console will:
- Print `run_dir`
- Create `experiments/results/CartPole-v1/q_learning/<timestamp>/`

#### DQN (CartPole)

```bash
python -m rl.scripts.train \
  --config src/rl/configs/dqn_cartpole.yaml
```

#### PPO (CartPole)

```bash
python -m rl.scripts.train \
  --config src/rl/configs/ppo_cartpole.yaml
```
Execution Artifacts:
```
experiments/results/
└── CartPole-v1/
    └── ppo/
        └── 2024-xx-xx_xx-xx-xx/
            ├── config.yaml
            ├── metrics.csv
            ├── latest.pt
            └── checkpoints/
```

#### Evaluate a Trained Agent

```bash
python -m rl.scripts.eval \
  --run experiments/results/<run_dir>
```

---

## Reproducibility

* All experiments are fully defined by **config files**
* Random seeds are centrally managed in `src/rl/core/seed.py`
* Training and evaluation use separate environments and seeds
* Results are stored per experiment run

**Example result directory:**

```
experiments/results/
└── cartpole/
    └── ppo/
        └── 2024-01-01_12-00-00/
            ├── config.yaml
            ├── metrics.csv
            ├── model.pt
            ├── plots.png
```

> The `experiments/results/` directory is not intended to be fully version-controlled.
> Only curated summaries and plots are tracked under `experiments/results/reports/`.

---

## Project Structure (High-level)

```
docs/               # Theory, math, and conceptual notes
src/rl/algos/       # Algorithm implementations
src/rl/core/        # Trainer, rollout, replay, evaluation, logging
src/rl/envs/        # Environment creation and wrappers
src/rl/configs/     # Reproducible experiment configurations
experiments/        # Benchmarks and summarized results
notebooks/          # Debugging, analysis, and ablations
tests/              # Determinism, shape, and sanity tests
```

---

## How to Add a New Algorithm

1. **Write the theory**

   ```
   docs/<category>/<algorithm>.md
   ```

2. **Implement the algorithm**

   ```
   src/rl/algos/<category>/<algorithm>.py
   ```

3. **Add a config**

   ```
   src/rl/configs/<algorithm>_<env>.yaml
   ```

4. **(Optional) Register benchmarks**

   ```
   experiments/benchmarks/<env>/
   ```

5. **Add minimal tests**

   * loss shape checks
   * action range checks
   * deterministic behavior under fixed seeds

All agents are expected to follow a minimal interface:

* `act(obs)`
* `update(batch or trajectory)`
* `save(path)` / `load(path)`

---

## Benchmarks & Comparison

Currently supported benchmark environments:

* CartPole
* MountainCar
* MuJoCo HalfCheetah (environment setup required)

To run algorithm sweeps under identical conditions:

```bash
python -m rl.scripts.sweep \
  --benchmark experiments/benchmarks/cartpole
```

Outputs:

* Learning curves: `experiments/results/plots/`
* Comparison reports: `experiments/results/reports/`

---

## Documentation Map (Theory ↔ Code)

* **Bellman Equations**
  → `docs/01_mdp_rl_basics/bellman_equations.md`
  → tabular updates, DQN target computation

* **Q-learning**
  → `docs/02_value_based/q_learning.md`
  → `src/rl/algos/tabular/q_learning.py`

* **Policy Gradient**
  → `docs/03_policy_based/reinforce.md`
  → `src/rl/algos/policy_grad/reinforce.py`

* **PPO Objective**
  → `docs/04_actor_critic_modern/ppo.md`
  → `src/rl/algos/actor_critic/ppo.py`, `losses.py`

---

## Development

```bash
pytest
```

* Code style and linting are enforced via pre-commit hooks
* Tests are intended as **guardrails**, not as formal correctness proofs

---

## Roadmap

* Full TRPO implementation
* PPO ablations (clip range, GAE λ, entropy bonus)
* Stabilized SAC and continuous-control benchmarks
* Automated hyperparameter sensitivity reports
