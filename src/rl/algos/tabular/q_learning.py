# src/rl/algos/tabular/q_learning.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from rl.algos.registry import register


def _to_numpy(obs) -> np.ndarray:
    if isinstance(obs, np.ndarray):
        return obs
    return np.asarray(obs, dtype=np.float32)


@dataclass
class EpsilonSchedule:
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 50_000

    def value(self, step: int) -> float:
        if self.eps_decay_steps <= 0:
            return float(self.eps_end)
        t = min(max(step, 0), self.eps_decay_steps)
        frac = t / float(self.eps_decay_steps)
        return float(self.eps_start + frac * (self.eps_end - self.eps_start))


class Discretizer:
    """
    Simple uniform binning for Box observations. Works for CartPole by default
    with reasonable clipping ranges.
    """

    def __init__(
        self,
        *,
        low: np.ndarray,
        high: np.ndarray,
        bins: np.ndarray,
    ) -> None:
        self.low = low.astype(np.float32)
        self.high = high.astype(np.float32)
        self.bins = bins.astype(np.int32)

        # precompute bin edges per dimension
        self.edges = []
        for i in range(len(self.low)):
            # bins[i] bins -> bins[i]-1 internal edges
            if self.bins[i] <= 1:
                self.edges.append(None)
                continue
            e = np.linspace(self.low[i], self.high[i], self.bins[i] + 1, dtype=np.float32)
            self.edges.append(e)

    def encode(self, obs: np.ndarray) -> tuple[int, ...]:
        obs = obs.astype(np.float32)
        obs = np.clip(obs, self.low, self.high)
        idxs = []
        for i, e in enumerate(self.edges):
            if e is None:
                idxs.append(0)
                continue
            # find bin index in [0, bins-1]
            # np.digitize returns [1..len(e)-1], so subtract 1
            b = int(np.digitize(obs[i], e[1:-1], right=False))
            b = max(0, min(b, int(self.bins[i]) - 1))
            idxs.append(b)
        return tuple(idxs)


@register("q_learning")
class QLearningAgent:
    """
    Minimal Q-learning agent that fits the Trainer contract.

    training_mode: on_policy
    - act(): epsilon-greedy on discretized state
    - update(trajectory): apply tabular Q-learning updates over transitions

    Algorithm:
        Q-learning (tabular, epsilon-greedy)

    Reference:
        Watkins & Dayan (1992), "Q-learning"

    Docs:
        docs/02_value_based/q_learning.md

    Training mode:
        on_policy (trajectory-based update in this implementation)

    Key equations:
        Q(s,a) <- Q(s,a) + alpha [ r + gamma max_a' Q(s',a') - Q(s,a) ]

    Config keys:
        algo.gamma
        algo.alpha
        algo.eps_start
        algo.eps_end
        algo.eps_decay_steps
        algo.bins
        algo.clip_low
        algo.clip_high

    Notes:
        - Continuous observations are discretized via uniform binning.
        - Intended for educational comparison, not performance benchmarks.
    """

    training_mode = "on_policy"

    def __init__(self, *, env: gym.Env, cfg: dict[str, Any]) -> None:
        self.env = env
        self.cfg = cfg

        algo = cfg.get("algo", {}) or {}
        self.gamma = float(algo.get("gamma", 0.99))
        self.alpha = float(algo.get("alpha", algo.get("lr", 0.1)))

        self.global_step = 0

        self.eps_sched = EpsilonSchedule(
            eps_start=float(algo.get("eps_start", 1.0)),
            eps_end=float(algo.get("eps_end", 0.05)),
            eps_decay_steps=int(algo.get("eps_decay_steps", 50_000)),
        )

        # Action space must be discrete
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError("QLearningAgent requires a Discrete action space.")
        self.n_actions = int(env.action_space.n)

        # Observation: Discrete or Box (we discretize Box)
        self.discretizer: Discretizer | None = None
        if isinstance(env.observation_space, gym.spaces.Discrete):
            self.obs_kind = "discrete"
            self.n_states = int(env.observation_space.n)
        elif isinstance(env.observation_space, gym.spaces.Box):
            self.obs_kind = "box"
            obs_dim = int(np.prod(env.observation_space.shape))

            # Reasonable CartPole clipping defaults; can override in config
            clip_low = np.array(algo.get("clip_low", [-4.8, -5.0, -0.418, -5.0]), dtype=np.float32)
            clip_high = np.array(algo.get("clip_high", [4.8, 5.0, 0.418, 5.0]), dtype=np.float32)

            bins = np.array(algo.get("bins", [9, 9, 9, 9]), dtype=np.int32)
            if bins.shape[0] != obs_dim:
                raise ValueError(f"bins length {bins.shape[0]} must match obs_dim {obs_dim}")

            if clip_low.shape[0] != obs_dim or clip_high.shape[0] != obs_dim:
                raise ValueError("clip_low/clip_high must match observation dimension.")

            self.discretizer = Discretizer(low=clip_low, high=clip_high, bins=bins)
        else:
            raise ValueError("Unsupported observation space for QLearningAgent.")

        # Q-table: dict for Box-discretized, array for Discrete states
        if self.obs_kind == "discrete":
            self.q = np.zeros((self.n_states, self.n_actions), dtype=np.float32)
        else:
            self.q = {}  # key: tuple bins -> np.array(n_actions)

        self._last_episode_return = 0.0
        self._last_episode_len = 0

    def _encode_state(self, obs) -> Any:
        if self.obs_kind == "discrete":
            return int(obs)
        x = _to_numpy(obs).reshape(-1)
        assert self.discretizer is not None
        return self.discretizer.encode(x)

    def _get_q_row(self, state) -> np.ndarray:
        if self.obs_kind == "discrete":
            return self.q[state]
        row = self.q.get(state)
        if row is None:
            row = np.zeros((self.n_actions,), dtype=np.float32)
            self.q[state] = row
        return row

    def act(self, obs, *, deterministic: bool):
        state = self._encode_state(obs)
        eps = 0.0 if deterministic else self.eps_sched.value(self.global_step)

        if (not deterministic) and (np.random.rand() < eps):
            action = int(self.env.action_space.sample())
        else:
            q_row = self._get_q_row(state)
            action = int(np.argmax(q_row))

        return {"action": action}

    def update(self, traj) -> dict[str, float]:
        # Tabular Q-learning update over transitions
        td_errors = []
        for obs, a, r, done, next_obs in zip(
            traj.obs, traj.actions, traj.rewards, traj.dones, traj.next_obs, strict=False
        ):
            s = self._encode_state(obs)
            ns = self._encode_state(next_obs)

            q_s = self._get_q_row(s)
            q_ns = self._get_q_row(ns)

            target = float(r)
            if not done:
                target += self.gamma * float(np.max(q_ns))

            td = target - float(q_s[int(a)])
            q_s[int(a)] += self.alpha * td
            td_errors.append(abs(float(td)))

            self.global_step += 1

        mean_td = float(np.mean(td_errors)) if td_errors else 0.0
        return {
            "td_error_mean": mean_td,
            "epsilon": float(self.eps_sched.value(self.global_step)),
        }

    def on_episode_end(self, info: dict) -> None:
        # no-op hook; could track episodic return if you wrap env to provide it in info
        pass

    def state_dict(self) -> dict:
        # store q-table + schedule state
        if self.obs_kind == "discrete":
            q_payload = self.q
        else:
            # convert dict-of-arrays to dict-of-lists for safe serialization
            q_payload = {k: v.tolist() for k, v in self.q.items()}

        return {
            "obs_kind": self.obs_kind,
            "q": q_payload,
            "global_step": int(self.global_step),
        }

    def load_state_dict(self, state: dict) -> None:
        self.obs_kind = state["obs_kind"]
        self.global_step = int(state.get("global_step", 0))

        if self.obs_kind == "discrete":
            self.q = np.asarray(state["q"], dtype=np.float32)
        else:
            qd = state["q"]
            self.q = {
                (
                    tuple(map(int, k.strip("()").split(","))) if isinstance(k, str) else tuple(k)
                ): np.asarray(v, dtype=np.float32)
                for k, v in qd.items()
            }
            # NOTE: if keys were serialized as tuples by torch,
            # the above handles both tuple and string forms.
