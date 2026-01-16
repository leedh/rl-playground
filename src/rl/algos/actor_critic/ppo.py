# src/rl/algos/actor_critic/ppo.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.algos.registry import register


def _to_tensor(x, device: torch.device) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def _flatten_obs_list(obs_list) -> np.ndarray:
    # Handles obs as np arrays or lists; returns (T, obs_dim)
    arr = np.asarray(obs_list, dtype=np.float32)
    return arr.reshape(arr.shape[0], -1)


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: tuple[int, ...] = (64, 64)) -> None:
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.Tanh())
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SquashedNormal:
    """
    Squashed Gaussian policy (tanh) with log-prob correction.
    Minimal implementation for Box action spaces.
    """

    def __init__(self, mean: torch.Tensor, log_std: torch.Tensor, eps: float = 1e-6) -> None:
        self.mean = mean
        self.log_std = torch.clamp(log_std, -20.0, 2.0)
        self.std = torch.exp(self.log_std)
        self.base = torch.distributions.Normal(self.mean, self.std)
        self.eps = eps

    def sample(self) -> torch.Tensor:
        z = self.base.rsample()
        return torch.tanh(z)

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        # action in [-1, 1], invert tanh: atanh
        a = torch.clamp(action, -1 + self.eps, 1 - self.eps)
        z = 0.5 * (torch.log1p(a) - torch.log1p(-a))
        logp = self.base.log_prob(z).sum(dim=-1)
        # correction: sum log(1 - tanh(z)^2)
        corr = torch.log(1 - torch.tanh(z) ** 2 + self.eps).sum(dim=-1)
        return logp - corr


@dataclass
class PPOBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    old_logp: torch.Tensor


@register("ppo")
class PPOAgent:
    """
    Minimal PPO agent matching Trainer contract.
    - training_mode: on_policy
    - update() consumes Trajectory from core.rollout (obs, actions, rewards, dones, next_obs)

    Algorithm:
        Proximal Policy Optimization (PPO-Clip)

    Reference:
        Schulman et al. (2017), "Proximal Policy Optimization Algorithms"

    Docs:
        docs/04_actor_critic_modern/ppo.md

    Training mode:
        on_policy

    Key equations:
        L^CLIP(theta) =
            E_t [ min(
                r_t(theta) * A_t,
                clip(r_t(theta), 1-eps, 1+eps) * A_t
            ) ]

        r_t(theta) = pi_theta(a_t|s_t) / pi_theta_old(a_t|s_t)

    Config keys:
        algo.gamma
        algo.gae_lambda
        algo.clip_range
        algo.lr
        algo.update_epochs
        algo.minibatch_size
        algo.entropy_coef
        algo.vf_coef

    Notes:
        - GAE is computed inside the agent.
        - Value clipping is NOT used.
        - Advantage normalization is applied per update.
    """

    training_mode = "on_policy"

    def __init__(self, *, env: gym.Env, cfg: dict[str, Any]) -> None:
        self.env = env
        self.cfg = cfg

        algo = cfg.get("algo", {}) or {}
        train = cfg.get("train", {}) or {}

        device_str = train.get("device", "auto") or "auto"
        self.device = self._resolve_device(device_str)

        # Spaces
        self.obs_space = env.observation_space
        self.act_space = env.action_space

        if not isinstance(self.obs_space, gym.spaces.Box):
            raise ValueError("PPOAgent expects Box observation space (vector observations).")

        self.obs_dim = int(np.prod(self.obs_space.shape))

        # Hyperparameters
        self.gamma = float(algo.get("gamma", 0.99))
        self.gae_lambda = float(algo.get("gae_lambda", 0.95))
        self.clip_range = float(algo.get("clip_range", 0.2))
        self.entropy_coef = float(algo.get("entropy_coef", 0.0))
        self.vf_coef = float(algo.get("vf_coef", 0.5))
        self.max_grad_norm = float(algo.get("max_grad_norm", 0.5))

        self.lr = float(algo.get("lr", 3e-4))
        self.update_epochs = int(algo.get("update_epochs", 10))
        self.minibatch_size = int(algo.get("minibatch_size", 64))

        hidden = tuple(algo.get("hidden_sizes", [64, 64]))

        # Networks
        self.v_net = MLP(self.obs_dim, 1, hidden=hidden).to(self.device)

        if isinstance(self.act_space, gym.spaces.Discrete):
            self.is_discrete = True
            self.pi_net = MLP(self.obs_dim, int(self.act_space.n), hidden=hidden).to(self.device)
        elif isinstance(self.act_space, gym.spaces.Box):
            self.is_discrete = False
            self.act_dim = int(np.prod(self.act_space.shape))
            self.pi_mean = MLP(self.obs_dim, self.act_dim, hidden=hidden).to(self.device)
            self.pi_log_std = nn.Parameter(torch.zeros(self.act_dim, device=self.device))
            # action rescaling
            self.act_low = torch.as_tensor(
                self.act_space.low.reshape(-1), dtype=torch.float32, device=self.device
            )
            self.act_high = torch.as_tensor(
                self.act_space.high.reshape(-1), dtype=torch.float32, device=self.device
            )
        else:
            raise ValueError("Unsupported action space for PPOAgent.")

        self.optim = torch.optim.Adam(self.parameters(), lr=self.lr)

        self._global_updates = 0

    def _resolve_device(self, device_str: str) -> torch.device:
        s = str(device_str).lower()
        if s == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(s)

    def parameters(self):
        if self.is_discrete:
            return list(self.pi_net.parameters()) + list(self.v_net.parameters())
        return list(self.pi_mean.parameters()) + [self.pi_log_std] + list(self.v_net.parameters())

    def act(self, obs, *, deterministic: bool):
        obs_t = _to_tensor(np.asarray(obs, dtype=np.float32).reshape(1, -1), self.device)

        with torch.no_grad():
            if self.is_discrete:
                logits = self.pi_net(obs_t)
                dist = torch.distributions.Categorical(logits=logits)
                if deterministic:
                    action = torch.argmax(logits, dim=-1)
                else:
                    action = dist.sample()
                return {"action": int(action.item())}
            else:
                mean = self.pi_mean(obs_t)
                log_std = self.pi_log_std.expand_as(mean)
                dist = SquashedNormal(mean, log_std)
                a01 = dist.sample() if not deterministic else torch.tanh(mean)
                # scale from [-1,1] to [low,high]
                action = self._rescale_action(a01)
                return {"action": action.cpu().numpy().reshape(self.act_space.shape)}

    def _rescale_action(self, a01: torch.Tensor) -> torch.Tensor:
        # a01 in [-1, 1]
        return self.act_low + (a01 + 1.0) * 0.5 * (self.act_high - self.act_low)

    def _unscale_action(self, a: torch.Tensor) -> torch.Tensor:
        # a in [low, high] -> [-1, 1]
        return 2.0 * (a - self.act_low) / (self.act_high - self.act_low + 1e-8) - 1.0

    def update(self, traj) -> dict[str, float]:
        # Build tensors
        obs_np = _flatten_obs_list(traj.obs)  # (T, obs_dim)
        next_obs_np = _flatten_obs_list(traj.next_obs)  # (T, obs_dim)
        T = obs_np.shape[0]

        obs = _to_tensor(obs_np, self.device)
        next_obs = _to_tensor(next_obs_np, self.device)
        rewards = _to_tensor(np.asarray(traj.rewards, dtype=np.float32), self.device)
        dones = _to_tensor(np.asarray(traj.dones, dtype=np.float32), self.device)

        # Actions tensor
        if self.is_discrete:
            actions = torch.as_tensor(np.asarray(traj.actions, dtype=np.int64), device=self.device)
        else:
            act_np = np.asarray(traj.actions, dtype=np.float32).reshape(T, -1)
            actions = _to_tensor(act_np, self.device)

        # Compute values
        with torch.no_grad():
            values = self.v_net(obs).squeeze(-1)  # (T,)
            next_values = self.v_net(next_obs).squeeze(-1)

        # GAE-Lambda advantage
        advantages, returns = self._compute_gae(rewards, dones, values, next_values)

        # Old log-prob under current policy BEFORE updates (standard PPO approximation)
        with torch.no_grad():
            old_logp = self._log_prob(obs, actions)

        batch = PPOBatch(
            obs=obs,
            actions=actions,
            advantages=advantages,
            returns=returns,
            old_logp=old_logp,
        )

        # Normalize advantages
        batch.advantages = (batch.advantages - batch.advantages.mean()) / (
            batch.advantages.std() + 1e-8
        )

        # PPO update
        pi_losses = []
        v_losses = []
        entropies = []
        clip_fracs = []

        idx = torch.randperm(T, device=self.device)
        mb = max(1, self.minibatch_size)

        for _ in range(self.update_epochs):
            for start in range(0, T, mb):
                mb_idx = idx[start : start + mb]
                metrics = self._update_minibatch(
                    obs=batch.obs[mb_idx],
                    actions=batch.actions[mb_idx],
                    advantages=batch.advantages[mb_idx],
                    returns=batch.returns[mb_idx],
                    old_logp=batch.old_logp[mb_idx],
                )
                pi_losses.append(metrics["loss_pi"])
                v_losses.append(metrics["loss_v"])
                entropies.append(metrics["entropy"])
                clip_fracs.append(metrics["clip_frac"])

        self._global_updates += 1

        return {
            "loss_pi": float(np.mean(pi_losses)) if pi_losses else 0.0,
            "loss_v": float(np.mean(v_losses)) if v_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
            "clip_frac": float(np.mean(clip_fracs)) if clip_fracs else 0.0,
            "adv_mean": float(batch.advantages.mean().item()),
        }

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        T = rewards.shape[0]
        adv = torch.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(T)):
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_values[t] * nonterminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * nonterminal * gae
            adv[t] = gae
        returns = adv + values
        return adv, returns

    def _dist_and_entropy(self, obs: torch.Tensor):
        if self.is_discrete:
            logits = self.pi_net(obs)
            dist = torch.distributions.Categorical(logits=logits)
            entropy = dist.entropy()
            return dist, entropy
        mean = self.pi_mean(obs)
        log_std = self.pi_log_std.expand_as(mean)
        dist = SquashedNormal(mean, log_std)
        # Approx entropy: entropy of base Normal (not exact for tanh-squash).
        base = torch.distributions.Normal(mean, torch.exp(torch.clamp(log_std, -20.0, 2.0)))
        entropy = base.entropy().sum(dim=-1)
        return dist, entropy

    def _log_prob(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if self.is_discrete:
            logits = self.pi_net(obs)
            dist = torch.distributions.Categorical(logits=logits)
            return dist.log_prob(actions)
        # Box: actions are in env scale; unscale to [-1,1] for squashed log_prob
        a = actions.reshape(actions.shape[0], -1)
        a01 = self._unscale_action(a)
        mean = self.pi_mean(obs)
        log_std = self.pi_log_std.expand_as(mean)
        dist = SquashedNormal(mean, log_std)
        return dist.log_prob(a01)

    def _update_minibatch(
        self,
        *,
        obs: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        old_logp: torch.Tensor,
    ) -> dict[str, float]:
        dist, entropy = self._dist_and_entropy(obs)
        if self.is_discrete:
            logp = dist.log_prob(actions)
        else:
            # unscale to [-1,1] then compute squashed log_prob
            a = actions.reshape(actions.shape[0], -1)
            a01 = self._unscale_action(a)
            logp = dist.log_prob(a01)

        ratio = torch.exp(logp - old_logp)
        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages
        loss_pi = -torch.mean(torch.min(unclipped, clipped))

        values = self.v_net(obs).squeeze(-1)
        loss_v = F.mse_loss(values, returns)

        ent = entropy.mean()
        loss = loss_pi + self.vf_coef * loss_v - self.entropy_coef * ent

        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optim.step()

        clip_frac = torch.mean((torch.abs(ratio - 1.0) > self.clip_range).float()).item()

        return {
            "loss_pi": float(loss_pi.item()),
            "loss_v": float(loss_v.item()),
            "entropy": float(ent.item()),
            "clip_frac": float(clip_frac),
        }

    def on_episode_end(self, info: dict) -> None:
        # optional hook; can be used for episodic logging if env supplies returns in info
        pass

    def state_dict(self) -> dict:
        if self.is_discrete:
            return {
                "pi": self.pi_net.state_dict(),
                "v": self.v_net.state_dict(),
                "optim": self.optim.state_dict(),
                "global_updates": int(self._global_updates),
                "is_discrete": True,
            }
        return {
            "pi_mean": self.pi_mean.state_dict(),
            "pi_log_std": self.pi_log_std.detach().cpu(),
            "v": self.v_net.state_dict(),
            "optim": self.optim.state_dict(),
            "global_updates": int(self._global_updates),
            "is_discrete": False,
        }

    def load_state_dict(self, state: dict) -> None:
        self._global_updates = int(state.get("global_updates", 0))
        is_disc = bool(state.get("is_discrete", True))

        if is_disc:
            self.pi_net.load_state_dict(state["pi"])
            self.v_net.load_state_dict(state["v"])
            self.optim.load_state_dict(state["optim"])
            return

        self.pi_mean.load_state_dict(state["pi_mean"])
        self.v_net.load_state_dict(state["v"])
        self.pi_log_std.data.copy_(state["pi_log_std"].to(self.device))
        self.optim.load_state_dict(state["optim"])
