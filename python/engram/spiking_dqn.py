"""Spiking DQN with surrogate gradients -- the training engine that actually works.

This implements the proven recipe from the spiking RL literature:
- LIF neurons with surrogate gradients (arctangent) for backpropagation
- Non-spiking leaky integrator output neurons (membrane voltage = Q-values)
- Experience replay buffer
- Target network for stability
- Soft reset mechanism

Based on DSQN (Chen et al. 2022) which beat standard DQN on 17 Atari games.

Phase 1: Train with surrogate gradients (standard DQN loop)
Phase 2: Switch to local adaptation for online continual learning
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class Environment(Protocol):
    def reset(self, seed: int | None = None) -> list[float]: ...
    def step(self, action: int) -> tuple[list[float], float, bool, dict]: ...


class SpikingQNetwork(nn.Module):
    """Spiking Q-Network using surrogate gradients.

    Architecture:
    - Direct input encoding (continuous values as current)
    - 1-2 hidden LIF layers with soft reset and learnable beta
    - Non-spiking leaky integrator output (membrane voltage = Q-values)
    - Max membrane voltage across timesteps = final Q-value estimate
    """

    def __init__(self, obs_dim: int, num_actions: int, hidden: int = 64, num_steps: int = 8):
        super().__init__()
        self.num_steps = num_steps
        self.num_actions = num_actions

        spike_grad = surrogate.atan(alpha=2.0)

        self.fc1 = nn.Linear(obs_dim, hidden)
        self.lif1 = snn.Leaky(
            beta=0.9, learn_beta=True, spike_grad=spike_grad,
            reset_mechanism='subtract',
        )

        self.fc2 = nn.Linear(hidden, hidden // 2)
        self.lif2 = snn.Leaky(
            beta=0.85, learn_beta=True, spike_grad=spike_grad,
            reset_mechanism='subtract',
        )

        self.fc_out = nn.Linear(hidden // 2, num_actions)
        self.li_out = snn.Leaky(
            beta=0.95, learn_beta=True, spike_grad=spike_grad,
            reset_mechanism='none', output=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem_out = self.li_out.init_leaky()
        max_mem = torch.full((batch, self.num_actions), -1e9, device=x.device)

        for step in range(self.num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            cur_out = self.fc_out(spk2)
            spk_out, mem_out = self.li_out(cur_out, mem_out)
            max_mem = torch.max(max_mem, mem_out)

        return max_mem


class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(obs)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_obs)),
            torch.FloatTensor(dones),
        )

    def __len__(self):
        return len(self.buffer)


@dataclass
class TrainResult:
    episode_rewards: list[float] = field(default_factory=list)
    episode_successes: list[bool] = field(default_factory=list)
    episode_steps: list[int] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return sum(self.episode_successes) / max(len(self.episode_successes), 1)

    @property
    def avg_reward_last_20(self) -> float:
        r = self.episode_rewards[-20:]
        return sum(r) / max(len(r), 1)

    @property
    def success_rate_last_20(self) -> float:
        s = self.episode_successes[-20:]
        return sum(s) / max(len(s), 1)

    def summary(self) -> str:
        return (
            f"Episodes: {len(self.episode_rewards)} | "
            f"Success: {self.success_rate*100:.1f}% | "
            f"Last-20 success: {self.success_rate_last_20*100:.1f}% | "
            f"Avg reward (last 20): {self.avg_reward_last_20:.2f}"
        )


class SpikingDQNTrainer:
    """Dual-phase spiking DQN trainer.

    Phase 1 (surrogate gradient): DQN training through spiking neurons.
    Phase 2 (online adaptation): Local updates on output layer only.
    """

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        hidden: int = 64,
        num_steps: int = 8,
        lr: float = 2.5e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 5000,
        target_update: int = 500,
        batch_size: int = 32,
        buffer_size: int = 10000,
    ):
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.batch_size = batch_size

        self.policy_net = SpikingQNetwork(obs_dim, num_actions, hidden, num_steps)
        self.target_net = SpikingQNetwork(obs_dim, num_actions, hidden, num_steps)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.requires_grad_(False)

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay = ReplayBuffer(buffer_size)
        self.total_steps = 0
        self.phase = 1
        self.adaptation_lr = 0.01

    def select_action(self, obs: list[float]) -> int:
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            max(0, 1 - self.total_steps / self.epsilon_decay)
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            q_values = self.policy_net(obs_t)
            return q_values.argmax(dim=1).item()

    def train_step(self) -> float | None:
        if len(self.replay) < self.batch_size:
            return None
        obs, actions, rewards, next_obs, dones = self.replay.sample(self.batch_size)
        q_values = self.policy_net(obs)
        q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(next_obs)
            target = rewards + self.gamma * next_q.max(dim=1).values * (1 - dones)
        loss = nn.SmoothL1Loss()(q_selected, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()

    def train_phase1(self, env, episodes=200, verbose=True, print_every=20) -> TrainResult:
        """Phase 1: Surrogate gradient DQN training."""
        self.phase = 1
        result = TrainResult()
        for ep in range(episodes):
            obs = env.reset()
            total_reward = 0.0
            steps = 0
            done = False
            while not done:
                action = self.select_action(obs)
                next_obs, reward, done, info = env.step(action)
                self.replay.push(obs, action, reward, next_obs, float(done))
                self.train_step()
                obs = next_obs
                total_reward += reward
                steps += 1
                self.total_steps += 1
                if self.total_steps % self.target_update == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
            result.episode_rewards.append(total_reward)
            result.episode_successes.append(info.get('reached_goal', total_reward > 5.0))
            result.episode_steps.append(steps)
            if verbose and (ep + 1) % print_every == 0:
                print(
                    f"  Phase 1 | ep {ep+1:4d} | "
                    f"reward {result.avg_reward_last_20:7.2f} | "
                    f"success {result.success_rate_last_20*100:5.1f}% | "
                    f"eps {self.epsilon:.3f} | steps {steps:4d}"
                )
        return result

    def adapt_phase2(self, env, episodes=50, verbose=True, print_every=10) -> TrainResult:
        """Phase 2: Online local adaptation (output layer only)."""
        self.phase = 2
        result = TrainResult()
        for param in self.policy_net.parameters():
            param.requires_grad = False
        for param in self.policy_net.fc_out.parameters():
            param.requires_grad = True
        adapt_opt = torch.optim.SGD(self.policy_net.fc_out.parameters(), lr=self.adaptation_lr)

        for ep in range(episodes):
            obs = env.reset()
            total_reward = 0.0
            steps = 0
            done = False
            while not done:
                if random.random() < 0.1:
                    action = random.randint(0, self.num_actions - 1)
                else:
                    with torch.no_grad():
                        q = self.policy_net(torch.FloatTensor(obs).unsqueeze(0))
                        action = q.argmax(dim=1).item()
                next_obs, reward, done, info = env.step(action)
                if abs(reward) > 0.05:
                    obs_t = torch.FloatTensor(obs).unsqueeze(0)
                    q = self.policy_net(obs_t)
                    with torch.no_grad():
                        tgt = reward + self.gamma * q.max().item() * (1 - float(done))
                    loss = nn.MSELoss()(q[0, action], torch.tensor(tgt))
                    adapt_opt.zero_grad()
                    loss.backward()
                    adapt_opt.step()
                obs = next_obs
                total_reward += reward
                steps += 1
            result.episode_rewards.append(total_reward)
            result.episode_successes.append(info.get('reached_goal', total_reward > 5.0))
            result.episode_steps.append(steps)
            if verbose and (ep + 1) % print_every == 0:
                print(
                    f"  Phase 2 | ep {ep+1:4d} | "
                    f"reward {result.avg_reward_last_20:7.2f} | "
                    f"success {result.success_rate_last_20*100:5.1f}% | steps {steps:4d}"
                )
        for param in self.policy_net.parameters():
            param.requires_grad = True
        return result

    def save(self, path: str):
        torch.save({
            'policy': self.policy_net.state_dict(),
            'target': self.target_net.state_dict(),
            'steps': self.total_steps,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, weights_only=True)
        self.policy_net.load_state_dict(ckpt['policy'])
        self.target_net.load_state_dict(ckpt['target'])
        self.total_steps = ckpt['steps']
