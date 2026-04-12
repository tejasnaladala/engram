"""Engram vs Q-Learning vs Random -- honest benchmark comparison.

Tests what brain-inspired online learning actually does better (or worse)
than standard RL on tasks that matter for adaptive systems.

Usage:
    python benchmarks/benchmark.py
"""

from __future__ import annotations

import time
import sys
import os
from dataclasses import dataclass

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engram import Runtime
from engram.environments.grid_world import GridWorldEnv
from engram.environments.pattern_learner import PatternLearnerEnv


# ============================================================
# BASELINES
# ============================================================

class RandomAgent:
    """Uniform random action selection."""

    def __init__(self, num_actions: int):
        self.num_actions = num_actions

    def act(self, obs: list[float]) -> int:
        return np.random.randint(0, self.num_actions)

    def learn(self, obs, action, reward, next_obs, done):
        pass

    def reset(self):
        pass


class QLearningAgent:
    """Tabular Q-learning with epsilon-greedy exploration.

    This is the standard RL baseline for discrete state/action spaces.
    Uses state discretization to handle continuous observations.
    """

    def __init__(
        self,
        num_actions: int,
        obs_dims: int,
        bins_per_dim: int = 6,
        lr: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.3,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05,
    ):
        self.num_actions = num_actions
        self.obs_dims = obs_dims
        self.bins = bins_per_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        # Q-table: discretized state -> action values
        self.q_table: dict[tuple, np.ndarray] = {}

    def _discretize(self, obs: list[float]) -> tuple:
        return tuple(min(int(v * self.bins), self.bins - 1) for v in obs)

    def _get_q(self, state: tuple) -> np.ndarray:
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.num_actions)
        return self.q_table[state]

    def act(self, obs: list[float]) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.num_actions)
        state = self._discretize(obs)
        q_vals = self._get_q(state)
        return int(np.argmax(q_vals))

    def learn(self, obs, action, reward, next_obs, done):
        state = self._discretize(obs)
        next_state = self._discretize(next_obs)
        q_vals = self._get_q(state)
        next_q = self._get_q(next_state)

        target = reward + (0.0 if done else self.gamma * np.max(next_q))
        q_vals[action] += self.lr * (target - q_vals[action])

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def reset(self):
        # Keep Q-table across episodes (continual learning)
        pass


class EngramAgent:
    """Wrapper around Engram Runtime for benchmark interface."""

    def __init__(self, input_dims: int, num_actions: int, ticks_per_step: int = 2):
        self.brain = Runtime(input_dims=input_dims, num_actions=num_actions, seed=42)
        self.ticks = ticks_per_step
        self.prev_reward = 0.0

    def act(self, obs: list[float]) -> int:
        action = 0
        for _ in range(self.ticks):
            action = self.brain.step(obs, reward=self.prev_reward)
        return action

    def learn(self, obs, action, reward, next_obs, done):
        self.brain.reward(reward)
        self.prev_reward = reward

    def reset(self):
        self.brain.end_episode()
        self.prev_reward = 0.0


# ============================================================
# BENCHMARK RUNNER
# ============================================================

@dataclass
class BenchResult:
    name: str
    rewards: list[float]
    successes: list[bool]
    steps_list: list[int]
    wall_time: float

    @property
    def avg_reward(self) -> float:
        return float(np.mean(self.rewards))

    @property
    def avg_reward_last_20(self) -> float:
        return float(np.mean(self.rewards[-20:])) if len(self.rewards) >= 20 else self.avg_reward

    @property
    def success_rate(self) -> float:
        return float(np.mean(self.successes))

    @property
    def success_rate_last_20(self) -> float:
        return float(np.mean(self.successes[-20:])) if len(self.successes) >= 20 else self.success_rate

    @property
    def avg_steps(self) -> float:
        return float(np.mean(self.steps_list))


def run_benchmark(agent, env, episodes: int, name: str) -> BenchResult:
    rewards = []
    successes = []
    steps_list = []
    start = time.time()

    for ep in range(episodes):
        obs = env.reset()
        agent.reset()
        total_r = 0.0
        steps = 0
        done = False

        while not done:
            action = agent.act(obs)
            next_obs, reward, done, info = env.step(action)
            agent.learn(obs, action, reward, next_obs, done)
            obs = next_obs
            total_r += reward
            steps += 1

        rewards.append(total_r)
        successes.append(info.get("reached_goal", total_r > 5.0))
        steps_list.append(steps)

    wall_time = time.time() - start
    return BenchResult(name, rewards, successes, steps_list, wall_time)


def print_comparison(results: list[BenchResult], title: str):
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}")
    print(f"  {'Agent':<14} {'Avg Reward':>10} {'Last-20 Rwd':>12} {'Success%':>9} {'Last-20 S%':>11} {'Avg Steps':>10} {'Time':>7}")
    print(f"  {'-' * 70}")
    for r in results:
        print(
            f"  {r.name:<14} "
            f"{r.avg_reward:>10.2f} "
            f"{r.avg_reward_last_20:>12.2f} "
            f"{r.success_rate * 100:>8.1f}% "
            f"{r.success_rate_last_20 * 100:>10.1f}% "
            f"{r.avg_steps:>10.1f} "
            f"{r.wall_time:>6.1f}s"
        )


# ============================================================
# BENCHMARK 1: GRID WORLD NAVIGATION
# ============================================================

def bench_grid_world():
    print("\n[1/3] Grid World Navigation (8x8, 100 episodes)")
    print("    Task: Navigate to goal, avoid walls and hazards")
    episodes = 100

    results = []
    for name, make_agent in [
        ("Random", lambda: RandomAgent(4)),
        ("Q-Learning", lambda: QLearningAgent(4, 8)),
        ("Engram", lambda: EngramAgent(8, 4)),
    ]:
        env = GridWorldEnv(size=8, num_walls=8, num_hazards=3, seed=42)
        agent = make_agent()
        print(f"    Running {name}...", end=" ", flush=True)
        r = run_benchmark(agent, env, episodes, name)
        print(f"done ({r.wall_time:.1f}s)")
        results.append(r)

    print_comparison(results, "BENCHMARK 1: Grid World Navigation")
    return results


# ============================================================
# BENCHMARK 2: CONTINUAL LEARNING (CATASTROPHIC FORGETTING)
# ============================================================

def bench_continual_learning():
    print("\n[2/3] Continual Learning -- Catastrophic Forgetting Test")
    print("    Phase 1: Train on layout A (50 eps)")
    print("    Phase 2: Train on layout B (50 eps)")
    print("    Phase 3: Test on layout A (20 eps) -- does it still work?")

    results_phase3 = []

    for name, make_agent in [
        ("Random", lambda: RandomAgent(4)),
        ("Q-Learning", lambda: QLearningAgent(4, 8)),
        ("Engram", lambda: EngramAgent(8, 4)),
    ]:
        print(f"    Running {name}...", end=" ", flush=True)
        agent = make_agent()

        # Phase 1: Train on layout A
        env_a = GridWorldEnv(size=8, num_walls=8, num_hazards=2, seed=100)
        run_benchmark(agent, env_a, 50, name)

        # Phase 2: Train on layout B (different seed = different layout)
        env_b = GridWorldEnv(size=8, num_walls=8, num_hazards=2, seed=200)
        run_benchmark(agent, env_b, 50, name)

        # Phase 3: Test on layout A again (WITHOUT retraining)
        env_a2 = GridWorldEnv(size=8, num_walls=8, num_hazards=2, seed=100)
        r = run_benchmark(agent, env_a2, 20, name)
        print(f"done (phase 3 success: {r.success_rate * 100:.0f}%)")
        results_phase3.append(r)

    print_comparison(results_phase3, "BENCHMARK 2: Continual Learning (Phase 3 -- recall of Layout A)")
    return results_phase3


# ============================================================
# BENCHMARK 3: ONLINE PATTERN CLASSIFICATION
# ============================================================

def bench_pattern_recognition():
    print("\n[3/3] Online Pattern Classification (4 classes, 50 episodes)")
    print("    Task: Classify noisy patterns from streaming data")
    print("    No separate training phase -- learns from reward signal only")
    episodes = 50

    results = []
    for name, make_agent in [
        ("Random", lambda: RandomAgent(4)),
        ("Q-Learning", lambda: QLearningAgent(4, 16, bins_per_dim=4)),
        ("Engram", lambda: EngramAgent(16, 4)),
    ]:
        env = PatternLearnerEnv(num_classes=4, pattern_size=16, noise_level=0.15, seed=42)
        agent = make_agent()
        print(f"    Running {name}...", end=" ", flush=True)
        r = run_benchmark(agent, env, episodes, name)
        print(f"done ({r.wall_time:.1f}s)")
        results.append(r)

    print_comparison(results, "BENCHMARK 3: Online Pattern Classification")
    return results


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 72)
    print("  ENGRAM BENCHMARK SUITE")
    print("  Engram vs Q-Learning vs Random Baseline")
    print("=" * 72)

    np.random.seed(42)

    r1 = bench_grid_world()
    r2 = bench_continual_learning()
    r3 = bench_pattern_recognition()

    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)

    # Verdict for each benchmark
    for title, results in [
        ("Grid World", r1),
        ("Continual Learning", r2),
        ("Pattern Classification", r3),
    ]:
        best = max(results, key=lambda r: r.avg_reward_last_20)
        engram = next(r for r in results if r.name == "Engram")
        qlearn = next(r for r in results if r.name == "Q-Learning")
        random = next(r for r in results if r.name == "Random")

        print(f"\n  {title}:")
        print(f"    Winner: {best.name} (reward: {best.avg_reward_last_20:.2f})")

        # Engram vs Q-Learning delta
        delta = engram.avg_reward_last_20 - qlearn.avg_reward_last_20
        print(f"    Engram vs Q-Learning: {delta:+.2f} reward")

        # Engram vs Random delta
        delta_r = engram.avg_reward_last_20 - random.avg_reward_last_20
        print(f"    Engram vs Random:     {delta_r:+.2f} reward")


if __name__ == "__main__":
    main()
