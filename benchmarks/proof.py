"""Reproducible proof: Spiking DQN vs Q-Learning vs Random on maze navigation.

Run with:
    python benchmarks/proof.py

This produces exact numbers with fixed seeds showing that:
1. The spiking neural network actually learns (success rate improves over episodes)
2. It competes with tabular Q-learning on maze navigation
3. Phase 2 local adaptation works on a new maze without full retraining
"""

import sys, os, time, random
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engram.spiking_dqn import SpikingDQNTrainer
from engram.environments.maze import MazeEnv

# Fix all seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class QLearning:
    """Tabular Q-learning baseline."""
    def __init__(self, n_actions, obs_dims, bins=6, lr=0.1, gamma=0.99, eps=0.3, eps_decay=0.995):
        self.n_actions = n_actions
        self.bins = bins
        self.lr = lr
        self.gamma = gamma
        self.eps = eps
        self.eps_decay = eps_decay
        self.q = {}

    def _key(self, obs):
        return tuple(min(int(v * self.bins), self.bins - 1) for v in obs)

    def act(self, obs):
        if random.random() < self.eps:
            return random.randint(0, self.n_actions - 1)
        k = self._key(obs)
        if k not in self.q:
            self.q[k] = np.zeros(self.n_actions)
        return int(np.argmax(self.q[k]))

    def learn(self, obs, action, reward, next_obs, done):
        k = self._key(obs)
        nk = self._key(next_obs)
        if k not in self.q: self.q[k] = np.zeros(self.n_actions)
        if nk not in self.q: self.q[nk] = np.zeros(self.n_actions)
        target = reward + (0 if done else self.gamma * np.max(self.q[nk]))
        self.q[k][action] += self.lr * (target - self.q[k][action])
        self.eps = max(0.05, self.eps * self.eps_decay)


def run_agent(agent_name, agent, env, episodes, is_spiking=False):
    """Run an agent and collect per-episode metrics."""
    rewards = []
    successes = []
    for ep in range(episodes):
        obs = env.reset()
        total_r = 0
        done = False
        steps = 0
        while not done:
            if is_spiking:
                action = agent.select_action(obs)
                next_obs, reward, done, info = env.step(action)
                agent.replay.push(obs, action, reward, next_obs, float(done))
                agent.train_step()
                agent.total_steps += 1
                if agent.total_steps % agent.target_update == 0:
                    agent.target_net.load_state_dict(agent.policy_net.state_dict())
            else:
                action = agent.act(obs)
                next_obs, reward, done, info = env.step(action)
                if hasattr(agent, 'learn'):
                    agent.learn(obs, action, reward, next_obs, done)

            obs = next_obs
            total_r += reward
            steps += 1

        rewards.append(total_r)
        successes.append(info.get('reached_goal', total_r > 5.0))

    return rewards, successes


def windowed_rate(arr, window=20):
    """Compute windowed success rate."""
    result = []
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        chunk = arr[start:i+1]
        result.append(sum(chunk) / len(chunk))
    return result


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    EPISODES = 150
    MAZE_SIZE = 4

    print_section("ENGRAM PROOF: Spiking DQN vs Q-Learning vs Random")
    print(f"  Maze: {MAZE_SIZE}x{MAZE_SIZE} procedural (seed={SEED})")
    print(f"  Episodes: {EPISODES}")
    print(f"  All seeds fixed for reproducibility")

    # Show the maze
    env = MazeEnv(width=MAZE_SIZE, height=MAZE_SIZE, seed=SEED)
    env.reset()
    print(f"\n  Maze layout:")
    for line in env.render().split('\n'):
        print(f"    {line}")

    # ── RANDOM BASELINE ──
    print_section("1. RANDOM BASELINE")
    env_r = MazeEnv(width=MAZE_SIZE, height=MAZE_SIZE, seed=SEED)
    t0 = time.time()

    class RandomAgent:
        def act(self, obs): return random.randint(0, 3)

    r_rewards, r_success = run_agent("Random", RandomAgent(), env_r, EPISODES)
    r_time = time.time() - t0
    r_rate = windowed_rate(r_success)
    print(f"  Time: {r_time:.1f}s")
    print(f"  Total success: {sum(r_success)}/{EPISODES} ({sum(r_success)/EPISODES*100:.1f}%)")
    print(f"  Last-20 success: {r_rate[-1]*100:.1f}%")
    print(f"  Avg reward (last 20): {np.mean(r_rewards[-20:]):.2f}")

    # ── Q-LEARNING ──
    print_section("2. TABULAR Q-LEARNING")
    env_q = MazeEnv(width=MAZE_SIZE, height=MAZE_SIZE, seed=SEED)
    q_agent = QLearning(4, 10)
    t0 = time.time()
    q_rewards, q_success = run_agent("Q-Learning", q_agent, env_q, EPISODES)
    q_time = time.time() - t0
    q_rate = windowed_rate(q_success)
    print(f"  Time: {q_time:.1f}s")
    print(f"  Total success: {sum(q_success)}/{EPISODES} ({sum(q_success)/EPISODES*100:.1f}%)")
    print(f"  Last-20 success: {q_rate[-1]*100:.1f}%")
    print(f"  Avg reward (last 20): {np.mean(q_rewards[-20:]):.2f}")
    print(f"  Q-table entries: {len(q_agent.q)}")

    # ── SPIKING DQN ──
    print_section("3. SPIKING DQN (Surrogate Gradients)")
    env_s = MazeEnv(width=MAZE_SIZE, height=MAZE_SIZE, seed=SEED)
    torch.manual_seed(SEED)
    spiking = SpikingDQNTrainer(
        obs_dim=10, num_actions=4, hidden=64, num_steps=8,
        lr=5e-4, gamma=0.99, epsilon_decay=EPISODES * 60,
        target_update=300, buffer_size=20000,
    )
    t0 = time.time()
    s_rewards, s_success = run_agent("SpikingDQN", spiking, env_s, EPISODES, is_spiking=True)
    s_time = time.time() - t0
    s_rate = windowed_rate(s_success)
    print(f"  Time: {s_time:.1f}s")
    print(f"  Total success: {sum(s_success)}/{EPISODES} ({sum(s_success)/EPISODES*100:.1f}%)")
    print(f"  Last-20 success: {s_rate[-1]*100:.1f}%")
    print(f"  Avg reward (last 20): {np.mean(s_rewards[-20:]):.2f}")

    # ── LEARNING CURVES ──
    print_section("LEARNING CURVES (windowed success rate)")
    print(f"  {'Episode':>8} {'Random':>8} {'Q-Learn':>8} {'Spiking':>8}")
    print(f"  {'-'*36}")
    checkpoints = [10, 20, 30, 50, 75, 100, 125, EPISODES]
    for ep in checkpoints:
        if ep <= len(r_rate):
            print(f"  {ep:>8} {r_rate[ep-1]*100:>7.1f}% {q_rate[ep-1]*100:>7.1f}% {s_rate[ep-1]*100:>7.1f}%")

    # ── PHASE 2: ONLINE ADAPTATION ──
    print_section("4. PHASE 2: ADAPTATION TO NEW MAZE")
    print(f"  Training the spiking network on a NEW maze layout")
    print(f"  using only output-layer local updates (no backprop)")

    env2 = MazeEnv(width=MAZE_SIZE, height=MAZE_SIZE, seed=99)
    env2.reset()
    print(f"\n  New maze:")
    for line in env2.render().split('\n'):
        print(f"    {line}")

    print()
    adapt_result = spiking.adapt_phase2(env2, episodes=50, verbose=True, print_every=10)
    print(f"\n  {adapt_result.summary()}")

    # ── COMPARISON TABLE ──
    print_section("FINAL COMPARISON")
    print(f"  {'Metric':<30} {'Random':>10} {'Q-Learn':>10} {'Spiking':>10}")
    print(f"  {'-'*62}")
    print(f"  {'Success rate (all eps)':<30} {sum(r_success)/EPISODES*100:>9.1f}% {sum(q_success)/EPISODES*100:>9.1f}% {sum(s_success)/EPISODES*100:>9.1f}%")
    print(f"  {'Success rate (last 20)':<30} {r_rate[-1]*100:>9.1f}% {q_rate[-1]*100:>9.1f}% {s_rate[-1]*100:>9.1f}%")
    print(f"  {'Avg reward (last 20)':<30} {np.mean(r_rewards[-20:]):>10.2f} {np.mean(q_rewards[-20:]):>10.2f} {np.mean(s_rewards[-20:]):>10.2f}")
    print(f"  {'Wall time':<30} {r_time:>9.1f}s {q_time:>9.1f}s {s_time:>9.1f}s")
    print(f"  {'Phase 2 adapt (new maze)':<30} {'N/A':>10} {'N/A':>10} {adapt_result.success_rate_last_20*100:>9.1f}%")

    # ── VERDICT ──
    print_section("VERDICT")
    if s_rate[-1] > r_rate[-1] + 0.05:
        print("  [PASS] Spiking DQN learns significantly better than random")
    else:
        print("  [FAIL] Spiking DQN does not beat random")

    if s_rate[-1] > q_rate[-1] * 0.5:
        print("  [PASS] Spiking DQN is competitive with Q-Learning (>50% of Q-Learn performance)")
    else:
        print("  [PARTIAL] Spiking DQN underperforms Q-Learning by >50%")

    if adapt_result.success_rate_last_20 > 0.3:
        print("  [PASS] Phase 2 local adaptation works on new maze (>30% success)")
    else:
        print("  [FAIL] Phase 2 local adaptation did not succeed")

    print()


if __name__ == "__main__":
    main()
