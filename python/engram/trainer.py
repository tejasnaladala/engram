"""Training loop and experiment runner for Engram agents."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from engram.runtime import Runtime


class Environment(Protocol):
    """Protocol for Engram-compatible environments."""

    def reset(self, seed: int | None = None) -> list[float]: ...
    def step(self, action: int) -> tuple[list[float], float, bool, dict]: ...


@dataclass
class EpisodeResult:
    """Metrics from a single episode."""

    episode: int
    total_reward: float
    steps: int
    reached_goal: bool
    prediction_error_avg: float
    vetoes: int


@dataclass
class TrainingResult:
    """Metrics from a full training run."""

    episodes: list[EpisodeResult] = field(default_factory=list)
    wall_time_s: float = 0.0
    total_steps: int = 0
    total_spikes: int = 0

    @property
    def rewards(self) -> list[float]:
        return [e.total_reward for e in self.episodes]

    @property
    def success_rate(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(1 for e in self.episodes if e.reached_goal) / len(self.episodes)

    @property
    def avg_reward_last_10(self) -> float:
        recent = self.rewards[-10:]
        return sum(recent) / max(len(recent), 1)

    @property
    def avg_steps_last_10(self) -> float:
        recent = [e.steps for e in self.episodes[-10:]]
        return sum(recent) / max(len(recent), 1)

    def summary(self) -> str:
        lines = [
            f"Training: {len(self.episodes)} episodes in {self.wall_time_s:.1f}s",
            f"  Success rate: {self.success_rate * 100:.1f}%",
            f"  Avg reward (last 10): {self.avg_reward_last_10:.2f}",
            f"  Avg steps  (last 10): {self.avg_steps_last_10:.0f}",
            f"  Total steps: {self.total_steps}",
            f"  Total spikes: {self.total_spikes}",
        ]
        return "\n".join(lines)

    def reward_curve(self, window: int = 10) -> list[float]:
        """Smoothed reward curve for plotting."""
        rewards = self.rewards
        if len(rewards) < window:
            return rewards
        return [
            sum(rewards[max(0, i - window) : i]) / min(i, window)
            for i in range(1, len(rewards) + 1)
        ]


class Trainer:
    """Trains an Engram agent on an environment.

    Handles the episode loop, reward delivery, metrics collection,
    and optional printing/logging.

    Example:
        from engram import Runtime, Trainer
        from engram.environments.grid_world import GridWorldEnv

        env = GridWorldEnv(size=8)
        brain = Runtime(input_dims=8, num_actions=4)
        trainer = Trainer(brain, env)
        result = trainer.train(episodes=200, verbose=True)
        print(result.summary())
    """

    def __init__(
        self,
        brain: Runtime,
        env: Environment,
        ticks_per_step: int = 5,
    ):
        self.brain = brain
        self.env = env
        self.ticks_per_step = ticks_per_step

    def train(
        self,
        episodes: int = 100,
        verbose: bool = False,
        print_every: int = 10,
    ) -> TrainingResult:
        """Run a full training loop.

        Args:
            episodes: Number of episodes to run.
            verbose: Print progress to stdout.
            print_every: Print every N episodes.

        Returns:
            TrainingResult with all episode metrics.
        """
        result = TrainingResult()
        start = time.time()

        for ep in range(episodes):
            ep_result = self._run_episode(ep)
            result.episodes.append(ep_result)
            result.total_steps += ep_result.steps

            if verbose and (ep + 1) % print_every == 0:
                recent_reward = sum(
                    e.total_reward for e in result.episodes[-print_every:]
                ) / print_every
                recent_success = sum(
                    1 for e in result.episodes[-print_every:] if e.reached_goal
                ) / print_every
                print(
                    f"  ep {ep + 1:4d} | "
                    f"reward {recent_reward:7.2f} | "
                    f"success {recent_success * 100:5.1f}% | "
                    f"steps {ep_result.steps:4d} | "
                    f"PE {ep_result.prediction_error_avg:.3f} | "
                    f"vetoes {ep_result.vetoes}"
                )

        result.wall_time_s = time.time() - start
        result.total_spikes = self.brain.total_spikes
        return result

    def _run_episode(self, episode_num: int) -> EpisodeResult:
        """Run a single episode."""
        obs = self.env.reset()
        total_reward = 0.0
        steps = 0
        done = False
        pe_sum = 0.0
        pe_count = 0
        vetoes_before = self.brain.total_vetoes

        prev_reward = 0.0
        while not done:
            # Run multiple ticks per environment step for richer neural dynamics
            action = 0
            for _ in range(self.ticks_per_step):
                action = self.brain.step(obs, reward=prev_reward)

            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            steps += 1

            # Deliver the actual reward from this step
            self.brain.reward(reward)
            prev_reward = reward

            pe_sum += self.brain.prediction_error
            pe_count += 1

        # Signal episode boundary (keeps memories, resets transient state)
        self.brain.end_episode()

        return EpisodeResult(
            episode=episode_num,
            total_reward=total_reward,
            steps=steps,
            reached_goal=info.get("reached_goal", total_reward > 5.0),
            prediction_error_avg=pe_sum / max(pe_count, 1),
            vetoes=self.brain.total_vetoes - vetoes_before,
        )

    def evaluate(
        self,
        episodes: int = 20,
        verbose: bool = False,
    ) -> TrainingResult:
        """Evaluate without learning (snapshot current performance)."""
        # For now, evaluation runs the same as training since the
        # brain always learns. Future: add a freeze mode.
        return self.train(episodes=episodes, verbose=verbose, print_every=5)


class RandomBaseline:
    """Random action agent for comparison."""

    def __init__(self, num_actions: int = 4):
        self.num_actions = num_actions

    def step(self, obs: list[float], reward: float = 0.0) -> int:
        return np.random.randint(0, self.num_actions)

    def reward(self, value: float) -> None:
        pass

    def end_episode(self) -> None:
        pass

    @property
    def prediction_error(self) -> float:
        return 0.0

    @property
    def total_vetoes(self) -> int:
        return 0

    @property
    def total_spikes(self) -> int:
        return 0
