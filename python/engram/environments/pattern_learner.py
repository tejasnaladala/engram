"""Pattern recognition environment for associative learning.

The agent sees a sequence of patterns and must learn to classify them.
Tests the associative memory and prediction capabilities of the brain.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class PatternLearnerEnv:
    """Online pattern classification from streaming data.

    The environment presents patterns (small binary images or vectors)
    and the agent must output the correct class. No separate training phase;
    the agent learns by observing the association between patterns and
    the reward signal indicating correct/incorrect classification.

    This tests what spiking networks are actually good at:
    - fast one-shot/few-shot association
    - continual learning of new classes without forgetting old ones
    - noise-robust pattern completion

    Observation (pattern_size dims):
      The pattern itself, as floats in [0, 1].

    Actions:
      0 to num_classes-1 = classify as that class

    Rewards:
      +1.0  correct classification
      -0.5  wrong classification
    """

    def __init__(
        self,
        num_classes: int = 4,
        pattern_size: int = 16,
        noise_level: float = 0.1,
        patterns_per_episode: int = 50,
        seed: Optional[int] = None,
    ):
        self.num_classes = num_classes
        self.pattern_size = pattern_size
        self.noise_level = noise_level
        self.patterns_per_episode = patterns_per_episode
        self.rng = np.random.RandomState(seed)

        # Generate canonical patterns for each class
        self.prototypes = self.rng.rand(num_classes, pattern_size).astype(np.float64)
        # Make them more distinct by binarizing
        self.prototypes = (self.prototypes > 0.5).astype(np.float64)

        self.current_class = 0
        self.current_pattern: list[float] = []
        self.step_count = 0
        self.correct = 0
        self.total = 0
        self.total_reward = 0.0

    def reset(self, seed: int | None = None) -> list[float]:
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        self.step_count = 0
        self.correct = 0
        self.total = 0
        self.total_reward = 0.0
        return self._next_pattern()

    def _next_pattern(self) -> list[float]:
        """Generate the next noisy pattern from a random class."""
        self.current_class = self.rng.randint(0, self.num_classes)
        base = self.prototypes[self.current_class].copy()
        # Add noise
        noise = self.rng.randn(self.pattern_size) * self.noise_level
        noisy = np.clip(base + noise, 0.0, 1.0)
        self.current_pattern = noisy.tolist()
        return self.current_pattern

    def step(self, action: int) -> tuple[list[float], float, bool, dict]:
        predicted_class = action % self.num_classes
        correct = predicted_class == self.current_class

        reward = 1.0 if correct else -0.5
        self.total_reward += reward
        self.total += 1
        if correct:
            self.correct += 1
        self.step_count += 1

        done = self.step_count >= self.patterns_per_episode

        info = {
            "correct": correct,
            "true_class": self.current_class,
            "predicted_class": predicted_class,
            "accuracy": self.correct / max(self.total, 1),
            "reached_goal": self.correct / max(self.total, 1) > 0.6,
            "total_reward": self.total_reward,
        }

        obs = self._next_pattern() if not done else self.current_pattern
        return obs, reward, done, info

    def render(self) -> str:
        # Show pattern as a small grid
        side = int(np.sqrt(self.pattern_size))
        if side * side != self.pattern_size:
            return f"Pattern (class {self.current_class}): {self.current_pattern[:8]}..."
        lines = [f"Class: {self.current_class}  Accuracy: {self.correct}/{self.total}"]
        for y in range(side):
            row = []
            for x in range(side):
                val = self.current_pattern[y * side + x]
                row.append("#" if val > 0.5 else ".")
            lines.append(" ".join(row))
        return "\n".join(lines)
