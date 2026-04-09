"""Streaming anomaly detection environment.

Tests the system's ability to learn normal patterns and detect anomalies
in a continuous data stream -- the exact use case where event-driven
processing and continual learning outperform batch-trained models.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class AnomalyStreamEnv:
    """Streaming sensor data with rare anomalies.

    The environment produces a stream of sensor readings. Most are normal
    (drawn from learned distributions). Occasionally, an anomaly occurs
    (distribution shift, spike, dropout). The agent must classify each
    reading as normal (action=0) or anomaly (action=1).

    This is the killer use case for brain-inspired architectures:
    - Event-driven: only process when something changes
    - Continual: normal distribution drifts over time
    - Memory: must remember what "normal" looks like
    - Prediction: anomalies are detected as prediction errors

    Observation (sensor_dims):
      Sensor readings in [0, 1].

    Actions:
      0 = normal, 1 = anomaly

    Rewards:
      +1.0  correct detection (true positive or true negative)
      -1.0  false alarm (false positive)
      -2.0  missed anomaly (false negative)
    """

    def __init__(
        self,
        sensor_dims: int = 8,
        anomaly_rate: float = 0.05,
        drift_rate: float = 0.001,
        readings_per_episode: int = 200,
        seed: Optional[int] = None,
    ):
        self.sensor_dims = sensor_dims
        self.anomaly_rate = anomaly_rate
        self.drift_rate = drift_rate
        self.readings_per_episode = readings_per_episode
        self.rng = np.random.RandomState(seed)

        # Normal distribution parameters (drift over time)
        self.normal_mean = self.rng.rand(sensor_dims) * 0.6 + 0.2
        self.normal_std = np.full(sensor_dims, 0.05)

        self.is_anomaly = False
        self.step_count = 0
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.true_negatives = 0
        self.total_reward = 0.0

    def reset(self, seed: int | None = None) -> list[float]:
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        self.normal_mean = self.rng.rand(self.sensor_dims) * 0.6 + 0.2
        self.step_count = 0
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.true_negatives = 0
        self.total_reward = 0.0
        return self._generate_reading()

    def _generate_reading(self) -> list[float]:
        """Generate a sensor reading (normal or anomalous)."""
        # Drift the normal distribution slowly
        self.normal_mean += self.rng.randn(self.sensor_dims) * self.drift_rate
        self.normal_mean = np.clip(self.normal_mean, 0.1, 0.9)

        self.is_anomaly = self.rng.rand() < self.anomaly_rate

        if self.is_anomaly:
            # Generate anomalous reading (one of several anomaly types)
            anomaly_type = self.rng.randint(0, 3)
            if anomaly_type == 0:
                # Spike: one sensor jumps to extreme
                reading = self.rng.randn(self.sensor_dims) * self.normal_std + self.normal_mean
                spike_idx = self.rng.randint(0, self.sensor_dims)
                reading[spike_idx] = self.rng.choice([0.0, 1.0])
            elif anomaly_type == 1:
                # Distribution shift: all sensors offset
                reading = self.rng.randn(self.sensor_dims) * self.normal_std + self.normal_mean + 0.3
            else:
                # Noise burst: high variance
                reading = self.rng.randn(self.sensor_dims) * 0.3 + self.normal_mean
        else:
            reading = self.rng.randn(self.sensor_dims) * self.normal_std + self.normal_mean

        return np.clip(reading, 0.0, 1.0).tolist()

    def step(self, action: int) -> tuple[list[float], float, bool, dict]:
        predicted_anomaly = action == 1

        if self.is_anomaly and predicted_anomaly:
            reward = 1.0
            self.true_positives += 1
        elif not self.is_anomaly and not predicted_anomaly:
            reward = 1.0
            self.true_negatives += 1
        elif not self.is_anomaly and predicted_anomaly:
            reward = -1.0
            self.false_positives += 1
        else:  # anomaly but not detected
            reward = -2.0
            self.false_negatives += 1

        self.total_reward += reward
        self.step_count += 1
        done = self.step_count >= self.readings_per_episode

        total = self.true_positives + self.false_positives + self.false_negatives + self.true_negatives
        accuracy = (self.true_positives + self.true_negatives) / max(total, 1)

        info = {
            "is_anomaly": self.is_anomaly,
            "predicted_anomaly": predicted_anomaly,
            "accuracy": accuracy,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "reached_goal": accuracy > 0.7,
            "total_reward": self.total_reward,
        }

        obs = self._generate_reading() if not done else [0.5] * self.sensor_dims
        return obs, reward, done, info

    def render(self) -> str:
        total = self.true_positives + self.false_positives + self.false_negatives + self.true_negatives
        acc = (self.true_positives + self.true_negatives) / max(total, 1) * 100
        return (
            f"Anomaly Stream | Step {self.step_count}/{self.readings_per_episode} | "
            f"Acc: {acc:.1f}% | TP:{self.true_positives} FP:{self.false_positives} "
            f"FN:{self.false_negatives} TN:{self.true_negatives}"
        )
