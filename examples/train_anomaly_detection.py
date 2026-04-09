"""Train an Engram agent for streaming anomaly detection.

Usage:
    python examples/train_anomaly_detection.py

The agent monitors a continuous sensor stream and must detect anomalies
(spikes, distribution shifts, noise bursts) in real-time. The normal
distribution drifts over time, requiring continual adaptation.

This is the killer use case for brain-inspired architectures:
- Event-driven: only compute when something changes
- Continual: adapts to distribution drift without retraining
- Memory: remembers what "normal" looks like
- Prediction: anomalies are detected as prediction errors
"""

from engram import Runtime, Trainer
from engram.environments.anomaly_stream import AnomalyStreamEnv


def main() -> None:
    print("=" * 60)
    print("  ENGRAM -- Streaming Anomaly Detection Training")
    print("=" * 60)
    print()

    env = AnomalyStreamEnv(
        sensor_dims=8,
        anomaly_rate=0.08,
        drift_rate=0.002,
        readings_per_episode=200,
        seed=42,
    )
    # 2 actions: normal (0) or anomaly (1)
    brain = Runtime(input_dims=8, num_actions=2, seed=42)

    print(f"Sensors: {env.sensor_dims}, Anomaly rate: {env.anomaly_rate*100:.0f}%")
    print(f"Drift rate: {env.drift_rate}, Readings/episode: {env.readings_per_episode}")
    print()

    trainer = Trainer(brain, env, ticks_per_step=2)
    print("Training for 50 episodes...")
    print("-" * 60)
    result = trainer.train(episodes=50, verbose=True, print_every=5)

    print("-" * 60)
    print()
    print(result.summary())


if __name__ == "__main__":
    main()
