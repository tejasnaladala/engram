"""Train an Engram agent on online pattern classification.

Usage:
    python examples/train_pattern_recognition.py

The agent learns to classify 4 classes of patterns from a streaming
data source. No separate training phase -- it learns by observing
the association between patterns and reward signals.

This tests what spiking networks excel at:
- Fast few-shot association
- Continual learning without forgetting
- Noise-robust pattern completion
"""

from engram import Runtime, Trainer
from engram.environments.pattern_learner import PatternLearnerEnv


def main() -> None:
    print("=" * 60)
    print("  ENGRAM -- Online Pattern Recognition Training")
    print("=" * 60)
    print()

    env = PatternLearnerEnv(
        num_classes=4,
        pattern_size=16,
        noise_level=0.15,
        patterns_per_episode=50,
        seed=42,
    )
    brain = Runtime(input_dims=16, num_actions=4, seed=42)

    print(f"Classes: {env.num_classes}, Pattern size: {env.pattern_size}")
    print(f"Noise level: {env.noise_level}, Patterns/episode: {env.patterns_per_episode}")
    print()

    trainer = Trainer(brain, env, ticks_per_step=3)
    print("Training for 100 episodes...")
    print("-" * 60)
    result = trainer.train(episodes=100, verbose=True, print_every=10)

    print("-" * 60)
    print()
    print(result.summary())


if __name__ == "__main__":
    main()
