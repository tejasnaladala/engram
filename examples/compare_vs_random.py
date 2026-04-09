"""Compare Engram agent vs random baseline on all environments.

Usage:
    python examples/compare_vs_random.py

Shows whether the spiking network actually learns better than chance.
"""

from engram import Runtime, Trainer, RandomBaseline
from engram.environments.grid_world import GridWorldEnv
from engram.environments.maze import MazeEnv
from engram.environments.pattern_learner import PatternLearnerEnv
from engram.environments.anomaly_stream import AnomalyStreamEnv


def run_comparison(env_name: str, env, input_dims: int, num_actions: int, episodes: int) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {env_name}")
    print(f"{'=' * 60}")

    # Engram
    brain = Runtime(input_dims=input_dims, num_actions=num_actions, seed=42)
    trainer = Trainer(brain, env, ticks_per_step=3)
    print(f"\n  [Engram] Training {episodes} episodes...")
    engram_result = trainer.train(episodes=episodes)

    # Random baseline
    env.reset(seed=42)
    random_agent = RandomBaseline(num_actions=num_actions)
    random_trainer = Trainer(random_agent, env, ticks_per_step=1)
    print(f"  [Random] Running {episodes} episodes...")
    random_result = random_trainer.train(episodes=episodes)

    # Compare
    print(f"\n  Results:")
    print(f"  {'Metric':<25} {'Engram':>10} {'Random':>10} {'Delta':>10}")
    print(f"  {'-'*55}")

    eng_r = engram_result.avg_reward_last_10
    rnd_r = random_result.avg_reward_last_10
    print(f"  {'Avg Reward (last 10)':<25} {eng_r:>10.2f} {rnd_r:>10.2f} {eng_r - rnd_r:>+10.2f}")

    eng_s = engram_result.success_rate * 100
    rnd_s = random_result.success_rate * 100
    print(f"  {'Success Rate':<25} {eng_s:>9.1f}% {rnd_s:>9.1f}% {eng_s - rnd_s:>+9.1f}%")

    eng_st = engram_result.avg_steps_last_10
    rnd_st = random_result.avg_steps_last_10
    print(f"  {'Avg Steps (last 10)':<25} {eng_st:>10.0f} {rnd_st:>10.0f} {eng_st - rnd_st:>+10.0f}")


def main() -> None:
    print("=" * 60)
    print("  ENGRAM vs RANDOM BASELINE -- Full Comparison")
    print("=" * 60)

    run_comparison(
        "Grid World (8x8)",
        GridWorldEnv(size=8, num_walls=8, num_hazards=3, seed=42),
        input_dims=8, num_actions=4, episodes=100,
    )

    run_comparison(
        "Maze Navigation (5x5)",
        MazeEnv(width=5, height=5, seed=42),
        input_dims=10, num_actions=4, episodes=100,
    )

    run_comparison(
        "Pattern Recognition (4 classes)",
        PatternLearnerEnv(num_classes=4, pattern_size=16, seed=42),
        input_dims=16, num_actions=4, episodes=50,
    )

    run_comparison(
        "Anomaly Detection",
        AnomalyStreamEnv(sensor_dims=8, anomaly_rate=0.08, seed=42),
        input_dims=8, num_actions=2, episodes=30,
    )

    print(f"\n{'=' * 60}")
    print("  COMPLETE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
