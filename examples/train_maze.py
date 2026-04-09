"""Train an Engram agent to solve procedurally generated mazes.

Usage:
    python examples/train_maze.py

Each episode generates a new random maze. The agent must learn general
navigation strategies, not memorize a single layout. Tests the system's
ability to generalize from experience.
"""

from engram import Runtime, Trainer
from engram.environments.maze import MazeEnv


def main() -> None:
    print("=" * 60)
    print("  ENGRAM -- Maze Navigation Training")
    print("=" * 60)
    print()

    env = MazeEnv(width=5, height=5, seed=42)
    brain = Runtime(input_dims=10, num_actions=4, seed=42)

    print(f"Maze: {env.width}x{env.height} (grid: {env.maze_w}x{env.maze_h})")
    print(f"Agent starts at (1,1), goal at ({env.goal_x},{env.goal_y})")
    print(f"Example maze:")
    print(env.render())
    print()

    trainer = Trainer(brain, env, ticks_per_step=3)
    print("Training for 300 episodes (new maze each episode)...")
    print("-" * 60)
    result = trainer.train(episodes=300, verbose=True, print_every=30)

    print("-" * 60)
    print()
    print(result.summary())


if __name__ == "__main__":
    main()
