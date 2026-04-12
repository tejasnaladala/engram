"""Solve a maze using Engram's dual-phase spiking DQN.

This demonstrates the novel dual-phase training approach:
Phase 1: Surrogate gradient pretraining on the maze
Phase 2: Online adaptation (local plasticity only) on a modified maze

Usage:
    # Solve a procedural maze:
    python examples/solve_maze.py

    # Solve a maze from an image:
    python examples/solve_maze.py --image path/to/maze.png

    # Save training animation frames:
    python examples/solve_maze.py --save-frames
"""

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engram.spiking_dqn import SpikingDQNTrainer
from engram.environments.maze import MazeEnv


def print_maze_with_path(env, episode, reward, success):
    """Print maze state with color if terminal supports it."""
    grid = env.render()
    status = "SOLVED!" if success else f"reward: {reward:.1f}"
    print(f"\r  Episode {episode:4d} | {status}")
    # Only print grid occasionally to avoid flooding
    if success or episode % 50 == 0:
        print(grid)
        print()


def main():
    parser = argparse.ArgumentParser(description='Engram Maze Solver')
    parser.add_argument('--image', type=str, help='Path to maze image (PNG/JPG)')
    parser.add_argument('--size', type=int, default=5, help='Procedural maze size (default: 5)')
    parser.add_argument('--episodes', type=int, default=300, help='Training episodes')
    parser.add_argument('--save-frames', action='store_true', help='Save frames for animation')
    parser.add_argument('--save-model', type=str, default=None, help='Save trained model path')
    args = parser.parse_args()

    print("=" * 60)
    print("  ENGRAM -- Spiking Neural Maze Solver")
    print("  Dual-Phase Training: Surrogate Gradients + Local Adaptation")
    print("=" * 60)
    print()

    # Create environment
    if args.image:
        try:
            from engram.environments.image_maze import ImageMazeEnv
            env = ImageMazeEnv(args.image, max_size=30)
            print(f"  Loaded maze from image: {args.image}")
            print(f"  Grid size: {env.w}x{env.h}")
        except ImportError:
            print("  Error: Pillow required for image mazes. Install with: pip install Pillow")
            return
    else:
        env = MazeEnv(width=args.size, height=args.size, seed=42)
        print(f"  Procedural maze: {args.size}x{args.size}")

    print(f"  Start: ({env.agent_x}, {env.agent_y})")
    if hasattr(env, 'goal_x'):
        print(f"  Goal:  ({env.goal_x}, {env.goal_y})")
    print()

    obs = env.reset()
    print("  Initial maze:")
    print(env.render())
    print()

    # Create spiking DQN trainer
    trainer = SpikingDQNTrainer(
        obs_dim=10,
        num_actions=4,
        hidden=64,
        num_steps=8,
        lr=5e-4,
        gamma=0.99,
        epsilon_decay=args.episodes * 50,
        target_update=300,
        buffer_size=20000,
    )

    # ========================================
    # PHASE 1: Surrogate Gradient Training
    # ========================================
    print("=" * 60)
    print("  PHASE 1: Surrogate Gradient Training")
    print("  Training spiking neurons with backpropagation through")
    print("  surrogate gradient approximation of spike function")
    print("=" * 60)
    print()

    start_time = time.time()
    phase1_result = trainer.train_phase1(
        env, episodes=args.episodes, verbose=True, print_every=30
    )
    phase1_time = time.time() - start_time

    print()
    print(f"  Phase 1 complete in {phase1_time:.1f}s")
    print(f"  {phase1_result.summary()}")
    print()

    # Show solved maze
    if phase1_result.success_rate_last_20 > 0:
        print("  Final solved path:")
        obs = env.reset()
        done = False
        while not done:
            with __import__('torch').no_grad():
                import torch
                q = trainer.policy_net(torch.FloatTensor(obs).unsqueeze(0))
                action = q.argmax(dim=1).item()
            obs, reward, done, info = env.step(action)
        print(env.render())
        print()

    # ========================================
    # PHASE 2: Online Adaptation
    # ========================================
    if hasattr(env, 'rng') or isinstance(env, MazeEnv):
        print("=" * 60)
        print("  PHASE 2: Online Local Adaptation")
        print("  New maze layout. Only output layer adapts (no backprop).")
        print("  Tests continual learning without catastrophic forgetting.")
        print("=" * 60)
        print()

        # Create a different maze
        if isinstance(env, MazeEnv):
            env2 = MazeEnv(width=args.size, height=args.size, seed=99)
        else:
            env2 = env  # use same env for image mazes

        print("  New maze layout:")
        env2.reset()
        print(env2.render())
        print()

        start_time = time.time()
        phase2_result = trainer.adapt_phase2(
            env2, episodes=50, verbose=True, print_every=10
        )
        phase2_time = time.time() - start_time

        print()
        print(f"  Phase 2 complete in {phase2_time:.1f}s")
        print(f"  {phase2_result.summary()}")
        print()

        # Test recall of original maze (catastrophic forgetting check)
        print("  -- Catastrophic Forgetting Check --")
        print("  Testing on ORIGINAL maze (no further training):")
        recall_successes = 0
        for trial in range(20):
            obs = env.reset()
            done = False
            while not done:
                with __import__('torch').no_grad():
                    import torch
                    q = trainer.policy_net(torch.FloatTensor(obs).unsqueeze(0))
                    action = q.argmax(dim=1).item()
                obs, _, done, info = env.step(action)
            if info.get('reached_goal', False):
                recall_successes += 1
        print(f"  Original maze recall: {recall_successes}/20 ({recall_successes*5}%)")

    if args.save_model:
        trainer.save(args.save_model)
        print(f"\n  Model saved to: {args.save_model}")

    print()
    print("=" * 60)
    print("  COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
