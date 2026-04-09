"""Train an Engram agent on the Grid World environment.

Usage:
    python examples/train_grid_world.py

The agent learns to navigate a 8x8 grid world with walls and hazards,
reaching the goal while avoiding obstacles. Learning happens online
via three-factor STDP with eligibility traces and neuromodulation.
"""

from engram import Runtime, Trainer
from engram.environments.grid_world import GridWorldEnv


def main() -> None:
    print("=" * 60)
    print("  ENGRAM -- Grid World Navigation Training")
    print("=" * 60)
    print()

    # Create environment and brain
    env = GridWorldEnv(size=8, num_walls=8, num_hazards=3, seed=42)
    brain = Runtime(input_dims=8, num_actions=4, seed=42)

    print(f"Environment: {env.size}x{env.size} grid, {env.num_walls} walls, {env.num_hazards} hazards")
    print(f"Brain: 672 spiking neurons, 6 regions, three-factor STDP")
    print(f"Initial grid:")
    print(env.render())
    print()

    # Train
    trainer = Trainer(brain, env, ticks_per_step=3)
    print("Training for 200 episodes...")
    print("-" * 60)
    result = trainer.train(episodes=200, verbose=True, print_every=20)

    print("-" * 60)
    print()
    print(result.summary())
    print()

    # Show reward curve trend
    curve = result.reward_curve(window=20)
    if len(curve) >= 20:
        early = sum(curve[:20]) / 20
        late = sum(curve[-20:]) / 20
        improvement = late - early
        print(f"Reward trend: early={early:.2f} -> late={late:.2f} (improvement: {improvement:+.2f})")


if __name__ == "__main__":
    main()
