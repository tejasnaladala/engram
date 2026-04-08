"""Engram Grid World Demo — Watch a brain-inspired agent learn to navigate."""

from engram import Runtime
from engram.environments import GridWorldEnv

def main():
    env = GridWorldEnv(size=12, num_walls=15, num_hazards=5, seed=42)
    rt = Runtime(input_dims=8, num_actions=4, seed=42)

    print("=" * 60)
    print("  ENGRAM — Brain-Inspired Cognitive Runtime")
    print("  Grid World Demo")
    print("=" * 60)

    num_episodes = 20
    results = []

    for ep in range(num_episodes):
        obs = env.reset(seed=42 + ep)
        done = False
        ep_reward = 0.0
        ep_steps = 0

        while not done:
            action = rt.step(obs, reward=0.0 if ep_steps == 0 else reward)
            obs, reward, done, info = env.step(action)
            rt.reward(reward)
            ep_reward += reward
            ep_steps += 1

        rt.end_episode()
        results.append((ep_steps, ep_reward))
        reached = "TARGET!" if reward >= 9.0 else "timeout"

        print(
            f"  Episode {ep+1:3d} | steps={ep_steps:4d} | "
            f"reward={ep_reward:7.2f} | error={rt.prediction_error:.3f} | {reached}"
        )

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  Total spikes:     {rt.total_spikes:>10,}")
    print(f"  Total vetoes:     {rt.total_vetoes:>10}")
    print(f"  Sim time:         {rt.sim_time:>10.0f}ms")
    print(f"  Pred. error:      {rt.prediction_error:>10.4f}")
    avg_steps = sum(r[0] for r in results) / len(results)
    avg_reward = sum(r[1] for r in results) / len(results)
    print(f"  Avg steps/ep:     {avg_steps:>10.1f}")
    print(f"  Avg reward/ep:    {avg_reward:>10.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
