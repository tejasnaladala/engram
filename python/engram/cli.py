"""Engram CLI — command-line interface for the cognitive runtime."""

import click
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

console = Console()


@click.group()
def main():
    """Engram — Brain-inspired adaptive intelligence runtime."""
    pass


@main.command()
@click.option("--episodes", default=50, help="Number of episodes to run")
@click.option("--render/--no-render", default=True, help="Show grid visualization")
@click.option("--seed", default=42, help="Random seed")
def run(episodes: int, render: bool, seed: int):
    """Run an Engram agent in the grid world environment."""
    from engram import Runtime
    from engram.environments import GridWorldEnv

    env = GridWorldEnv(size=12, seed=seed)
    rt = Runtime(input_dims=8, num_actions=4, seed=seed)

    console.print(Panel(
        "[bold cyan]Engram[/] — Brain-Inspired Cognitive Runtime",
        subtitle="Grid World Demo",
    ))

    results = []
    for ep in range(episodes):
        obs = env.reset(seed=seed + ep)
        done = False
        ep_reward = 0.0
        ep_steps = 0

        while not done:
            action = rt.step(obs)
            obs, reward, done, info = env.step(action)
            rt.reward(reward)
            ep_reward += reward
            ep_steps += 1

            if render and ep_steps % 10 == 0:
                console.clear()
                console.print(f"Episode {ep+1}/{episodes} | Step {ep_steps}")
                console.print(env.render())
                console.print(f"Reward: {ep_reward:.2f} | Error: {rt.prediction_error:.4f}")

        rt.end_episode()
        results.append((ep + 1, ep_steps, ep_reward))
        console.print(
            f"  Ep {ep+1:3d}: steps={ep_steps:4d}, "
            f"reward={ep_reward:7.2f}, "
            f"error={rt.prediction_error:.4f}, "
            f"spikes={rt.total_spikes}"
        )

    # Summary table
    table = Table(title="Results Summary")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Total Episodes", str(episodes))
    table.add_row("Total Spikes", f"{rt.total_spikes:,}")
    table.add_row("Total Vetoes", str(rt.total_vetoes))
    table.add_row("Final Pred. Error", f"{rt.prediction_error:.4f}")
    avg_reward = sum(r[2] for r in results) / len(results)
    table.add_row("Avg Episode Reward", f"{avg_reward:.2f}")
    console.print(table)


@main.command()
@click.option("--port", default=9000, help="WebSocket port")
def dashboard(port: int):
    """Start the Engram server and open the dashboard."""
    import subprocess
    import sys

    console.print(f"[cyan]Starting Engram server on port {port}...[/]")
    console.print(f"[green]Connect dashboard to ws://localhost:{port}/ws[/]")

    # Start the Rust server binary
    try:
        subprocess.run(
            ["cargo", "run", "-p", "engram-server", "--release"],
            cwd=r"C:\Users\tejas\engram",
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped.[/]")


@main.command()
def benchmark():
    """Run the Engram benchmark suite."""
    from engram import Runtime
    from engram.environments import GridWorldEnv

    console.print(Panel(
        "[bold cyan]Engram Benchmark Suite[/]",
        subtitle="Continual Learning | Adaptation | Efficiency",
    ))

    # Benchmark 1: Continual Learning
    console.print("\n[bold]1. Continual Learning (Catastrophic Forgetting Test)[/]")
    rt = Runtime(input_dims=8, num_actions=4, seed=42)
    env_a = GridWorldEnv(size=10, seed=100)
    env_b = GridWorldEnv(size=10, seed=200)

    # Phase 1: Train on Task A
    task_a_rewards = []
    for ep in range(30):
        obs = env_a.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action = rt.step(obs)
            obs, reward, done, _ = env_a.step(action)
            rt.reward(reward)
            ep_reward += reward
        rt.end_episode()
        task_a_rewards.append(ep_reward)
    task_a_final = sum(task_a_rewards[-5:]) / 5

    # Phase 2: Train on Task B
    for ep in range(30):
        obs = env_b.reset()
        done = False
        while not done:
            action = rt.step(obs)
            obs, reward, done, _ = env_b.step(action)
            rt.reward(reward)
        rt.end_episode()

    # Phase 3: Re-test Task A
    task_a_retest = []
    for ep in range(10):
        obs = env_a.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action = rt.step(obs)
            obs, reward, done, _ = env_a.step(action)
            ep_reward += reward
        task_a_retest.append(ep_reward)
    task_a_after = sum(task_a_retest) / len(task_a_retest)

    forgetting = max(0, (task_a_final - task_a_after) / abs(task_a_final) * 100) if task_a_final != 0 else 0
    console.print(f"  Task A performance (before B): {task_a_final:.2f}")
    console.print(f"  Task A performance (after B):  {task_a_after:.2f}")
    console.print(f"  Forgetting: {forgetting:.1f}%")

    # Benchmark 2: Adaptation Speed
    console.print("\n[bold]2. Adaptation Speed[/]")
    console.print(f"  Total spikes: {rt.total_spikes:,}")
    console.print(f"  Total ticks: {rt.tick_count:,}")

    # Benchmark 3: Energy Efficiency
    console.print("\n[bold]3. Energy Efficiency[/]")
    spikes_per_tick = rt.total_spikes / max(1, rt.tick_count)
    console.print(f"  Spikes per tick: {spikes_per_tick:.1f}")
    console.print(f"  Total vetoes: {rt.total_vetoes}")

    console.print("\n[green]Benchmark complete![/]")


if __name__ == "__main__":
    main()
