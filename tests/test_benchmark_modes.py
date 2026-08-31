from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path


class FakeNativeRuntime:
    def __init__(self, input_dims=8, num_actions=4, seed=42):
        self.reward_calls = []
        self.observations = []

    def set_observation(self, observation):
        self.observations.append(observation)

    def set_reward(self, reward):
        self.reward_calls.append(reward)

    def step(self):
        return 0


native_module = types.ModuleType("engram._engram_native")
native_module.PyRuntime = FakeNativeRuntime
sys.modules["engram._engram_native"] = native_module

BENCHMARK_PATH = Path(__file__).parents[1] / "benchmarks" / "benchmark.py"
SPEC = importlib.util.spec_from_file_location("engram_benchmark", BENCHMARK_PATH)
benchmark = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


class OneStepEnvironment:
    def reset(self):
        return [0.0]

    def step(self, action):
        return [1.0], 1.0, True, {"reached_goal": True}


class ProbeAgent:
    def __init__(self):
        self.learn_calls = 0
        self.learning_modes: list[bool] = []
        self.events: list[str] = []

    def set_learning_enabled(self, enabled):
        self.learning_modes.append(enabled)

    def reset(self):
        self.events.append("reset")

    def act(self, obs):
        return 0

    def learn(self, obs, action, reward, next_obs, done):
        self.learn_calls += 1
        self.events.append("learn")


class BenchmarkModeTests(unittest.TestCase):
    def test_frozen_benchmark_does_not_relearn(self):
        agent = ProbeAgent()

        benchmark.run_benchmark(
            agent,
            OneStepEnvironment(),
            episodes=2,
            name="probe",
            learning=False,
        )

        self.assertEqual(agent.learning_modes, [False])
        self.assertEqual(agent.learn_calls, 0)
        self.assertEqual(agent.events, ["reset", "reset"])

    def test_training_finalizes_each_episode_after_its_last_reward(self):
        agent = ProbeAgent()

        benchmark.run_benchmark(
            agent,
            OneStepEnvironment(),
            episodes=2,
            name="probe",
            learning=True,
        )

        self.assertEqual(agent.events, ["learn", "reset", "learn", "reset"])


if __name__ == "__main__":
    unittest.main()
