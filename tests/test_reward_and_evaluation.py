from __future__ import annotations

import copy
import sys
import types
import unittest


class FakeNativeRuntime:
    def __init__(self, input_dims=8, num_actions=4, seed=42):
        self.reward_calls: list[float] = []
        self.observations: list[list[float]] = []
        self.steps = 0

    def set_observation(self, observation):
        self.observations.append(observation)

    def set_reward(self, reward):
        self.reward_calls.append(reward)

    def step(self):
        self.steps += 1
        return 0


native_module = types.ModuleType("engram._engram_native")
native_module.PyRuntime = FakeNativeRuntime
sys.modules["engram._engram_native"] = native_module

from engram.runtime import Runtime
from engram.trainer import Trainer


class TwoStepEnvironment:
    def __init__(self):
        self.step_index = 0

    def reset(self, seed=None):
        self.step_index = 0
        return [0.0]

    def step(self, action):
        self.step_index += 1
        done = self.step_index == 2
        return (
            [float(self.step_index)],
            float(self.step_index),
            done,
            {"reached_goal": done},
        )


class ProbeBrain:
    def __init__(self):
        self.reward_calls: list[float] = []
        self.step_reward_args: list[float | None] = []
        self.step_calls = 0
        self.end_calls = 0
        self.learning_enabled = True
        self.learned_state = 0
        self.last_evaluation_copy = None

    def step(self, obs, reward=None):
        self.step_reward_args.append(reward)
        self.step_calls += 1
        if self.learning_enabled:
            self.learned_state += 1
        return 0

    def reward(self, value):
        self.reward_calls.append(value)
        if self.learning_enabled:
            self.learned_state += 1

    def end_episode(self):
        self.end_calls += 1

    def evaluation_copy(self):
        evaluation_copy = copy.deepcopy(self)
        evaluation_copy.last_evaluation_copy = None
        self.last_evaluation_copy = evaluation_copy
        return evaluation_copy

    @property
    def learning_state_hash(self):
        return f"{self.learned_state:016x}"

    @property
    def prediction_error(self):
        return 0.0

    @property
    def total_vetoes(self):
        return 0

    @property
    def total_spikes(self):
        return self.step_calls


class RewardSemanticsTests(unittest.TestCase):
    def test_omitted_step_reward_does_not_overwrite_pending_feedback(self):
        runtime = Runtime(input_dims=1, num_actions=1)

        runtime.reward(0.75)
        runtime.step([0.5])

        self.assertEqual(runtime._rt.reward_calls, [0.75])

    def test_explicit_step_reward_is_delivered_once(self):
        runtime = Runtime(input_dims=1, num_actions=1)

        runtime.step([0.5], reward=-0.25)

        self.assertEqual(runtime._rt.reward_calls, [-0.25])

    def test_trainer_delivers_each_environment_reward_once(self):
        brain = ProbeBrain()
        trainer = Trainer(brain, TwoStepEnvironment(), ticks_per_step=3)

        trainer.train(episodes=1)

        self.assertEqual(brain.reward_calls, [1.0, 2.0])
        self.assertEqual(brain.step_reward_args, [None] * 6)


class FrozenEvaluationTests(unittest.TestCase):
    def test_evaluate_uses_a_frozen_copy_and_preserves_original_state(self):
        brain = ProbeBrain()
        trainer = Trainer(brain, TwoStepEnvironment(), ticks_per_step=2)
        original_hash = brain.learning_state_hash

        result = trainer.evaluate(episodes=1)

        self.assertEqual(len(result.episodes), 1)
        self.assertEqual(brain.learning_state_hash, original_hash)
        self.assertEqual(brain.step_calls, 0)
        self.assertIsNotNone(brain.last_evaluation_copy)
        self.assertFalse(brain.last_evaluation_copy.learning_enabled)
        self.assertEqual(
            brain.last_evaluation_copy.learning_state_hash,
            original_hash,
        )

    def test_evaluation_reports_only_spikes_from_the_evaluation_run(self):
        brain = ProbeBrain()
        trainer = Trainer(brain, TwoStepEnvironment(), ticks_per_step=2)
        trainer.train(episodes=1)

        result = trainer.evaluate(episodes=1)

        self.assertEqual(brain.total_spikes, 4)
        self.assertEqual(result.total_spikes, 4)


if __name__ == "__main__":
    unittest.main()
