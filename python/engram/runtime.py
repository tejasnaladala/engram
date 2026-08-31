"""High-level Python wrapper for the Engram cognitive runtime."""

from __future__ import annotations

import json

from engram._engram_native import PyRuntime


class Runtime:
    """Brain-inspired adaptive intelligence runtime.

    Wraps the Rust-native Engram engine with a Pythonic API.

    Example:
        rt = Runtime(input_dims=8, num_actions=4)
        for episode in range(100):
            obs = env.reset()
            done = False
            while not done:
                action = rt.step(obs)
                obs, reward, done, info = env.step(action)
                rt.reward(reward)
            rt.end_episode()
    """

    def __init__(
        self,
        input_dims: int = 8,
        num_actions: int = 4,
        seed: int = 42,
    ):
        self._rt = PyRuntime(
            input_dims=input_dims,
            num_actions=num_actions,
            seed=seed,
        )

    def step(self, observation: list[float], reward: float | None = None) -> int:
        """Observe, think, and act in one call.

        Args:
            observation: List of float values in [0, 1].
            reward: Optional incremental reward from the previous action. Omit
                this when feedback was already delivered through ``reward()``.

        Returns:
            Selected action ID.
        """
        self._rt.set_observation(observation)
        if reward is not None:
            self._rt.set_reward(reward)
        return self._rt.step()

    def reward(self, value: float) -> None:
        """Deliver one incremental environment reward."""
        self._rt.set_reward(value)

    def evaluation_copy(self) -> Runtime:
        """Return an isolated runtime copy with persistent learning disabled."""
        evaluation_runtime = object.__new__(type(self))
        evaluation_runtime._rt = self._rt.evaluation_copy()
        return evaluation_runtime

    @property
    def learning_enabled(self) -> bool:
        """Whether persistent learning is enabled."""
        return self._rt.learning_enabled()

    @learning_enabled.setter
    def learning_enabled(self, enabled: bool) -> None:
        self._rt.set_learning_enabled(enabled)

    @property
    def learning_state_hash(self) -> str:
        """Deterministic non-cryptographic fingerprint of persistent learning state."""
        return f"{self._rt.learning_state_hash():016x}"

    def end_episode(self) -> None:
        """Signal the end of an episode (preserves learned memories)."""
        self._rt.reset_episode()

    def reset(self) -> None:
        """Full reset including all learned memories."""
        self._rt.full_reset()

    @property
    def prediction_error(self) -> float:
        """Current prediction error magnitude."""
        return self._rt.prediction_error()

    @property
    def tick_count(self) -> int:
        """Current simulation tick."""
        return self._rt.tick()

    @property
    def total_spikes(self) -> int:
        """Lifetime spike count."""
        return self._rt.total_spikes()

    @property
    def total_vetoes(self) -> int:
        """Lifetime safety veto count."""
        return self._rt.total_vetoes()

    @property
    def total_reward(self) -> float:
        """Cumulative reward."""
        return self._rt.total_reward()

    @property
    def sim_time(self) -> float:
        """Simulation time in milliseconds."""
        return self._rt.sim_time()

    def snapshot(self) -> dict:
        """Get the current runtime state as a dictionary."""
        return json.loads(self._rt.snapshot_json())

    def __repr__(self) -> str:
        return (
            f"Runtime(tick={self.tick_count}, spikes={self.total_spikes}, "
            f"vetoes={self.total_vetoes}, error={self.prediction_error:.4f})"
        )
