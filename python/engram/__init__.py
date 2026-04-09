"""
Engram -- Framework for building AI systems that learn continuously from experience.

Quick start:
    from engram import Runtime, Trainer
    from engram.environments.grid_world import GridWorldEnv

    env = GridWorldEnv(size=8)
    brain = Runtime(input_dims=8, num_actions=4)
    trainer = Trainer(brain, env)
    result = trainer.train(episodes=100, verbose=True)
    print(result.summary())
"""

from engram.runtime import Runtime
from engram.trainer import Trainer, RandomBaseline, TrainingResult, EpisodeResult

__version__ = "0.1.0"
__all__ = ["Runtime", "Trainer", "RandomBaseline", "TrainingResult", "EpisodeResult"]
