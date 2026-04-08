"""
Engram -- Brain-inspired adaptive intelligence runtime.

A modular cognitive architecture with real spiking neurons, STDP learning,
predictive coding, episodic memory, and a safety kernel.

Quick Start:
    from engram import Runtime
    rt = Runtime()
    action = rt.step([0.5] * 8)
"""

from engram.runtime import Runtime

__version__ = "0.1.0"
__all__ = ["Runtime"]
