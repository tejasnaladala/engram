# engram

> Brain-inspired adaptive intelligence runtime with real spiking neurons, online STDP learning, and modular cognitive architecture.

**Engram** is an open-source cognitive runtime that makes brain-inspired AI buildable by any developer. It features real Leaky Integrate-and-Fire neurons, Spike-Timing-Dependent Plasticity (STDP) learning, predictive coding, episodic memory with replay, and a safety kernel — all orchestrated through a 10-step cognitive loop.

## Quick Start

```python
from engram import Runtime
from engram.environments import GridWorldEnv

env = GridWorldEnv(size=12)
rt = Runtime(input_dims=8, num_actions=4)

obs = env.reset()
for step in range(1000):
    action = rt.step(obs)
    obs, reward, done, info = env.step(action)
    rt.reward(reward)
    if done:
        rt.end_episode()
        obs = env.reset()

print(f"Total spikes: {rt.total_spikes:,}")
print(f"Prediction error: {rt.prediction_error:.4f}")
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Event Bus (Typed Message Router)            │
├────────┬─────────┬──────────┬─────────┬─────────────────┤
│Sensory │Predict. │Assoc.    │Action   │Safety Kernel    │
│Encoder │Error    │Memory    │Selector │(rule-based +    │
│(SNN /  │Module   │(sparse   │(policy  │ learned gates)  │
│ pop.)  │(pred.   │ vector + │ network │                 │
│        │ coding) │ episodic)│ + reflex│                 │
├────────┴─────────┴──────────┴─────────┴─────────────────┤
│              Experience Store / Replay Engine            │
├─────────────────────────────────────────────────────────┤
│           Observability Dashboard (WebSocket)            │
└─────────────────────────────────────────────────────────┘
```

## Features

- **Real spiking neurons** — Leaky Integrate-and-Fire (LIF) model with configurable parameters
- **STDP learning** — Online, trace-based Spike-Timing-Dependent Plasticity
- **Modular brain regions** — Sensory encoder, associative memory, predictive error, episodic memory, action selector, safety kernel
- **Continuous learning** — Learns from experience without retraining; resistant to catastrophic forgetting via Sparse Distributed Memory
- **Safety kernel** — Hard constraints + learned inhibition can veto dangerous actions
- **Real-time dashboard** — WebSocket-connected visualization of spike rasters, memory formation, prediction error, and module activity
- **Browser demo** — WASM-compiled version runs entirely in the browser

## Installation

```bash
pip install engram
```

## CLI

```bash
engram run --episodes 50          # Run agent in grid world
engram dashboard --port 9000      # Start server + dashboard
engram benchmark                  # Run benchmark suite
```

## License

Apache-2.0
