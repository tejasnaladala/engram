<div align="center">

<br>

```
                              ███████╗███╗   ██╗ ██████╗ ██████╗  █████╗ ███╗   ███╗
                              ██╔════╝████╗  ██║██╔════╝ ██╔══██╗██╔══██╗████╗ ████║
                              █████╗  ██╔██╗ ██║██║  ███╗██████╔╝███████║██╔████╔██║
                              ██╔══╝  ██║╚██╗██║██║   ██║██╔══██╗██╔══██║██║╚██╔╝██║
                              ███████╗██║ ╚████║╚██████╔╝██║  ██║██║  ██║██║ ╚═╝ ██║
                              ╚══════╝╚═╝  ╚═══╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝
```

**An open-source framework for building AI systems that learn continuously from experience.**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.94%2B-orange.svg)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![Tests](https://img.shields.io/badge/tests-17%2F17-brightgreen.svg)]()

[Docs](docs/) &middot; [Examples](examples/) &middot; [Observatory](#the-observatory) &middot; [Contributing](CONTRIBUTING.md)

<br>

</div>

---

## What is Engram?

Engram is a framework for building adaptive systems that **learn from experience in real-time** &mdash; without retraining, without fine-tuning, without a training loop.

You define a brain as a composition of learning regions. Each region has neurons, connections, and learning rules. Data flows through the system as events. Learning happens locally and continuously. Memory persists across sessions. A safety layer gates dangerous actions.

```python
import engram as eg

brain = eg.Brain.default(input_dims=8, num_actions=4)

env = eg.envs.GridWorld(size=12)
obs = env.reset()

for step in range(1000):
    action = brain.step(obs)
    obs, reward, done, _ = env.step(action)
    brain.reward(reward)
    if done:
        obs = env.reset()
```

**No `loss.backward()`. No `optimizer.step()`. No epochs.**
The brain learns online through local learning rules and neuromodulatory signals.

---

## Why does this exist?

Every major ML framework assumes training and inference are separate phases. But a growing class of systems need **continuous adaptation**:

| Problem | Why standard ML fails | How Engram helps |
|---------|----------------------|-----------------|
| Robots in new environments | Retraining takes hours + cloud | Online adaptation in real-time |
| Agents with persistent memory | LLMs are stateless per-session | Associative + episodic memory |
| Edge AI personalization | Fine-tuning needs GPUs | Local learning rules, no backprop |
| Safety-critical autonomy | No built-in action gating | Safety kernel vetoes dangerous actions |
| Anomaly detection on streams | Batch models miss new patterns | Event-driven, always learning |

---

## Core Architecture

```
         ┌─────────────────────────────────────┐
         │         EXPERIENCE STREAM            │
         └──────────────┬──────────────────────┘
                        │
              ┌─────────▼─────────┐
              │  SENSORY ENCODER   │  observation → spikes
              └────┬──────────┬───┘
                   │          │
         ┌─────────▼──┐  ┌───▼──────────┐
         │ ASSOCIATIVE │  │  PREDICTIVE   │
         │   MEMORY    │  │    LAYER      │
         └──────┬──────┘  └──────┬────────┘
                │                │
         ┌──────▼──────┐  ┌─────▼────────┐
         │   ACTION    │  │  PREDICTION   │
         │  SELECTOR   │  │    ERROR      │
         └──────┬──────┘  └──────────────┘
                │
         ┌──────▼──────┐
         │   SAFETY    │  veto if unsafe
         │    GATE     │
         └──────┬──────┘
                │
         ┌──────▼──────┐
         │   ACTION    │
         └─────────────┘
```

Each box is a **Region** &mdash; a composable module with neurons and local learning dynamics. Regions connect via **Pathways** that learn through local rules.

---

## Core Abstractions

| Abstraction | Purpose | PyTorch Analogy |
|-------------|---------|-----------------|
| `Brain` | Top-level container. Wires regions, runs experience loop. | Training loop |
| `Region` | Population of neurons with internal connections. | `nn.Module` |
| `Pathway` | Weighted, plastic connection between regions. | `nn.Linear` (but learns locally) |
| `LearningRule` | How connections change. Runs locally per pathway. | `autograd` (but local) |
| `Modulator` | Global signals that gate learning. | `optimizer` |
| `SafetyGate` | Constraint layer that vetoes actions. | *(no equivalent)* |

---

## Learning Rules

| Rule | Mechanism | Use case |
|------|-----------|----------|
| `ThreeFactorSTDP` | STDP + eligibility traces + neuromodulation | **Default.** Handles delayed reward via credit assignment. |
| `STDP` | Spike-timing-dependent plasticity | Unsupervised pattern learning |
| `RewardModulatedSTDP` | STDP gated by reward signal | Fast reward-driven adaptation |
| `Hebbian` | Co-activation strengthening | Simple association |
| *Custom* | Implement the `LearningRule` trait | Research / novel algorithms |

The neuromodulatory system broadcasts four signals:
- **Reward** (dopamine-like): reward prediction error drives learning direction
- **Surprise** (acetylcholine-like): prediction error amplifies learning rate
- **Arousal** (norepinephrine-like): scales overall learning magnitude
- **Inhibition** (serotonin-like): dampens exploration when things go well

---

## The Observatory

A real-time visualization platform for understanding what adaptive systems learn.

```bash
engram dashboard --port 3000
```

- 3D brain topology with signal pulses flowing between regions
- Multi-channel spike raster (672 neurons, color-coded by region)
- Prediction error waveform with live numeric readout
- Memory formation heatmap (fMRI-style hot colormap)
- Safety veto event log with expandable details
- Module activity readouts with neuron counts
- Interactive: hover tooltips, click-to-inspect, ablation toggles

---

## Technical Details

| Component | Implementation |
|-----------|---------------|
| Core runtime | Rust (zero-cost abstractions, WASM target, no GC) |
| Neurons | Leaky Integrate-and-Fire, 672 across 6 regions |
| Synapses | Compressed Sparse Row format, ~45K connections |
| Learning | Three-factor STDP with eligibility traces (tau=1000ms) |
| Memory | Sparse Distributed Memory (content-addressable, forgetting-resistant) |
| Safety | Hard constraints + learned inhibitions |
| Dashboard | React + Three.js + Recharts, WebSocket @ 30fps |
| Browser demo | WebAssembly (87KB compiled) |
| Python API | PyO3 bindings via maturin |
| Tests | 17/17 passing |

---

## Roadmap

- [x] Core runtime with LIF neurons and sparse synapses
- [x] 6 brain region modules
- [x] Three-factor STDP with eligibility traces
- [x] Neuromodulatory system (reward, surprise, arousal, inhibition)
- [x] Observatory dashboard
- [x] WASM browser demo
- [x] Python bindings
- [ ] YAML brain configuration
- [ ] Module SDK (custom regions in Python)
- [ ] Learning Rule SDK
- [ ] Benchmark suite with DQN/PPO baselines
- [ ] Sleep/consolidation mode
- [ ] Curiosity-driven exploration
- [ ] Multi-agent simulation
- [ ] ROS 2 integration
- [ ] NIR export for neuromorphic hardware

---

## What This Is NOT

- **Not AGI.** A framework for adaptive systems, not artificial general intelligence.
- **Not a PyTorch replacement.** Complements PyTorch for continuous online learning.
- **Not biologically accurate.** Brain-*inspired*, not brain-simulating.
- **Not magic.** Requires informative reward signals to learn effectively.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Good first issues:

- Add a learning rule (BCM, Oja, anti-Hebbian)
- Add an environment (CartPole, sensor stream, maze)
- Add a dashboard widget
- Improve docs or examples

---

## Inspiration

Computational neuroscience principles adapted for engineering:

- Spiking neural networks (Maass 1997)
- Spike-timing-dependent plasticity (Bi & Poo 1998)
- Three-factor learning with eligibility traces (Gerstner et al.)
- Sparse Distributed Memory (Kanerva 1988)
- Predictive coding (Rao & Ballard 1999)
- Sleep replay consolidation (Tadros et al. 2022)

---

## License

Apache 2.0. See [LICENSE](LICENSE).

<div align="center">
<br>

**[Get Started](#what-is-engram)** &middot; **[Examples](examples/)** &middot; **[Observatory](#the-observatory)** &middot; **[Contribute](CONTRIBUTING.md)**

</div>
