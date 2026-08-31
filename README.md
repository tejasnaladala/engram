# Engram

An open-source framework for building systems that learn continuously from experience.

Engram is a cognitive runtime, not a model. You wire together brain regions with plastic
pathways, feed it a stream of observations and rewards, and it adapts online while it runs.
The native runtime has no optimizer or backward pass. Each connection updates itself from
local activity and a global neuromodulatory signal, the way biological synapses are thought
to. The repository also contains a separate experimental Spiking DQN that does use PyTorch
surrogate-gradient backpropagation and replay; claims about local learning do not apply to it.

The core is Rust. The API you actually touch is Python (via PyO3). There is a React
"Observatory" dashboard that streams the network's internal state at 30fps, and a WebAssembly
build so the whole thing can run in a browser tab.

```python
from engram import Runtime, Trainer
from engram.environments import GridWorldEnv

env = GridWorldEnv(size=8)
brain = Runtime(input_dims=8, num_actions=4)      # 672 spiking neurons across 6 regions
result = Trainer(brain, env).train(episodes=200)
print(result.summary())
```

## Why the native runtime is not another backprop framework

A standard deep-learning stack collects a batch, computes a loss, and pushes gradients
backward through the whole graph. The weights are frozen at deployment. Adapting to a new
distribution means stopping, retraining, and redeploying.

The native Engram runtime learns the way a nervous system does:

- **No backward pass.** Every pathway carries its own `LearningRule`. A synapse changes
  because the neurons on either side of it fired in a particular order, scaled by a global
  reward/surprise signal. Credit assignment is local in space and resolved over time through
  eligibility traces, not by differentiating a loss.
- **Online by default, frozen when requested.** `Runtime.step()` observes, thinks, acts, and
  updates local state while learning is enabled. `Trainer.evaluate()` instead runs an isolated
  copy with persistent updates disabled and verifies that its learning-state hash is unchanged.
- **Memory is a first-class structure, not a context window.** Associative memory (a
  sparse-distributed store) and episodic memory (a replay buffer that consolidates during
  quiet periods) persist across episodes by design. The runtime deliberately does *not* reset
  them at episode boundaries.
- **Safety is inside the loop.** A `SafetyKernel` runs alongside action selection and can veto
  a proposed action before it reaches the environment, using both hard constraints and
  inhibitions learned from past negative outcomes.

The tradeoff is honest: this is sample-hungry online learning on small networks, not a way to
top a benchmark leaderboard. It is built for systems that have to keep adapting after they
ship, where you care more about behavior over time than about a single converged number.

## Core abstractions

The framework is built from a small set of pieces. Conceptually:

| Concept | What it is | Where it lives |
|---|---|---|
| **State cell** | A leaky integrate-and-fire neuron with membrane potential and a refractory period | `LIFNeuron`, `NeuronPopulation` (`crates/engram-core/src/neuron.rs`) |
| **Connection** | A sparse (CSR) weight matrix between two populations | `SynapseMatrix` (`crates/engram-core/src/synapse.rs`) |
| **Learning rule** | The plasticity that updates a connection from local spikes + modulation | `LearningRule` trait, `ThreeFactorSTDP`, `HebbianRule` (`crates/engram-core/src/learning_rule.rs`) |
| **Region** | A brain module that turns incoming spikes into outgoing spikes | `BrainModule` trait (`crates/engram-core/src/module_trait.rs`); six concrete regions in `engram-modules` |
| **Pathway** | A connection plus the learning rule that trains it, wiring one region to another | inter-module synapse + rule pairs in `crates/engram-runtime/src/runtime.rs` |
| **Brain** | The runtime that orchestrates regions and pathways through the cognitive loop | `EngramRuntime` (`engram-runtime`), exposed to Python as `Runtime` |
| **Modulator** | The global reward/surprise/arousal/inhibition signal, the "third factor" | `Neuromodulators` (`crates/engram-core/src/learning_rule.rs`) |
| **Experience stream** | The sequence of observations and rewards driving the brain | environments in `python/engram/environments/` |
| **Observer** | A snapshot of the brain's internal state for visualization | `RuntimeSnapshot` (`crates/engram-core/src/types.rs`), streamed by `engram-server` |
| **Safety envelope** | The gate that vetoes dangerous actions | `SafetyKernel` (`crates/engram-modules/src/safety_kernel.rs`) |

In the current release the high-level Python API is the `Runtime` constructor, which assembles
the six default regions and four learning pathways for you:

```python
brain = Runtime(input_dims=8, num_actions=4, seed=42)
```

The composable `Brain(regions=[...], pathways=[...])` builder that exposes those regions and
pathways directly is on the roadmap (see [CONTRIBUTING.md](CONTRIBUTING.md)); today the wiring
is fixed in `engram-runtime` and configured through `RuntimeConfig`.

## How learning works

The primary rule is three-factor STDP with eligibility traces:

```
dw_ij = eta * e_ij * M(t)

  e_ij  eligibility trace: accumulates spike-timing correlations, decays with tau_e
  M(t)  modulatory signal: reward prediction error, scaled by surprise / arousal / inhibition
```

A pre-then-post spike pair leaves a positive trace; post-then-pre leaves a negative one. The
trace lingers (tau_e is ~1 second by default) so a reward arriving later can still strengthen
the connections that led to it. When `M(t)` is near zero, nothing moves; when reward beats the
running baseline, eligible connections strengthen. That is the whole credit-assignment story,
and it runs per-pathway with no global graph.

Four neuromodulators shape `M(t)`, loosely after their biological analogs:

| Signal | Analog | Effect on learning |
|---|---|---|
| Reward | Dopamine | Sets the direction and sign of weight change |
| Surprise | Acetylcholine | Amplifies the rate when input was unexpected |
| Arousal | Norepinephrine | Scales overall magnitude |
| Inhibition | Serotonin | Damps exploration when things are going well |

`HebbianRule` is also provided for unmodulated "fire together, wire together" plasticity, and
new rules are the most common contribution: implement the `LearningRule` trait and drop it onto
a pathway.

## Quickstart

### Python

```bash
git clone https://github.com/tejasnaladala/engram
cd engram
pip install maturin
maturin develop          # builds the Rust core and installs the `engram` package
```

```python
from engram import Runtime, Trainer
from engram.environments import GridWorldEnv

env = GridWorldEnv(size=8, num_walls=8, num_hazards=3, seed=42)
brain = Runtime(input_dims=8, num_actions=4, seed=42)

result = Trainer(brain, env, ticks_per_step=3).train(episodes=200, verbose=True)
print(result.summary())
```

The low-level loop, if you want to drive the brain yourself:

```python
brain = Runtime(input_dims=8, num_actions=4)
obs = env.reset()
done = False
while not done:
    action = brain.step(obs)            # encode -> think -> select -> safety-gate
    obs, reward, done, info = env.step(action)
    brain.reward(reward)                # one incremental feedback event
brain.end_episode()                     # flushes terminal feedback; keeps learned memory
```

There is also a CLI installed with the package:

```bash
engram run --episodes 50          # run the grid-world demo with a live render
engram benchmark                  # continual-learning / adaptation / efficiency probes
engram dashboard                  # start the WebSocket server for the Observatory
```

### Rust

```bash
cargo check --workspace           # type-check the whole workspace (no GPU, no Python)
cargo run -p engram-server --release   # start the dashboard server on ws://localhost:9000/ws
```

### Dashboard

```bash
cd dashboard && npm install && npm run dev
```

The Observatory is a React + react-three-fiber app. It connects to `engram-server` over a
WebSocket, decodes MessagePack snapshots, and renders region activity, spike traffic,
prediction error, memory formation, and safety vetoes in real time.

## Architecture

```
engram-core       Pure computation. LIF neurons, CSR synapses, learning rules, spike buffers.
                  No platform dependencies. Compiles to native and to WebAssembly.
engram-modules    The six brain regions, each implementing the BrainModule trait:
                  SensoryEncoder, AssociativeMemory, PredictiveError, EpisodicMemory,
                  ActionSelector, SafetyKernel.
engram-runtime    The cognitive loop. Wires regions together with learning pathways,
                  drives the modulators, and produces snapshots.
engram-server     Axum WebSocket server. Streams snapshots to the Observatory at 30fps.
engram-python     PyO3 bindings. Exposes the runtime as the `engram` Python package.
engram-wasm       WebAssembly target for the in-browser demo.
```

Each tick runs a fixed sequence: sensory encoding, routing to associative and predictive
layers, a memory step, prediction-error computation, modulator update, action selection,
safety evaluation, episodic recording/replay, and finally the learning-rule updates across all
pathways. The full step-by-step is in [ARCHITECTURE.md](ARCHITECTURE.md).

Design choices worth calling out:

- **Rust core, Python skin.** Spike processing is hot; the ergonomics live in Python.
- **Sparse by default.** Synapses are CSR, and the loop processes spikes as events rather than
  dense tensors, which pays off at the high sparsity these networks run at.
- **Memory persists across episodes.** `reset_episode()` clears transient neuron state but
  leaves associative memory and learned safety inhibitions intact. A full wipe is a separate
  `reset()`.

The default `Runtime(input_dims=8, num_actions=4)` builds **672 spiking neurons** across the
six regions (128 sensory, 256 associative, 64 predictive, 64 episodic, 128 action, 32 safety)
connected by four learning pathways.

## Benchmarks

The repository ships a runnable benchmark suite rather than a table of numbers to take on
faith. Run it yourself:

```bash
python benchmarks/benchmark.py
```

It pits the Engram runtime against tabular Q-learning and a random baseline on three tasks that
target what online local learning is supposed to be good at:

1. **Grid-world navigation**: reach the goal, avoid walls and hazards.
2. **Continual learning**: train on layout A, then layout B, then re-test A with learning
   disabled and a learning-state hash check, to measure catastrophic forgetting.
3. **Online pattern classification**: classify noisy streamed patterns from the reward signal
   alone, with no labeled training phase.

`benchmarks/proof.py` runs a separate seeded comparison of a surrogate-gradient spiking DQN
against Q-learning and random on small mazes. Its second phase fine-tunes the output layer with
SGD and backpropagation on an unseen layout; it is not an update-free recall test or a local
plasticity result. Both scripts print exact numbers under fixed seeds so results are
reproducible on your own hardware; they are deliberately not pre-baked into this README.

## Status

This is `0.1.0`, research-grade and early. The native Rust workspace
(`engram-core`, `engram-modules`, `engram-runtime`, `engram-server`, `engram-python`)
type-checks clean with `cargo check --workspace`. The core has unit tests for neuron dynamics,
STDP eligibility accumulation, and reward-driven weight change. The composable `Brain` builder,
YAML brain configs, and hardware backends (Loihi, Akida) are roadmap items, not shipped.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). New learning rules and new brain regions are the most
useful contributions; both are single-trait implementations.

## License

Apache-2.0. See [LICENSE](LICENSE).

## Contact

Tejas Naladala, `tejas.naladala@gmail.com`. Independent reproduction is welcome.
