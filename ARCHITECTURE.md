# Engram Architecture

## System Overview

Engram is a cognitive runtime built in Rust with Python bindings and a React-based Observatory dashboard. The system processes experience streams through modular brain regions connected by plastic pathways.

## Signal Flow

```
Input → Sensory Encoding → Association → Prediction → Action Selection → Safety Gate → Output
  ↑                            ↑                                                        │
  │                      Memory Replay                                                  │
  │                     (consolidation)                                                 │
  └─────────────────── Reward Signal ◄──────────────────────────────────────────────────┘
```

## Crate Architecture

```
engram-core          Pure computation. No platform dependencies.
    │                Types, neurons, synapses, learning rules, spike buffers.
    │                Compiles to native AND WebAssembly.
    │
engram-modules       Brain region implementations.
    │                SensoryEncoder, AssociativeMemory, PredictiveError,
    │                EpisodicMemory, ActionSelector, SafetyKernel.
    │                Each implements the BrainModule trait.
    │
engram-runtime       The 10-step cognitive loop orchestrator.
    │                Wires modules together, manages inter-module synapses,
    │                handles neuromodulatory signals, generates snapshots.
    │
engram-server        Axum WebSocket server.
    │                Streams RuntimeSnapshots to the Observatory at 30fps.
    │
engram-python        PyO3 bindings.
    │                Exposes PyRuntime to Python for pip-installable usage.
    │
engram-wasm          WebAssembly target.
                     87KB binary for browser-based demos.
```

## The 10-Step Cognitive Loop

Each tick of the simulation executes these steps in order:

1. **Sensory Encoding**: Observation vector → spike trains via population coding
2. **Route to Memory**: Sensory spikes propagate through learned synapses to associative memory
3. **Route to Predictor**: Sensory spikes also reach the predictive layer (actual input)
4. **Memory Step**: Associative memory processes input, generates predictions
5. **Prediction Error**: Compare predicted vs actual patterns, compute surprise
6. **Neuromodulation**: Update reward/surprise/arousal/inhibition signals from environment feedback
7. **Action Selection**: Spikes from cognitive modules compete via population vote
8. **Safety Evaluation**: Safety kernel checks proposed action against constraints
9. **Episodic Recording**: Current experience frame recorded; replay if interval elapsed
10. **Learning**: Three-factor STDP updates all inter-module pathway weights

## Learning System

### Three-Factor STDP

The primary learning mechanism uses eligibility traces for temporal credit assignment:

```
dw_ij = eta * e_ij * M(t)

where:
  e_ij = eligibility trace (accumulates STDP correlations, decays with tau_e)
  M(t) = modulatory signal (composite of reward, surprise, arousal, inhibition)
```

This allows the system to learn from delayed rewards: spike correlations are "remembered" in the eligibility trace until a reward signal arrives to confirm or deny them.

### Neuromodulatory System

Four global signals influence learning dynamics:

| Signal | Biological Analog | Effect |
|--------|------------------|--------|
| Reward | Dopamine | Drives learning direction (positive = strengthen, negative = weaken) |
| Surprise | Acetylcholine | Amplifies learning rate when input is unexpected |
| Arousal | Norepinephrine | Scales overall learning magnitude |
| Inhibition | Serotonin | Dampens exploration when performance is good |

### Safety Kernel

Two-tier protection:
- **Hard constraints**: Immutable rules (e.g., "never exceed energy budget")
- **Learned inhibitions**: Patterns learned from negative outcomes via STDP

## Data Flow to Dashboard

The runtime generates `RuntimeSnapshot` structs at 30fps containing:
- Module activity levels (6 regions)
- Recent spike events
- Prediction error scalar
- Memory formation events
- Safety veto events
- Performance metrics

Snapshots are serialized as MessagePack and streamed via WebSocket to the Observatory dashboard.

## Key Design Decisions

1. **Rust core, Python API**: Performance where it matters (spike processing), ergonomics where it matters (user-facing API).
2. **Event-driven, not clock-cycle**: Process only when spikes occur. At >90% sparsity, this is faster than dense simulation.
3. **CSR sparse synapses**: Compressed Sparse Row format. Memory-efficient for the primary access pattern (outgoing connections from a spiking neuron).
4. **Local learning, not backprop**: Each pathway learns independently through its own learning rule. No global backward pass.
5. **Safety as a first-class citizen**: The safety kernel is not optional middleware -- it's part of the cognitive loop.
