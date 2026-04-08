# Contributing to Engram

Thank you for your interest in contributing to Engram! This document provides guidelines for contributing.

## Ways to Contribute

### Good First Issues

- **Add a learning rule**: Implement BCM, Oja's rule, or anti-Hebbian learning in `crates/engram-core/src/learning_rule.rs`. Follow the `LearningRule` trait.
- **Add an environment**: Create a new Gymnasium-compatible environment in `python/engram/environments/`.
- **Add a dashboard widget**: Build a new visualization component in `dashboard/src/components/`.
- **Improve documentation**: Fix typos, add examples, clarify explanations.
- **Write tests**: Increase test coverage for any module.

### Intermediate Contributions

- **New brain region module**: Implement a new `BrainModule` in `crates/engram-modules/src/`.
- **Benchmark comparison**: Add a baseline comparison in `benchmarks/`.
- **YAML brain configs**: Help build the configuration parser.
- **Python API improvements**: Ergonomic wrappers in `python/engram/`.

### Advanced Contributions

- **Learning rule research**: Implement e-prop, surrogate gradients, or novel plasticity rules.
- **Hardware backends**: Lava/Loihi integration, Akida support.
- **Multi-agent simulation**: Architecture for multiple brains in one environment.
- **Performance optimization**: Profile and optimize the Rust core.

## Development Setup

### Prerequisites

- Rust 1.75+ (`rustup install stable`)
- Python 3.10+ with `maturin` (`pip install maturin`)
- Node.js 18+ (for dashboard)

### Build

```bash
# Rust core
cargo check --workspace
cargo test --workspace

# Python bindings
maturin develop

# Dashboard
cd dashboard && npm install && npm run dev
```

### Code Style

- **Rust**: Follow `rustfmt` defaults. Run `cargo fmt` before committing.
- **Python**: Follow PEP 8. Type hints required for public APIs.
- **TypeScript**: Follow the existing style in `dashboard/src/`.

### Pull Request Process

1. Fork the repo and create a branch from `master`.
2. Add tests for new functionality.
3. Ensure `cargo test --workspace` passes.
4. Update documentation if needed.
5. Open a PR with a clear description.

### Commit Messages

Use conventional commits:
```
feat: add BCM learning rule
fix: correct eligibility trace decay
docs: add quickstart tutorial
test: add three-factor STDP integration test
```

## Architecture Overview

```
crates/
  engram-core/      Core primitives (neurons, synapses, learning rules)
  engram-modules/   Brain region implementations
  engram-runtime/   Cognitive loop orchestrator
  engram-server/    WebSocket server for dashboard
  engram-python/    PyO3 bindings
  engram-wasm/      WebAssembly target
python/             Python package and environments
dashboard/          React Observatory dashboard
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed system design.

## Code of Conduct

Be respectful. Be constructive. Be welcoming. We're building something ambitious together.

## License

By contributing, you agree that your contributions will be licensed under Apache 2.0.
