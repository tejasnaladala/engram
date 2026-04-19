# Procedural Maze RL Baselines

A fully reproducible procedural-maze benchmark with the following pattern, on the same audited test harness at 9x9 mazes:

- A 5-line egocentric wall-following heuristic solves **100%** of unseen instances
- A BC-distilled MLP (same architecture, same observation, same optimizer as MLP_DQN) reaches **97.4%**
- The best of seven HP-tuned modern reward-driven baselines (SB3 PPO, DQN, A2C across three LRs each, 70 runs) reaches **31.4%**, statistically tied with uniform Random (32.7%)
- A behavioral-cloning warm-start experiment (initialize MLP_DQN at the 97% distilled weights, fine-tune via standard DQN) collapses test success to **13.6%** across all 5 seeds

The neural policy class can express the maze-solving policy. Standard reward-driven RL does not discover this policy from random initialization, and actively pushes the network out of the high-performing basin even when initialized inside it.

## Paper

- [`PAPER_SHORT.pdf`](PAPER_SHORT.pdf) — 6-page paper with all headline tables and claims
- [`PAPER_PREVIEW.pdf`](PAPER_PREVIEW.pdf) — 11-page longer version with appendices
- [`paper.md`](paper.md) — full working draft (markdown)

## Headline Result

| Tier | Agent | Mean success (%) | sd | n |
|---|---|---|---|---|
| Oracle | BFSOracle | 100.0 | 0.0 | 20 |
| Heuristic (5-line, ego-only) | EgoWallFollowerLeft | 100.0 | 0.0 | 20 |
| Distillation (same arch as DQN) | DistilledMLP_from_BFSOracle | 97.4 | 2.5 | 20 |
| Random walk | NoBackRandom | 51.5 | 6.6 | 50 |
| Random walk | Random | 32.7 | 6.1 | 50 |
| Modern RL (SB3 DQN default LR) | SB3_DQN_lr5e-4 | 31.4 | 7.2 | 10 |
| Neural RL (custom) | MLP_DQN | 19.3 | 6.7 | 40 |
| Modern RL (SB3 A2C) | A2C_default | 8.4 | 4.3 | 10 |
| Modern RL (SB3 PPO best LR) | PPO_lr3e-4 | 6.0 | 6.9 | 10 |

## BC Warm-Start Collapse

| Seed | BC test (%) | Post-fine-tune (%) | Drop (pp) |
|---|---|---|---|
| 42 | 98.0 | 0.0 | -98.0 |
| 123 | 100.0 | 18.0 | -82.0 |
| 456 | 98.0 | 16.0 | -82.0 |
| 789 | 90.0 | 12.0 | -78.0 |
| 1024 | 100.0 | 22.0 | -78.0 |
| **Mean** | **97.2** | **13.6** | **-83.6** |

## Reproducing the Headline

```bash
git clone https://github.com/tejasnaladala/maze-rl-baselines
cd maze-rl-baselines
pip install -r requirements.txt

# Verify all SHA-256 hashes + recompute every numerical claim
python reproduce.py verify --manifest manifest_final.json

# 3-minute smoke test (covers every agent class on consumer GPU)
python smoke_test.py
```

## Key Experiments (re-runnable)

| Script | Runs | Wall time on RTX 5070 Ti |
|---|---|---|
| `launch_modern_baselines.py` | 70 (PPO/DQN/A2C × 3 LRs × 10 seeds) | ~7 hr |
| `launch_bc_warmstart.py` | 5 seeds (BC + DQN fine-tune) | ~50 min |
| `launch_loopy_pilot.py` | 25 (5 agents × 5 seeds) | ~4 min |
| `launch_ppo_shaped.py` | 10 seeds | ~85 min |
| `launch_policy_distillation.py` | distillation headline | ~30 min |

## Methodology

- **20+ seeds per cell**, paired bootstrap with Holm-Bonferroni correction
- **SHA-256 manifest** of every result file (4,200+ JSON records)
- **Code-hash pinned** per result (current main-sweep hash: `ed681d75c27fe352`)
- **Single-file statistics pipeline** (`stats_pipeline.py`): paired bootstrap, Mann-Whitney U, Cohen d, Holm-Bonferroni, BCa
- **Harness-bug audit trail** in §3.2.1 of the paper (caught and fixed during development)

## What This Paper Is and Is Not

**It is**: a small, well-audited benchmark with one specific narrow falsifiable claim, supported by ~3,500 runs across 20+ seeds per cell, complete reproducibility apparatus, and a striking BC warm-start result that did not exist in prior work.

**It is not**: a method paper, a paradigm-shifting contribution, or a result on large-scale environments. It is documented at 9x9 mazes and replicated on 4 MiniGrid environments. Larger-scale generalization is open follow-up.

## License

Apache-2.0. See [`LICENSE`](LICENSE).

## Citing

Single-author preprint. Citation format will be added when arXiv id is assigned.

## Contact

Tejas Naladala — `tejas.naladala@gmail.com`

Independent reproduction is welcomed and encouraged.
