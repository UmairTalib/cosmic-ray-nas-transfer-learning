# Task 3 — First Transfer-Learning Experiment, Predictor Prototype, Evolutionary Search

The first small-scale proof of concept for the whole project: does transferring
a NAS-Bench-201 backbone to the Cosmic ray detection task actually work? This
folder also prototypes the two search strategies later scaled up in
[task4](../task4_full_pipeline_v1/) and [task5](../task5_final_predictor_selection/).

**Full write-up with all results: [`EXPERIMENT_NOTES.md`](EXPERIMENT_NOTES.md).**

## Contents

| File | Purpose |
|---|---|
| `cosmic_data.py` | Cosmic dataset loader, masked BCE loss, pixel-wise AUROC metric |
| `run_experiment.py` | Main experiment: baseline → Phase 1 (head-only) → Phase 2 (full fine-tune) |
| `finetune_cosmic.py` | Standalone, CLI-configurable fine-tuner |
| `setup_cosmic.py` | One-time Cosmic data setup (builds file-path lists) |
| `performance_predictor.py` | First (47-architecture) performance predictor — Random Forest / XGBoost, Kendall-Tau p-sweep |
| `evolutionary_search.py` | Evolutionary NAS with an XGBoost surrogate — see [`../../docs/SURROGATE_MODEL_EXPLAINED.md`](../../docs/SURROGATE_MODEL_EXPLAINED.md) for a full breakdown of this run's results |

## Headline result

Transfer learning from the best NinaPro architecture (arch 006186, see
[task2](../task2_architecture_ranking/)) to Cosmic ray detection, two-phase
fine-tuning:

| Stage | Val AUROC | Test AUROC |
|---|---|---|
| Baseline (random head, untrained) | 0.489 | 0.444 |
| Phase 1 (head only) | 0.589 | 0.601 |
| **Phase 2 (full fine-tune)** | **0.976** | **0.944** |

This validated the entire transfer-learning recipe (two-phase, frozen-then-unfrozen)
reused at scale in tasks 4 and 5.

## Usage
```bash
python setup_cosmic.py       # one-time: build train/val/test path lists
python run_experiment.py     # baseline + Phase 1 + Phase 2, saves results.json
python performance_predictor.py
python evolutionary_search.py
```
Requires the Cosmic dataset (see repo-root README "Data" section) and the
weight files from [task1](../task1_load_pretrained_weights/)/[task2](../task2_architecture_ranking/).
