# Task 4 — Performance Prediction + Transfer Learning to Cosmic

> **Status: superseded by [task5](../task5_final_predictor_selection/).** This was
> our first full-scale attempt (XGBoost predictor on all 15,625 architectures).
> It met only 5 of 9 Kendall-Tau targets — kept here for the record, since
> task5's decision to switch to an SVR ensemble was motivated directly by this
> run's shortfall. Task5 is the version that should actually be reproduced.

## Research Question
Which pre-training dataset (CIFAR-10, CIFAR-100, ImageNet16-120) transfers best to
cosmic ray detection on Hubble Space Telescope images?

## Steps

### Step 1 — Improved Performance Predictor (`step1_predictor.py`)
Uses the full NAS-Bench-201 ground truth (15,625 architectures) instead of just the
47 we had before. Trains an XGBoost surrogate model at different training set sizes
(32, 128, 1024 architectures) and evaluates Kendall Tau on a fixed held-out test set.

**Professor's target benchmarks (Kendall Tau):**

| Train size | CIFAR-10 | CIFAR-100 | ImageNet16-120 |
|-----------|---------|---------|---------|
| 32        | 0.39    | 0.39    | 0.40    |
| 128       | 0.54    | 0.56    | 0.56    |
| 1024      | 0.74    | 0.74    | 0.74    |

Outputs: `results/predictor_results.json`, `results/predictor_tau_*.png`

First run builds a cache (`results/nb201_cache.npz`) from all 15,625 NATS-tss files.
Cache takes ~5 minutes to build once; all subsequent runs load in <1 second.

### Step 2 — Random Search + Selection (`step2_random_search.py`)
1. Randomly samples 300 architectures from the NB201 search space (15,625 total)
2. Uses the Step 1 predictor (trained on 1024 architectures) to rank the 300
3. Selects the top 10 per source dataset (CIFAR-10, CIFAR-100, ImageNet16-120)
   = 30 architectures total to fine-tune

Outputs: `results/random_search_results.json`

### Step 3 — Cosmic Fine-Tuning (`step3_finetune.py`)
Fine-tunes the 30 selected architectures on the Cosmic dataset (HST cosmic ray detection).
Each network uses its source-dataset pre-trained weights (from NATS-tss) as starting point.

Two-phase training:
- Phase 1: head only (EinspaceCosmicHead frozen backbone), 3 epochs
- Phase 2: full fine-tune, 5 epochs

Records test AUROC for each architecture and compares across source datasets.

Outputs: `results/finetune_results.json`, `results/transfer_comparison.png`

## Architecture Encoding
Every NB201 architecture has 6 directed edges, each assigned one of 5 operations:
`none`, `skip_connect`, `nor_conv_1x1`, `nor_conv_3x3`, `avg_pool_3x3`

This is encoded as a 30-dimensional binary vector (6 edges × 5 ops, one-hot per edge).

## File Structure
```
task4/
  README.md               This file
  step1_predictor.py      Performance predictor on full NB201 data
  step2_random_search.py  Random sample 300, rank, select top 10 per dataset
  step3_finetune.py       Fine-tune 30 networks on Cosmic, compare transfer
  results/
    nb201_cache.npz       Cached encodings + accuracies (15,625 archs, built once)
    predictor_results.json
    predictor_tau_cifar10.png
    predictor_tau_cifar100.png
    predictor_tau_imagenet.png
    random_search_results.json
    finetune_results.json
    transfer_comparison.png
```

## Dependencies
- Python environment: see [`requirements.txt`](../../requirements.txt) at the repo root
- NATS-tss data: `NATS-tss-v1_0-3ffb9-simple/` (15,625 architecture files — see repo root README "Data" section for source)
- Cosmic dataset: `Cosmic/` (128×128 HST image patches — see repo root README "Data" section)
- `../task3_transfer_prototype/cosmic_data.py`: dataset + loss + metric utilities (reused here)

## How to Run
```bash
# from the repo root, with the NATS-tss and Cosmic data in place (see main README)
pip install -r requirements.txt

python src/task4_full_pipeline_v1/step1_predictor.py   # ~5 min first run (cache build), <1 min after
python src/task4_full_pipeline_v1/step2_random_search.py
python src/task4_full_pipeline_v1/step3_finetune.py    # ~2-3 hours (30 networks x 8 epochs)
```
