# Task 5 — NAS Transfer Learning for Cosmic Ray Detection

**Course:** RAML, Uni Siegen
**Supervisor:** Jun.-Prof. Dr. Jovita Lukasik
**Authors:** Muhammad Salim, Umair Talib

> **This is the core local contribution of the project**: the final predictor
> (9/9 Kendall-Tau targets) and the architecture selection used for fine-tuning.
> `step3_finetune.py` here reproduces the **pipeline-validation baseline**
> (random initialisation). The **definitive fine-tuning results**, using the
> genuine NAS-Bench-201 pre-trained weights extracted on the Omni HPC cluster,
> are reported in [`../../report/RAML_Final_Report.pdf`](../../report/RAML_Final_Report.pdf)
> and (once added) [`../../results_omni_cluster/`](../../results_omni_cluster/).

---

## Overview

This folder implements the professor's June 27 requirements:

1. **Improved performance predictor** using all 15,625 NAS-Bench-201 architectures, targeting the Kendall Tau benchmarks at different training set sizes.
2. **Random search**: sample 300 architectures, rank by predictor, select top 10 per dataset.
3. **Fine-tune** the 30 selected architectures on the Cosmic ray detection dataset and compare which source dataset transfers best.

---

## Key Improvement Over task4: Path Encoding

The predictor in task4 used a **30-dim one-hot encoding** (6 edges × 5 ops, edge-independent).  
Task5 adds a **180-dim path encoding** for a total **210-dim feature vector**.

### NB201 Cell DAG

```
Node 0 (input) --e0--> Node 1
Node 0         --e1--> Node 2
Node 1         --e2--> Node 2
Node 0         --e3--> Node 3 (output)
Node 1         --e4--> Node 3
Node 2         --e5--> Node 3
```

All paths from input (0) to output (3):

| Path | Edges used | Feature dimension |
|------|-----------|-------------------|
| 0→3 | e3 | 5¹ = 5 |
| 0→1→3 | e0, e4 | 5² = 25 |
| 0→2→3 | e1, e5 | 5² = 25 |
| 0→1→2→3 | e0, e2, e5 | 5³ = 125 |
| **Total** | | **180** |

Each path gets a one-hot over all possible operation combinations on its edges.  
For example, path 0→1→3 with ops `(nor_conv_3x3, nor_conv_3x3)` activates exactly one of the 25 features.  
This captures **interaction effects** that edge-independent one-hot cannot represent.

---

## Files

| File | Purpose |
|------|---------|
| `step1_predictor.py` | 210-dim predictor, Kendall Tau p-sweep, plots |
| `step2_random_search.py` | Sample 300 archs, rank, select top 30 |
| `step3_finetune.py` | Fine-tune 30 networks on Cosmic (two-phase) |
| `results/` | All outputs (JSON, PNG) |

---

## How to Run

```bash
# from the repo root, with the data in place (see main README "Data" section)
pip install -r requirements.txt

# Step 1: ~2-3 min (rebuilds the 210-dim encoding cache on first run)
python src/task5_final_predictor_selection/step1_predictor.py

# Step 2: ~1 min
python src/task5_final_predictor_selection/step2_random_search.py

# Step 3: ~4-5 hours on GPU (30 networks x 8 epochs, random-init baseline;
# see note above regarding the definitive pretrained-weight run)
python src/task5_final_predictor_selection/step3_finetune.py
```

---

## Data Sources

### Architecture accuracy data
- **NATS-tss-v1_0-3ffb9-simple** (local): all 15,625 NB201 architectures, test accuracy for CIFAR-10/100/ImageNet16-120 (mean over 3 seeds: 777, 888, 999), epoch 199.
- **NASLib** (`get_dataset_api('nasbench201', ...)`): downloaded via gdown from the official NASLib Google Drive folder. Contains training/validation curves for all 15,625 architectures.

### Pre-trained weights
The professor asked to use NAS-bench-201 pre-trained weights for fine-tuning.  
- The **simple** NATS-tss version (which we have) contains accuracy data only — **no model state_dicts**.
- The **full** NATS-tss version (~15 GB) contains pre-trained weights but requires a separate download.
- The **NASLib pickle files** (`nb201_cifar10_full_training.pickle` etc.) also contain only accuracy curves, not model weights.

**Consequence:** Fine-tuning is performed from **random initialisation**.  
This still demonstrates the complete NAS → predictor → transfer learning pipeline.  
Results are excellent (0.945–0.982 AUROC on Cosmic validation set, all 30 networks).  
With proper pre-trained weights, convergence would be faster and performance equal or higher.

---

## Architecture Encoding Reference

Operations (5 total):
- `N` = none
- `S` = skip_connect
- `1` = nor_conv_1x1
- `3` = nor_conv_3x3
- `P` = avg_pool_3x3

Example: arch string `|nor_conv_3x3~0|+|nor_conv_1x1~0|nor_conv_3x3~1|+|none~0|nor_conv_3x3~1|nor_conv_3x3~2|`  
=> label `3-1-3-N-3-3`

---

## Professor's Target Kendall Tau vs Results — ALL 9 TARGETS MET

**Ground truth:** NAS-Bench-201 *validation* accuracy read directly from NASLib
(`nb201_all.pickle` → `eval_acc1es`), i.e. the ground truth the professor pointed to.

**Predictor:** rank-average ensemble of two RBF Support-Vector-Regressors on the
210-dim encoding — one tuned for small budgets (γ=0.02), one for large budgets
(γ='scale'). SVR dominates tree models (XGBoost/RF) at small training sizes, and
rank-averaging blends the two regimes. Mean over 20 random train draws, fixed
2000-architecture held-out test set.

| Dataset | p=32 | target | p=128 | target | p=1024 | target |
|---------|------|--------|-------|--------|--------|--------|
| CIFAR-10 | **0.439** ✓ | 0.39 | **0.596** ✓ | 0.54 | **0.741** ✓ | 0.74 |
| CIFAR-100 | **0.437** ✓ | 0.39 | **0.601** ✓ | 0.56 | **0.750** ✓ | 0.74 |
| ImageNet16-120 | **0.433** ✓ | 0.40 | **0.596** ✓ | 0.56 | **0.743** ✓ | 0.74 |

**9 / 9 targets met.** Earlier XGBoost/RF experiments (see `experiment_models.py`)
met only 5/9 — the switch to SVR closed the small-budget gap that trees could not.

### Why SVR beat gradient-boosted trees
At p=32 a 500-tree depth-6 XGBoost simply memorises the 32 points (Kendall ≈ 0.30).
An RBF-SVR is a smooth, heavily-regularised interpolator that generalises far
better in the low-data regime (Kendall ≈ 0.44 at the same p). See
`experiment_models.py` and `experiment_final.py` for the full model comparison.

---

## Transfer Learning Results (Step 3) — pipeline-validation baseline, random init

30 networks fine-tuned on Cosmic (two-phase: 3 epochs head-only + 5 epochs full, random init),
using the final 18-architecture selection listed in `results/random_search_results.json`.

| Source dataset | mean AUROC | max AUROC | min AUROC |
|----------------|-----------|-----------|-----------|
| CIFAR-10 | 0.9583 | 0.9808 | 0.9452 |
| CIFAR-100 | 0.9665 | 0.9786 | 0.9517 |
| **ImageNet16-120** | **0.9677** | **0.9823** | 0.9191 |

**Best source dataset:** ImageNet16-120 (highest mean and highest single-run AUROC, 0.9823).
See `results/finetune_results.json` for the full per-architecture breakdown.

**This is a pipeline-validation baseline (random initialisation), run before the
official pre-trained weights were available.** The definitive fine-tuning run —
same 18 architectures, genuine NAS-Bench-201 pre-trained weights, executed on the
Omni HPC cluster — reaches a substantially tighter and higher AUROC band
(0.976–0.989); see [`../../report/RAML_Final_Report.pdf`](../../report/RAML_Final_Report.pdf)
for the verified final numbers.
