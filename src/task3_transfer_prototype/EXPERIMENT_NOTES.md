# Cosmic Fine-Tuning Experiment

## Goal

Take a pre-trained NAS-Bench-201 TinyNetwork (trained on NinaPro EMG classification)
and fine-tune it on the Cosmic dataset for binary pixel segmentation (cosmic ray detection).

---

## Approach

### Why NinaPro as the starting backbone?

NAS-Bench-360 provides pre-trained weights for downstream tasks including NinaPro and
DarcyFlow. The Cosmic dataset is a **new downstream task** not covered by the original
benchmark. We use the best NinaPro architecture (arch-006186) as the starting point
because:
- NinaPro uses a 1-channel input — directly compatible with Cosmic's 1-channel images
- 100% of pre-trained weights load cleanly (format=new, no missing keys)
- It is the best-performing architecture on the NinaPro task (AUROC 94.39%)

### Architecture

```
Input  (batch, 1, 128, 128)
         |
    [NB201 TinyNetwork backbone]
         |
    stem (Conv2d 3x3, 1 -> 16 ch)     (batch, 16, 128, 128)
    5x InferCell  (stage 1)            (batch, 16, 128, 128)
    ResNetBasicblock stride=2          (batch, 32,  64,  64)
    5x InferCell  (stage 2)            (batch, 32,  64,  64)
    ResNetBasicblock stride=2          (batch, 64,  32,  32)
    5x InferCell  (stage 3)            (batch, 64,  32,  32)
    BN + ReLU (lastact)                (batch, 64,  32,  32)
         |
    [EinspaceCosmicHead  -- professor-provided]
         |
    AdaptiveAvgPool2d(128, 128)        (batch, 64, 128, 128)
    Conv2d(64, 1, kernel=1)            (batch,  1, 128, 128)
         |
    Output: pixel-level logit map
```

Total parameters: 371,473

### Loss function: CosmicBCEWithLogitsLoss

Standard binary cross-entropy with logits, but ignoring pixels where the target mask
is negative (IGNORE_FLAG = -9999). These are pixels near bright stars or chip edges
that cannot be reliably labelled.

```
valid pixels: target >= 0  (0 = background, 1 = cosmic ray)
ignored:      target < 0   (-9999 or -9998)
```

### Metric: AUROC

Area Under the ROC Curve, computed pixel-wise over all valid pixels in the dataset.
- 1.0 = perfect detection
- 0.5 = random (coin flip)

Uses raw logit scores (not thresholded) — the detector's ranking quality matters,
not just a fixed threshold.

### Dataset

- Source: deepCR ACS-WFC (Hubble Space Telescope)
- Filter: f435w, GAL (galaxy) fields only
- Train: 4,347 image patches  |  Val: 483  |  Test: 420
- Each patch: 128x128 crop from a 256x256 HST sub-image

---

## Fine-Tuning Strategy

### Why two phases?

Directly fine-tuning the entire network risks **catastrophic forgetting** — updating
the backbone with a large learning rate can destroy the useful features already learned
during NinaPro training.

| Phase | Layers Trained | Learning Rate | Rationale |
|-------|---------------|---------------|-----------|
| Phase 1 | Head only (EinspaceCosmicHead) | 1e-3 | Adapt the head to produce meaningful cosmic predictions. Backbone stays fixed as a feature extractor. |
| Phase 2 | All layers | 1e-4 | Allow backbone to specialize for astronomy images at a cautious rate. |

Phase 2 starts from the best Phase-1 checkpoint, so the head is already warm.

---

## Results

| Stage | Val AUROC | Test AUROC | Delta vs Baseline |
|-------|-----------|------------|-------------------|
| Baseline (random head, 0 epochs) | 0.4890 | 0.4435 | — |
| Phase 1 (head only, 3 epochs) | 0.5887 | 0.6007 | +0.1571 |
| Phase 2 (full fine-tune, 5 epochs) | **0.9762** | **0.9436** | **+0.5001** |

### Training curves

**Phase 1 — Head only (backbone frozen, 65 trainable params)**

| Epoch | Train Loss | Val AUROC |
|-------|-----------|-----------|
| 1 | 0.1628 | 0.5667 |
| 2 | 0.0511 | 0.5087 |
| 3 | 0.0432 | 0.5357 |

**Phase 2 — Full fine-tune (all 371,473 params)**

| Epoch | Train Loss | Val AUROC |
|-------|-----------|-----------|
| 1 | 0.0329 | 0.9500 |
| 2 | 0.0257 | 0.9756 |
| 3 | 0.0240 | 0.9681 |
| 4 | 0.0230 | **0.9762** |
| 5 | 0.0224 | 0.9612 |

Best checkpoint: epoch 4 of Phase 2 (val AUROC = 0.9762)

### Key observations

1. **Baseline below chance (0.44)**: the random head outputs predictions that are slightly
   anti-correlated with the cosmic ray mask — expected when the head weights are random
   and the backbone was never trained on images.

2. **Phase 1 is noisy**: with only 65 trainable parameters (the 1x1 conv has 64 weights
   + 1 bias), the head has very limited capacity. AUROC oscillates between 0.51 and 0.57.
   The backbone features are not yet useful for this domain.

3. **Phase 2 jumps immediately**: at epoch 1 of full fine-tuning, AUROC leaps to 0.950.
   Once the backbone layers are free to adapt, the entire network quickly specializes for
   cosmic ray detection. Loss continues decreasing steadily.

4. **Final test AUROC of 0.9436** is very close to the reference deepCR model (~0.97),
   achieved in just 8 total training epochs on a single GPU.

---

## How to Reproduce

```powershell
# Activate environment
C:\Users\Flash\NASLib_venv\Scripts\activate

# From RAML/ directory:

# Step 1 (one-time): setup cosmic data
python task3/setup_cosmic.py

# Step 2: run the experiment
python task3/run_experiment.py

# Results saved to task3/results.json
# Checkpoints: task3/cosmic_phase1_best.pth  and  task3/cosmic_phase2_best.pth
```

---

## Files

| File | Description |
|------|-------------|
| task3/cosmic_data.py | Dataset, loss, and AUROC metric (from professor's reference) |
| task3/run_experiment.py | Main experiment: baseline eval + Phase 1 + Phase 2 + report |
| task3/finetune_cosmic.py | Standalone fine-tuner (configurable via CLI) |
| task3/setup_cosmic.py | One-time data setup (generates dir lists) |
| task3/results.json | Saved numerical results from run_experiment.py |
| task3/cosmic_phase1_best.pth | Best Phase-1 checkpoint |
| task3/cosmic_phase2_best.pth | Best Phase-2 checkpoint |
| Cosmic/npy_train/ | Extracted training data (~4,830 .npy patches) |
| Cosmic/npy_test/ | Extracted test data (~420 .npy patches) |
