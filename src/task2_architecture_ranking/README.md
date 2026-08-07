# Task 2 — Ranking Architectures per Dataset

Given the weight files loadable via [task1](../task1_load_pretrained_weights/),
this script determines the single best architecture per dataset among the
provided checkpoints.

## What it does

1. Scans all 27 `.pickle.pbz2` result files (CIFAR-10 / CIFAR-100 /
   ImageNet16-120) and reads each architecture's final-epoch test accuracy,
   averaged across random-seed reruns.
2. Scans all 20 `.pth` files (DarcyFlow / NinaPro) and reads the final-epoch
   validation metric — **note the sort direction differs by task**:
   DarcyFlow reports a *loss* (lower is better), NinaPro reports *accuracy*
   (higher is better).
3. Prints a ranked table per dataset and highlights the best architecture.

## Key result

The best NinaPro architecture found here (**arch 006186**, 94.39% accuracy)
was selected as the starting backbone for
[task3](../task3_transfer_prototype/)'s first transfer-learning experiment —
chosen because its 1-channel input already matches Cosmic's grayscale images,
and its weights loaded with zero missing keys.

## Usage
```bash
python investigate_weights.py
```
Requires the same weight files as [task1](../task1_load_pretrained_weights/).
