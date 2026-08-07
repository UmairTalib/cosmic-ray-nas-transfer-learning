# Task 1 — Loading Pre-Trained NAS-Bench-201 Weights

The starting point of the project: the professor provided 20 pre-trained PyTorch
checkpoints (`.pth`) for two NAS-Bench-360 downstream tasks — **DarcyFlow**
(fluid-flow prediction, 10 files) and **NinaPro** (EMG gesture classification,
10 files) — saved with an older version of the XAutoDL library. This script
rebuilds the matching architectures in the current library version and loads
the weights correctly.

| Dataset | Files | Stem channels | Spatial input | Reduction-cell key format |
|---|---|---|---|---|
| DarcyFlow | 10 | 3 | 85×85 | old (`layers.N.*`) |
| NinaPro | 10 | 1 | 16×52 | new (`conv_a`/`conv_b`) |

## Three problems solved

1. **Spatial-size mismatch (DarcyFlow, 85×85 inputs).** The current XAutoDL
   downsampling block's two internal paths disagree on output size for an
   85-pixel input (42 vs. 43 pixels), causing a shape error when they're added.
   The original block used matched-stride convolutions on both paths. Fixed by
   re-implementing the legacy block (`LegacyResNetBasicblock`).
2. **State-dict key mismatch.** Older checkpoints name certain layers
   differently than the current library expects. DarcyFlow files use the old
   naming, NinaPro files use the new naming. Fixed by detecting the format
   per file and loading accordingly.
3. **Input-channel mismatch.** The library assumes 3-channel (RGB) input by
   default; NinaPro's EMG data is single-channel. Fixed by patching the stem
   layer to 1 channel before loading weights.

## Usage
```bash
python load_pth.py
```
Requires the DarcyFlow/NinaPro `.pth` checkpoint files (not included in this
repository — obtained from the course materials) and `xautodl` installed
(see the repo-root `requirements.txt`).
