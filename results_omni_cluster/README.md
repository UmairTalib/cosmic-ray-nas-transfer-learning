# Omni HPC Cluster — Pre-Trained Weight Extraction & Definitive Fine-Tuning

**Status: results already reported in [`../report/RAML_Final_Report.pdf`](../report/RAML_Final_Report.pdf) — code/artifacts to be added here.**

This directory is reserved for Umair's contribution: the work executed on the
University of Siegen Omni HPC cluster that produced the project's definitive
results.

## What belongs here

1. **Weight extraction code** — scripts used to download the NAS-Bench-201
   full weight archive (~360 GB, split across 12 parts), reassemble it, and
   extract the `net_state_dict` tensors for the 18 architectures selected in
   [`../src/task5_final_predictor_selection/`](../src/task5_final_predictor_selection/)
   (a starting kit for this step is already provided in
   [`../src/task5_final_predictor_selection/omni_handoff/`](../src/task5_final_predictor_selection/omni_handoff/)).
2. **The final fine-tuning script/config** used on the cluster — i.e. the
   version of `step3_finetune.py` modified to load the genuine pre-trained
   weights (`load_state_dict(..., strict=False)`) instead of the
   random-initialisation baseline in task5.
3. **Raw results** — per-run logs / JSON output from the 30 fine-tuning runs.
4. Any cluster-specific setup notes (SLURM/job scripts, environment modules, etc.).

## Reported results (for reference — see the final report for full detail)

| Source dataset | Mean AUROC | Max AUROC | Min AUROC | Runs |
|---|---|---|---|---|
| CIFAR-10 | 0.9843 | 0.9875 | 0.9760 | 10 |
| CIFAR-100 | 0.9857 | 0.9886 | 0.9841 | 10 |
| ImageNet16-120 | 0.9862 | 0.9886 | 0.9823 | 10 |
| **All combined** | **0.9854** | **0.9886** | **0.9760** | **30** |

Compute: single NVIDIA GPU node, Omni HPC cluster (CUDA 12.4); ~5–12 GPU-minutes
per run, ~3.5 hours total.
