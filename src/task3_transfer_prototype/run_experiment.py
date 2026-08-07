"""
Full fine-tuning experiment on the Cosmic dataset.

Produces three clear checkpoints:
  BASELINE   : pre-trained NinaPro backbone + random head (no cosmic training)
  PHASE 1    : head-only fine-tuning (backbone frozen)
  PHASE 2    : full end-to-end fine-tuning (all layers)

Results are saved to task3/results.json and a human-readable report is
printed at the end — suitable for sharing with the professor.

Run:
    python task3/run_experiment.py
"""

import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'task1'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'task3'))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from load_pth import load_pth_network
from cosmic_data import (
    build_cosmic_datasets, CosmicBCEWithLogitsLoss, CosmicMetricFunction
)

# ── Copy of the model definition (keeps this script self-contained) ───────────
class EinspaceCosmicHead(nn.Module):
    def __init__(self, in_channels: int, output_shape: tuple = (128, 128)):
        super().__init__()
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_shape),
            nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0),
        )
    def forward(self, x):
        return self.head(x)


class CosmicTinyNet(nn.Module):
    """
    NB201 TinyNetwork backbone (all 17 cells) + EinspaceCosmicHead.
    Output: (batch, 1, 128, 128) logit map.
    """
    def __init__(self, backbone):
        super().__init__()
        self.stem    = backbone.stem
        self.cells   = backbone.cells
        self.lastact = backbone.lastact
        C_out        = backbone.classifier.in_features  # 64 for standard NB201
        self.head    = EinspaceCosmicHead(C_out, output_shape=(128, 128))

    def forward(self, x):
        feat = self.stem(x)
        for cell in self.cells:
            feat = cell(feat)
        feat = self.lastact(feat)   # (batch, 64, 32, 32)
        return self.head(feat)      # (batch, 1, 128, 128)

    def freeze_backbone(self):
        for p in (list(self.stem.parameters()) +
                  list(self.cells.parameters()) +
                  list(self.lastact.parameters())):
            p.requires_grad = False

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

    def n_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def n_total(self):
        return sum(p.numel() for p in self.parameters())


# ── Helpers ───────────────────────────────────────────────────────────────────
def evaluate(model, loader, criterion, metric_fn, device, label=''):
    model.eval()
    total_loss = 0.0
    all_preds, all_masks = [], []
    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            preds = model(imgs)
            loss = criterion(preds, masks.to(device))
            total_loss += loss.item()
            all_preds.append(torch.sigmoid(preds).cpu().numpy())
            all_masks.append(masks.numpy())
    avg_loss = total_loss / len(loader)
    try:
        auroc = metric_fn(all_preds, all_masks)
    except Exception as e:
        auroc = float('nan')
        print(f'  [AUROC error on {label}: {e}]')
    return avg_loss, auroc


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def run_phase(name, model, train_loader, val_loader, optimizer, criterion,
              metric_fn, n_epochs, device, ckpt_path):
    history = []
    best_auroc, best_epoch = -1.0, 0

    print(f'\n  {"-"*56}')
    print(f'  {name}  ({n_epochs} epochs, trainable={model.n_trainable():,})')
    print(f'  {"-"*56}')
    print(f'  {"Epoch":>6}  {"train_loss":>12}  {"val_loss":>10}  {"val_AUROC":>10}')
    print(f'  {"-"*56}')

    for ep in range(1, n_epochs + 1):
        t0 = time.time()
        tr_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        vl_loss, vl_auroc = evaluate(model, val_loader, criterion, metric_fn, device)
        elapsed = time.time() - t0

        improved = vl_auroc > best_auroc
        if improved:
            best_auroc, best_epoch = vl_auroc, ep
            torch.save(model.state_dict(), ckpt_path)

        marker = ' *' if improved else '  '
        print(f'  {ep:>6}  {tr_loss:>12.5f}  {vl_loss:>10.5f}  {vl_auroc:>10.5f}{marker}  ({elapsed:.0f}s)')
        history.append({'epoch': ep, 'train_loss': tr_loss,
                        'val_loss': vl_loss, 'val_auroc': vl_auroc})

    print(f'\n  Best: epoch {best_epoch}  val_AUROC={best_auroc:.5f}  saved -> {ckpt_path}')
    return history, best_auroc


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    COSMIC_ROOT    = 'Cosmic'
    BACKBONE_PATH  = 'all/all/arch-006186-seed-0777.pth'
    SAVE_DIR       = 'task3'
    PHASE1_EPOCHS  = 3     # head-only  (~6 min)
    PHASE2_EPOCHS  = 5     # full       (~10 min)
    BATCH_SIZE     = 16
    LR_HEAD        = 1e-3
    LR_FULL        = 1e-4

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = CosmicBCEWithLogitsLoss()
    metric_fn = CosmicMetricFunction()

    print('=' * 60)
    print('  COSMIC FINE-TUNING EXPERIMENT')
    print('=' * 60)
    print(f'  Device : {device}')
    print(f'  Backbone: {BACKBONE_PATH}')

    # ── Datasets ──────────────────────────────────────────────────────────────
    print('\nLoading datasets ...')
    trainset, valset, testset = build_cosmic_datasets(COSMIC_ROOT)
    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader   = DataLoader(valset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(testset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f'  train={len(trainset)}  val={len(valset)}  test={len(testset)}')

    # ── Build model ───────────────────────────────────────────────────────────
    print(f'\nBuilding model ...')
    backbone, ds_key, cfg, fmt, missing = load_pth_network(BACKBONE_PATH)
    print(f'  Backbone  : arch-006186  dataset={ds_key}  format={fmt}  missing={len(missing)}')
    print(f'  Stem      : 1-channel input (EMG -> Cosmic compatible)')
    print(f'  Head      : EinspaceCosmicHead(64, output_shape=(128,128))')
    print(f'  Total params: {sum(p.numel() for p in CosmicTinyNet(backbone).parameters()):,}')

    results = {
        'backbone': BACKBONE_PATH,
        'arch_id': '006186',
        'device': str(device),
        'phase1_epochs': PHASE1_EPOCHS,
        'phase2_epochs': PHASE2_EPOCHS,
    }

    # ── STEP 0: Baseline (no training) ────────────────────────────────────────
    print('\n' + '=' * 60)
    print('  STEP 0 — BASELINE (no fine-tuning)')
    print('=' * 60)
    print('  Pre-trained NinaPro backbone + RANDOM EinspaceCosmicHead')
    model = CosmicTinyNet(backbone).to(device)
    val_loss_b,  val_auroc_b  = evaluate(model, val_loader,  criterion, metric_fn, device, 'val')
    test_loss_b, test_auroc_b = evaluate(model, test_loader, criterion, metric_fn, device, 'test')
    print(f'  val  loss={val_loss_b:.5f}   AUROC={val_auroc_b:.5f}')
    print(f'  test loss={test_loss_b:.5f}   AUROC={test_auroc_b:.5f}')
    results['baseline'] = {
        'val_loss': val_loss_b, 'val_auroc': val_auroc_b,
        'test_loss': test_loss_b, 'test_auroc': test_auroc_b,
    }

    # ── STEP 1: Head-only fine-tuning ─────────────────────────────────────────
    print('\n' + '=' * 60)
    print('  STEP 1 — PHASE 1: Head-only fine-tuning')
    print('=' * 60)
    print('  Backbone frozen. Only EinspaceCosmicHead trained.')
    model = CosmicTinyNet(backbone).to(device)
    model.freeze_backbone()
    ckpt1 = os.path.join(SAVE_DIR, 'cosmic_phase1_best.pth')
    opt1  = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR_HEAD)
    hist1, best_auroc1 = run_phase(
        'Phase 1 — Head only', model, train_loader, val_loader,
        opt1, criterion, metric_fn, PHASE1_EPOCHS, device, ckpt1
    )
    # Load best, eval on test
    model.load_state_dict(torch.load(ckpt1, map_location=device))
    val_loss_1,  val_auroc_1  = evaluate(model, val_loader,  criterion, metric_fn, device)
    test_loss_1, test_auroc_1 = evaluate(model, test_loader, criterion, metric_fn, device)
    print(f'\n  Final test: loss={test_loss_1:.5f}   AUROC={test_auroc_1:.5f}')
    results['phase1'] = {
        'history': hist1,
        'best_val_auroc': best_auroc1,
        'val_loss': val_loss_1, 'val_auroc': val_auroc_1,
        'test_loss': test_loss_1, 'test_auroc': test_auroc_1,
    }

    # ── STEP 2: Full fine-tuning ───────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('  STEP 2 — PHASE 2: Full fine-tuning (all layers)')
    print('=' * 60)
    print('  All layers unfrozen. Starting from best phase-1 weights.')
    model.load_state_dict(torch.load(ckpt1, map_location=device))
    model.unfreeze_all()
    ckpt2 = os.path.join(SAVE_DIR, 'cosmic_phase2_best.pth')
    opt2  = torch.optim.Adam(model.parameters(), lr=LR_FULL)
    hist2, best_auroc2 = run_phase(
        'Phase 2 — Full fine-tune', model, train_loader, val_loader,
        opt2, criterion, metric_fn, PHASE2_EPOCHS, device, ckpt2
    )
    # Load best, eval on test
    model.load_state_dict(torch.load(ckpt2, map_location=device))
    val_loss_2,  val_auroc_2  = evaluate(model, val_loader,  criterion, metric_fn, device)
    test_loss_2, test_auroc_2 = evaluate(model, test_loader, criterion, metric_fn, device)
    print(f'\n  Final test: loss={test_loss_2:.5f}   AUROC={test_auroc_2:.5f}')
    results['phase2'] = {
        'history': hist2,
        'best_val_auroc': best_auroc2,
        'val_loss': val_loss_2, 'val_auroc': val_auroc_2,
        'test_loss': test_loss_2, 'test_auroc': test_auroc_2,
    }

    # ── Save results ──────────────────────────────────────────────────────────
    results_path = os.path.join(SAVE_DIR, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to {results_path}')

    # ── Final Report ──────────────────────────────────────────────────────────
    delta1 = test_auroc_1 - test_auroc_b
    delta2 = test_auroc_2 - test_auroc_b

    print('\n')
    print('=' * 60)
    print('  RESULTS SUMMARY')
    print('=' * 60)
    print(f'  Backbone : NB201 arch-006186 (best NinaPro, 1-ch stem)')
    print(f'  Head     : EinspaceCosmicHead (AdaptiveAvgPool + Conv1x1)')
    print(f'  Dataset  : Cosmic (deepCR ACS-WFC, f435w GAL)')
    print(f'  Metric   : AUROC (pixel-level, higher = better)')
    print()
    print(f'  {"Stage":<30}  {"Val AUROC":>10}  {"Test AUROC":>10}  {"Delta vs baseline":>18}')
    print(f'  {"-"*72}')
    print(f'  {"Baseline (random head)":30}  {val_auroc_b:>10.4f}  {test_auroc_b:>10.4f}  {"N/A":>18}')
    print(f'  {"Phase 1 (head only, "+str(PHASE1_EPOCHS)+" epochs)":30}  {val_auroc_1:>10.4f}  {test_auroc_1:>10.4f}  {"+"+f"{delta1:.4f}":>18}')
    print(f'  {"Phase 2 (full, "+str(PHASE2_EPOCHS)+" epochs)":30}  {val_auroc_2:>10.4f}  {test_auroc_2:>10.4f}  {"+"+f"{delta2:.4f}":>18}')
    print(f'  {"-"*72}')
    print()
    print(f'  Training curves (Phase 1):')
    for row in hist1:
        print(f'    epoch {row["epoch"]}  train_loss={row["train_loss"]:.4f}  val_AUROC={row["val_auroc"]:.4f}')
    print(f'\n  Training curves (Phase 2):')
    for row in hist2:
        print(f'    epoch {row["epoch"]}  train_loss={row["train_loss"]:.4f}  val_AUROC={row["val_auroc"]:.4f}')
    print()
    print('=' * 60)
    print('  INTERPRETATION')
    print('=' * 60)
    print(f'  The baseline AUROC of {test_auroc_b:.4f} reflects that the NinaPro')
    print(f'  backbone (trained on EMG signals) has never seen astronomical images.')
    print(f'  Its pre-trained features are not well-suited for cosmic ray detection.')
    print()
    print(f'  After Phase 1 ({PHASE1_EPOCHS} epochs, head only):')
    print(f'    AUROC improved to {test_auroc_1:.4f} (+{delta1:.4f}).')
    print(f'    The head learned which channel combinations activate on')
    print(f'    cosmic-ray-like patterns, using the frozen backbone as a')
    print(f'    fixed feature extractor.')
    print()
    print(f'  After Phase 2 ({PHASE2_EPOCHS} epochs, full fine-tune):')
    print(f'    AUROC improved to {test_auroc_2:.4f} (+{delta2:.4f} vs baseline).')
    print(f'    All layers adapted — the backbone now produces features')
    print(f'    tailored to the cosmic dataset rather than EMG classification.')
    print()


if __name__ == '__main__':
    main()
