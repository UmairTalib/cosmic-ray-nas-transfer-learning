"""
Fine-tune a pre-trained NAS-Bench-201 TinyNetwork on the Cosmic dataset.

Architecture
------------
  backbone  : NinaPro best arch (006186), already 1-channel stem, format=new
              Runs all 17 cells + lastact -> (batch, 64, 32, 32) spatial features
  head      : EinspaceCosmicHead(in_channels=64, output_shape=(128, 128))
              AdaptiveAvgPool2d(128x128) + Conv2d(64, 1, 1x1)
              -> (batch, 1, 128, 128)

Two-phase training
------------------
  Phase 1 (head only): freeze backbone weights, train only the head
  Phase 2 (full):      unfreeze everything, fine-tune end-to-end at lower LR

Loss   : CosmicBCEWithLogitsLoss  (BCEWithLogits, ignore pixels where mask < 0)
Metric : AUROC (pixel-wise, higher = better)

Usage
-----
  python task3/finetune_cosmic.py
  python task3/finetune_cosmic.py --cosmic_root Cosmic --phase1_epochs 5 --phase2_epochs 10
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ── Path setup ──────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'task1'))
sys.path.insert(0, os.path.join(ROOT, 'task3'))

from load_pth import load_pth_network
from cosmic_data import (
    build_cosmic_datasets, build_dir_lists,
    CosmicBCEWithLogitsLoss, CosmicMetricFunction,
)


# ── EinspaceCosmicHead (as given by the professor) ────────────────────────────
class EinspaceCosmicHead(nn.Module):
    def __init__(self, in_channels: int, output_shape: tuple = (128, 128)):
        super().__init__()
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_shape),
            nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# ── CosmicTinyNet: backbone wrapper ───────────────────────────────────────────
class CosmicTinyNet(nn.Module):
    """
    Wraps a TinyNetwork (or LegacyTinyNetwork) backbone for pixel segmentation.

    The standard TinyNetwork ends with GlobalAvgPool -> Linear (classification).
    Here we strip those final layers and attach EinspaceCosmicHead instead,
    so the output is a (batch, 1, 128, 128) logit map.

    Forward path:
      x  (batch, 1, 128, 128)
      -> stem              (batch, 16, 128, 128)
      -> cells[0..16]      (batch, 64,  32,  32)   all 17 cells
      -> lastact (BN+ReLU) (batch, 64,  32,  32)
      -> EinspaceCosmicHead
           AdaptiveAvgPool2d(128,128)  (batch, 64, 128, 128)
           Conv2d(64, 1, 1x1)         (batch,  1, 128, 128)
    """

    def __init__(self, backbone):
        super().__init__()
        self.stem    = backbone.stem
        self.cells   = backbone.cells
        self.lastact = backbone.lastact
        # Infer backbone output channels: always 4*C = 64 for standard NB201
        C_out = backbone.classifier.in_features  # Linear's input size = 64
        self.head = EinspaceCosmicHead(in_channels=C_out, output_shape=(128, 128))

    def forward(self, x):
        feat = self.stem(x)
        for cell in self.cells:
            feat = cell(feat)
        feat = self.lastact(feat)   # (batch, 64, 32, 32) for 128x128 input
        return self.head(feat)      # (batch, 1, 128, 128)

    def freeze_backbone(self):
        for p in (list(self.stem.parameters()) +
                  list(self.cells.parameters()) +
                  list(self.lastact.parameters())):
            p.requires_grad = False

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

    def trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]


# ── Training helpers ──────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n_batches  = 0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        preds = model(images)              # (batch, 1, 128, 128)
        loss  = criterion(preds, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches  += 1
    return total_loss / max(n_batches, 1)


def eval_epoch(model, loader, metric_fn, device):
    model.eval()
    all_preds = []
    all_masks = []
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            preds  = model(images)
            # Sigmoid converts logits to [0,1] probabilities for AUROC
            all_preds.append(torch.sigmoid(preds).cpu().numpy())
            all_masks.append(masks.numpy())
    try:
        return metric_fn(all_preds, all_masks)
    except Exception as e:
        print(f'    [metric error: {e}]')
        return float('nan')


def run_phase(label, model, train_loader, val_loader, optimizer, criterion,
              metric_fn, n_epochs, device, save_path):
    print(f'\n{"="*60}')
    print(f'  {label}')
    print(f'{"="*60}')
    best_auroc  = -1.0
    best_epoch  = 0

    for epoch in range(1, n_epochs + 1):
        t0         = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_auroc  = eval_epoch(model, val_loader, metric_fn, device)
        elapsed    = time.time() - t0

        improved = val_auroc > best_auroc
        if improved:
            best_auroc = val_auroc
            best_epoch = epoch
            torch.save(model.state_dict(), save_path)

        marker = '  *' if improved else ''
        print(f'  Epoch {epoch:3d}/{n_epochs}  loss={train_loss:.5f}  '
              f'val_AUROC={val_auroc:.5f}  ({elapsed:.1f}s){marker}')

    print(f'\n  Best: epoch {best_epoch}  AUROC={best_auroc:.5f}  -> saved to {save_path}')
    return best_auroc


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Fine-tune NB201 on Cosmic dataset')
    parser.add_argument('--cosmic_root',    default='Cosmic',       help='Path to Cosmic/ folder')
    parser.add_argument('--weights',        default='all/all/arch-006186-seed-0777.pth',
                                            help='Pre-trained .pth backbone')
    parser.add_argument('--phase1_epochs',  type=int, default=5,    help='Head-only epochs')
    parser.add_argument('--phase2_epochs',  type=int, default=10,   help='Full fine-tune epochs')
    parser.add_argument('--batch_size',     type=int, default=16)
    parser.add_argument('--lr_head',        type=float, default=1e-3,  help='LR for head-only phase')
    parser.add_argument('--lr_full',        type=float, default=1e-4,  help='LR for full fine-tune')
    parser.add_argument('--num_workers',    type=int, default=0)
    parser.add_argument('--save_dir',       default='task3',         help='Where to save checkpoints')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── 1. Generate dir lists if needed ──────────────────────────────────────
    train_dirs_path = os.path.join(args.cosmic_root, 'train_dirs.npy')
    test_dirs_path  = os.path.join(args.cosmic_root, 'test_dirs.npy')
    if not os.path.isfile(train_dirs_path) or not os.path.isfile(test_dirs_path):
        print('Generating train_dirs.npy / test_dirs.npy ...')
        build_dir_lists(args.cosmic_root)
    else:
        print(f'Found existing dir lists in {args.cosmic_root}')

    # ── 2. Datasets & loaders ─────────────────────────────────────────────────
    print('\nBuilding datasets ...')
    trainset, valset, testset = build_cosmic_datasets(args.cosmic_root)
    print(f'  train: {len(trainset)}  val: {len(valset)}  test: {len(testset)}')

    train_loader = DataLoader(trainset, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers, drop_last=True)
    val_loader   = DataLoader(valset,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers)
    test_loader  = DataLoader(testset,  batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers)

    # ── 3. Build model ────────────────────────────────────────────────────────
    print(f'\nLoading backbone from: {args.weights}')
    backbone, ds, cfg, fmt, missing = load_pth_network(args.weights)
    print(f'  dataset={ds}  format={fmt}  num_classes={cfg["num_classes"]}  '
          f'missing_keys={len(missing)}')

    model = CosmicTinyNet(backbone).to(device)
    print(f'  CosmicTinyNet built: backbone 64ch -> EinspaceCosmicHead -> (1,128,128)')

    criterion = CosmicBCEWithLogitsLoss()
    metric_fn = CosmicMetricFunction()

    os.makedirs(args.save_dir, exist_ok=True)
    save_phase1 = os.path.join(args.save_dir, 'cosmic_phase1_best.pth')
    save_phase2 = os.path.join(args.save_dir, 'cosmic_phase2_best.pth')

    # ── 4. Phase 1: head only ─────────────────────────────────────────────────
    if args.phase1_epochs > 0:
        model.freeze_backbone()
        n_trainable = sum(p.numel() for p in model.trainable_params())
        print(f'\n  Backbone frozen. Trainable params: {n_trainable:,}')

        optimizer1 = torch.optim.Adam(model.trainable_params(), lr=args.lr_head)
        run_phase('Phase 1 — Head only', model, train_loader, val_loader,
                  optimizer1, criterion, metric_fn, args.phase1_epochs,
                  device, save_phase1)

    # ── 5. Phase 2: full fine-tune ────────────────────────────────────────────
    if args.phase2_epochs > 0:
        model.unfreeze_all()
        # Restore best phase-1 weights before starting phase 2
        if args.phase1_epochs > 0 and os.path.isfile(save_phase1):
            model.load_state_dict(torch.load(save_phase1, map_location=device))
            print('\n  Loaded best phase-1 weights as starting point for phase 2.')
        model.unfreeze_all()
        n_trainable = sum(p.numel() for p in model.trainable_params())
        print(f'  All layers unfrozen. Trainable params: {n_trainable:,}')

        optimizer2 = torch.optim.Adam(model.parameters(), lr=args.lr_full)
        run_phase('Phase 2 — Full fine-tune', model, train_loader, val_loader,
                  optimizer2, criterion, metric_fn, args.phase2_epochs,
                  device, save_phase2)

    # ── 6. Final test evaluation ──────────────────────────────────────────────
    final_ckpt = save_phase2 if args.phase2_epochs > 0 else save_phase1
    if os.path.isfile(final_ckpt):
        model.load_state_dict(torch.load(final_ckpt, map_location=device))
        print(f'\nLoaded best checkpoint: {final_ckpt}')

    test_auroc = eval_epoch(model, test_loader, metric_fn, device)
    print(f'\n{"="*60}')
    print(f'  FINAL TEST AUROC: {test_auroc:.5f}')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
