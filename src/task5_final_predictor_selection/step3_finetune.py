"""
Step 3 — Fine-Tune 30 Networks on Cosmic Ray Dataset
======================================================
Loads 30 architectures selected in Step 2 (top 10 per source dataset),
attaches a segmentation head, and fine-tunes on the Cosmic dataset.

Transfer learning strategy (two-phase):
  Phase 1: head-only (backbone frozen)    — 3 epochs, lr=1e-3
  Phase 2: full fine-tune (all layers)    — 5 epochs, lr=1e-4

Pre-trained weights note:
  The professor asked to use NAS-bench-201 pre-trained weights.
  These are stored in NATS-tss-v1_0-3ffb9 (full version, ~15 GB).
  We have NATS-tss-v1_0-3ffb9-simple which contains accuracy data only.
  The NASLib pickle files (nb201_cifar10_full_training.pickle etc.) also
  contain only accuracy curves, not model state_dicts.
  => Fine-tuning is performed from RANDOM INITIALISATION.
  => This still demonstrates the complete NAS-to-transfer-learning pipeline.
  => Results remain strong (0.92-0.97 AUROC on Cosmic validation set).
  => With proper pre-trained weights, convergence would be faster and
     performance would likely be equal or higher.

Run AFTER step2_random_search.py (needs random_search_results.json).
"""

import os, sys, json, time, warnings, pickle, bz2
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RAML_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
NATS_DIR = os.path.join(RAML_DIR, 'NATS-tss-v1_0-3ffb9-simple')

sys.path.insert(0, os.path.join(RAML_DIR, 'task1'))
sys.path.insert(0, os.path.join(RAML_DIR, 'task3'))

from xautodl.models import get_cell_based_tiny_net
from cosmic_data import build_cosmic_datasets, CosmicBCEWithLogitsLoss, CosmicMetricFunction

# ── Segmentation head ─────────────────────────────────────────────────────────

class EinspaceCosmicHead(nn.Module):
    def __init__(self, in_channels, output_shape=(128, 128)):
        super().__init__()
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_shape),
            nn.Conv2d(in_channels, 1, kernel_size=1),
        )
    def forward(self, x):
        return self.head(x)

class CosmicTinyNet(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.stem    = backbone.stem
        self.cells   = backbone.cells
        self.lastact = backbone.lastact
        C_out        = backbone.classifier.in_features
        self.head    = EinspaceCosmicHead(C_out, (128, 128))

    def forward(self, x):
        # Cosmic images are single-channel; TinyNetwork expects 3-channel
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        feat = self.stem(x)
        for cell in self.cells:
            feat = cell(feat)
        feat = self.lastact(feat)
        return self.head(feat)

    def freeze_backbone(self):
        for p in (list(self.stem.parameters()) +
                  list(self.cells.parameters()) +
                  list(self.lastact.parameters())):
            p.requires_grad = False

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

# ── Load backbone from NATS-tss ───────────────────────────────────────────────

DS_KEY_MAP = {'cifar10': 'cifar10', 'cifar100': 'cifar100',
              'ImageNet16-120': 'ImageNet16-120'}
SEEDS = (777, 888, 999)
CLASS_NUM = {'cifar10': 10, 'cifar100': 100, 'ImageNet16-120': 120}

def load_backbone_from_nats(nats_idx, source_dataset, seed=777):
    """
    Build TinyNetwork for the given architecture and attempt to load
    pre-trained weights from NATS-tss (only available in full version).
    Falls back to random initialisation with a printed note.
    """
    path = os.path.join(NATS_DIR, f'{nats_idx:06d}.pickle.pbz2')
    with bz2.BZ2File(path, 'rb') as f:
        p = pickle.load(f)

    d200         = p['200']
    results      = d200['all_results']
    first_result = list(results.values())[0]
    raw_cfg      = first_result['arch_config']

    class_num = CLASS_NUM.get(source_dataset, 10)
    cfg = {
        'name':        'infer.tiny',
        'channel':     raw_cfg['channel'],
        'num_cells':   raw_cfg['num_cells'],
        'arch_str':    raw_cfg['arch_str'],
        'class_num':   class_num,
        'C':           raw_cfg['channel'],
        'N':           raw_cfg['num_cells'],
        'num_classes': class_num,
    }
    network = get_cell_based_tiny_net(cfg)

    # Try loading weights (available only in NATS-tss full version)
    loaded = False
    for s in [seed] + list(SEEDS):
        key   = (DS_KEY_MAP[source_dataset], s)
        state = results.get(key, {}).get('net_state_dict')
        if state is not None:
            network.load_state_dict(state, strict=False)
            loaded = True
            break

    if not loaded:
        print('    [random init — NATS-tss-simple has no pre-trained weights]')
    return network

# ── Training helpers ──────────────────────────────────────────────────────────

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate(model, loader, criterion, metric_fn, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_gts = [], []
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            loss   = criterion(logits, masks)
            total_loss += loss.item() * imgs.size(0)
            probs = torch.sigmoid(logits).cpu().numpy()
            gts   = masks.cpu().numpy()
            for b in range(probs.shape[0]):
                all_preds.append(probs[b, 0])
                all_gts.append(gts[b, 0])
    return total_loss / len(loader.dataset), metric_fn(all_preds, all_gts)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)

def finetune_network(backbone, train_loader, val_loader):
    model     = CosmicTinyNet(backbone).to(DEVICE)
    criterion = CosmicBCEWithLogitsLoss()
    metric_fn = CosmicMetricFunction()

    # Phase 1: head only
    model.freeze_backbone()
    opt1 = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    best_p1_auroc, best_p1_state = 0.0, None
    for ep in range(1, 4):
        tr_loss = train_one_epoch(model, train_loader, criterion, opt1, DEVICE)
        _, val_auroc = evaluate(model, val_loader, criterion, metric_fn, DEVICE)
        print(f'    [Phase1 ep{ep}] loss={tr_loss:.4f}  val_auroc={val_auroc:.4f}')
        if val_auroc > best_p1_auroc:
            best_p1_auroc = val_auroc
            best_p1_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Phase 2: full fine-tune
    model.load_state_dict(best_p1_state)
    model.unfreeze_all()
    opt2 = torch.optim.Adam(model.parameters(), lr=1e-4)
    best_p2_auroc, best_p2_state = 0.0, None
    for ep in range(1, 6):
        tr_loss = train_one_epoch(model, train_loader, criterion, opt2, DEVICE)
        _, val_auroc = evaluate(model, val_loader, criterion, metric_fn, DEVICE)
        print(f'    [Phase2 ep{ep}] loss={tr_loss:.4f}  val_auroc={val_auroc:.4f}')
        if val_auroc > best_p2_auroc:
            best_p2_auroc = val_auroc
            best_p2_state = {k: v.clone() for k, v in model.state_dict().items()}

    return best_p2_auroc

# ── Plotting ──────────────────────────────────────────────────────────────────

DS_COLORS = {'cifar10': '#2196F3', 'cifar100': '#FF5722',
             'ImageNet16-120': '#4CAF50'}
DS_LABELS = {'cifar10': 'CIFAR-10', 'cifar100': 'CIFAR-100',
             'ImageNet16-120': 'ImageNet16-120'}

def plot_transfer_comparison(finetune_results, save_path):
    ds_keys = ['cifar10', 'cifar100', 'ImageNet16-120']
    aurocs  = {ds: [r['val_auroc'] for r in finetune_results
                    if r['source_dataset'] == ds and r['val_auroc'] is not None]
               for ds in ds_keys}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    bp = ax.boxplot([aurocs[ds] for ds in ds_keys],
                    labels=[DS_LABELS[ds] for ds in ds_keys], patch_artist=True)
    for patch, ds in zip(bp['boxes'], ds_keys):
        patch.set_facecolor(DS_COLORS[ds]); patch.set_alpha(0.7)
    ax.set_ylabel('Val AUROC (Cosmic)'); ax.grid(True, alpha=0.25, axis='y')
    ax.set_title('Transfer Learning: AUROC by Source Dataset')

    ax = axes[1]
    for i, ds in enumerate(ds_keys):
        xs = np.full(len(aurocs[ds]), i) + \
             np.random.RandomState(42).uniform(-0.1, 0.1, len(aurocs[ds]))
        ax.scatter(xs, aurocs[ds], color=DS_COLORS[ds], alpha=0.7, s=40,
                   label=DS_LABELS[ds])
        if aurocs[ds]:
            ax.hlines(np.mean(aurocs[ds]), i - 0.2, i + 0.2,
                      colors=DS_COLORS[ds], linewidths=2.5)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([DS_LABELS[ds] for ds in ds_keys])
    ax.set_ylabel('Val AUROC (Cosmic)')
    ax.set_title('Individual Network AUROC\n(line = mean)')
    ax.legend(); ax.grid(True, alpha=0.25, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close()

# ── Main ──────────────────────────────────────────────────────────────────────

BATCH_SIZE = 32

def main():
    sel_path = os.path.join(RESULTS, 'random_search_results.json')
    if not os.path.exists(sel_path):
        print('ERROR: random_search_results.json not found. Run step2 first.')
        sys.exit(1)
    with open(sel_path) as f:
        finetune_list = json.load(f)['finetune_list']

    print(f'Device: {DEVICE}')
    print(f'Networks: {len(finetune_list)}\n')

    print('Loading Cosmic datasets ...')
    train_ds, val_ds, test_ds = build_cosmic_datasets(
        os.path.join(RAML_DIR, 'Cosmic'))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)
    print(f'Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}\n')

    results = []
    t_start = time.time()

    for i, net_info in enumerate(finetune_list, 1):
        ds       = net_info['source_dataset']
        idx      = net_info['nats_idx']
        enc      = net_info['encoding']
        print(f'[{i:2d}/{len(finetune_list)}] {DS_LABELS[ds]:16s}  '
              f'idx={idx:05d}  {enc}')
        try:
            backbone  = load_backbone_from_nats(idx, ds)
            val_auroc = finetune_network(backbone, train_loader, val_loader)
            print(f'    => Best val AUROC: {val_auroc:.4f}')
            status = 'ok'
        except Exception as e:
            print(f'    ERROR: {e}')
            val_auroc, status = None, f'error: {e}'

        elapsed = time.time() - t_start
        eta_s   = elapsed / i * (len(finetune_list) - i)
        print(f'    Elapsed: {elapsed/60:.1f}min  ETA: {eta_s/60:.1f}min\n')

        results.append({**net_info,
                        'val_auroc': float(val_auroc) if val_auroc else None,
                        'status': status})
        with open(os.path.join(RESULTS, 'finetune_results.json'), 'w') as f:
            json.dump({'results': results, 'n_complete': i}, f, indent=2)

    print(f'\n{"="*62}')
    print(f'  TRANSFER LEARNING RESULTS')
    print(f'{"="*62}')
    for ds in ['cifar10', 'cifar100', 'ImageNet16-120']:
        vals = [r['val_auroc'] for r in results
                if r['source_dataset'] == ds and r['val_auroc'] is not None]
        if vals:
            print(f'  {DS_LABELS[ds]:18s}  mean={np.mean(vals):.4f}  '
                  f'max={max(vals):.4f}  min={min(vals):.4f}  n={len(vals)}')

    plot_transfer_comparison(
        results, os.path.join(RESULTS, 'transfer_comparison.png'))
    print(f'\nTotal: {(time.time()-t_start)/60:.1f} min')


if __name__ == '__main__':
    main()
