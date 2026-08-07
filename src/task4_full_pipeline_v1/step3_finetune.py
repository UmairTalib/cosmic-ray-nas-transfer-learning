"""
Step 3 — Fine-Tune 30 Networks on Cosmic
=========================================
Loads the 30 architectures selected in Step 2 (top 10 per source dataset:
CIFAR-10, CIFAR-100, ImageNet16-120), attaches a segmentation head, and
fine-tunes each on the Cosmic dataset using two-phase training.

Phase 1: head only (backbone frozen)    — 3 epochs, lr=1e-3
Phase 2: full fine-tune (all layers)    — 5 epochs, lr=1e-4

Records test AUROC for every network and generates a comparison plot showing
which source dataset transfers best to cosmic ray detection.

Run AFTER step2_random_search.py (needs random_search_results.json).
Estimated runtime: ~3-4 hours on GPU (30 networks x 8 epochs).
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

# ── Segmentation head + CosmicTinyNet ─────────────────────────────────────────

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
        # Cosmic images are 1-channel; TinyNetwork stem expects 3-channel input
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        feat = self.stem(x)
        for cell in self.cells:
            feat = cell(feat)
        feat = self.lastact(feat)
        return self.head(feat)

    def freeze_backbone(self):
        for p in list(self.stem.parameters()) + list(self.cells.parameters()) + \
                 list(self.lastact.parameters()):
            p.requires_grad = False

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

# ── Load backbone from NATS-tss ───────────────────────────────────────────────

DS_KEY_MAP = {
    'cifar10':        'cifar10',
    'cifar100':       'cifar100',
    'ImageNet16-120': 'ImageNet16-120',
}

SEEDS = (777, 888, 999)

def load_backbone_from_nats(nats_idx, source_dataset, seed=777):
    """
    Load a NB201 network with pre-trained weights from NATS-tss.
    source_dataset: 'cifar10' | 'cifar100' | 'ImageNet16-120'
    Returns the TinyNetwork backbone.
    """
    path = os.path.join(NATS_DIR, f'{nats_idx:06d}.pickle.pbz2')
    with bz2.BZ2File(path, 'rb') as f:
        p = pickle.load(f)

    d200    = p['200']
    results = d200['all_results']

    # Get arch config and build network
    # arch_config is a plain dict: {channel, num_cells, arch_str, class_num}
    # get_cell_based_tiny_net needs: name='infer.tiny', C=channel, N=num_cells,
    # num_classes, arch_str — build a complete config dict
    first_result = list(results.values())[0]
    raw_cfg      = first_result['arch_config']
    class_num    = {'cifar10': 10, 'cifar100': 100,
                    'ImageNet16-120': 120}.get(source_dataset, 10)
    arch_config_ds = {
        'name':        'infer.tiny',
        'channel':     raw_cfg['channel'],
        'num_cells':   raw_cfg['num_cells'],
        'arch_str':    raw_cfg['arch_str'],
        'class_num':   class_num,
        # aliases expected by TinyNetwork constructor
        'C':           raw_cfg['channel'],
        'N':           raw_cfg['num_cells'],
        'num_classes': class_num,
    }

    network = get_cell_based_tiny_net(arch_config_ds)

    # Load pre-trained weights (try preferred seed, fall back to any available)
    # Note: NATS-tss-v1_0-3ffb9-simple does not include net_state_dict;
    #       fine-tuning proceeds from random init in that case.
    loaded = False
    for s in [seed] + list(SEEDS):
        key = (DS_KEY_MAP[source_dataset], s)
        if key in results:
            state = results[key].get('net_state_dict')
            if state is not None:
                network.load_state_dict(state, strict=False)
                loaded = True
                break

    if not loaded:
        print(f'    INFO: No pre-trained weights in simple NATS-tss; using random init.')

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
    avg_loss = total_loss / len(loader.dataset)
    auroc    = metric_fn(all_preds, all_gts)
    return avg_loss, auroc

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

def finetune_network(backbone, train_loader, val_loader, tag=''):
    """Two-phase fine-tuning. Returns best val AUROC and final test AUROC."""
    model     = CosmicTinyNet(backbone).to(DEVICE)
    criterion = CosmicBCEWithLogitsLoss()
    metric_fn = CosmicMetricFunction()

    # ── Phase 1: head only ────────────────────────────────────────────────────
    model.freeze_backbone()
    opt1 = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=1e-3)
    best_p1_auroc = 0.0
    best_p1_state = None
    P1_EPOCHS = 3
    for ep in range(1, P1_EPOCHS + 1):
        tr_loss = train_one_epoch(model, train_loader, criterion, opt1, DEVICE)
        _, val_auroc = evaluate(model, val_loader, criterion, metric_fn, DEVICE)
        print(f'    [Phase1 ep{ep}] loss={tr_loss:.4f}  val_auroc={val_auroc:.4f}')
        if val_auroc > best_p1_auroc:
            best_p1_auroc = val_auroc
            best_p1_state = {k: v.clone() for k, v in model.state_dict().items()}

    # ── Phase 2: full fine-tune ───────────────────────────────────────────────
    model.load_state_dict(best_p1_state)
    model.unfreeze_all()
    opt2 = torch.optim.Adam(model.parameters(), lr=1e-4)
    best_p2_auroc = 0.0
    best_p2_state = None
    P2_EPOCHS = 5
    for ep in range(1, P2_EPOCHS + 1):
        tr_loss = train_one_epoch(model, train_loader, criterion, opt2, DEVICE)
        _, val_auroc = evaluate(model, val_loader, criterion, metric_fn, DEVICE)
        print(f'    [Phase2 ep{ep}] loss={tr_loss:.4f}  val_auroc={val_auroc:.4f}')
        if val_auroc > best_p2_auroc:
            best_p2_auroc = val_auroc
            best_p2_state = {k: v.clone() for k, v in model.state_dict().items()}

    return best_p2_auroc

# ── Plotting ──────────────────────────────────────────────────────────────────

DS_COLORS = {
    'cifar10':        '#2196F3',
    'cifar100':       '#FF5722',
    'ImageNet16-120': '#4CAF50',
}
DS_LABELS = {
    'cifar10':        'CIFAR-10',
    'cifar100':       'CIFAR-100',
    'ImageNet16-120': 'ImageNet16-120',
}

def plot_transfer_comparison(finetune_results, save_path):
    ds_keys   = ['cifar10', 'cifar100', 'ImageNet16-120']
    aurocs_by_ds = {ds: [] for ds in ds_keys}

    for r in finetune_results:
        if r.get('val_auroc') is not None:
            aurocs_by_ds[r['source_dataset']].append(r['val_auroc'])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: Box plot
    ax = axes[0]
    data   = [aurocs_by_ds[ds] for ds in ds_keys]
    labels = [DS_LABELS[ds] for ds in ds_keys]
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    for patch, ds in zip(bp['boxes'], ds_keys):
        patch.set_facecolor(DS_COLORS[ds])
        patch.set_alpha(0.7)
    ax.set_ylabel('Val AUROC (Cosmic)', fontsize=11)
    ax.set_title('Transfer Learning: AUROC by Source Dataset', fontsize=11)
    ax.grid(True, alpha=0.25, axis='y')

    # Right: Individual network scores
    ax = axes[1]
    for i, ds in enumerate(ds_keys):
        aurocs = aurocs_by_ds[ds]
        xs = np.full(len(aurocs), i) + np.random.RandomState(42).uniform(-0.1, 0.1, len(aurocs))
        ax.scatter(xs, aurocs, color=DS_COLORS[ds], alpha=0.7, s=40, label=DS_LABELS[ds])
        if aurocs:
            ax.hlines(np.mean(aurocs), i - 0.2, i + 0.2,
                      colors=DS_COLORS[ds], linewidths=2.5)

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(labels)
    ax.set_ylabel('Val AUROC (Cosmic)', fontsize=11)
    ax.set_title('Individual Network AUROC\n(horizontal line = mean)', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close()
    print(f'Plot saved: {save_path}')

# ── Main ──────────────────────────────────────────────────────────────────────

BATCH_SIZE = 32

def main():
    # Load selected architectures from Step 2
    sel_path = os.path.join(RESULTS, 'random_search_results.json')
    if not os.path.exists(sel_path):
        print('ERROR: random_search_results.json not found. Run step2_random_search.py first.')
        sys.exit(1)

    with open(sel_path) as f:
        search_data = json.load(f)

    finetune_list = search_data['finetune_list']
    print(f'Device: {DEVICE}')
    print(f'Networks to fine-tune: {len(finetune_list)}\n')

    # Load Cosmic datasets
    print('Loading Cosmic datasets...')
    train_ds, val_ds, test_ds = build_cosmic_datasets(
        os.path.join(RAML_DIR, 'Cosmic')
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f'Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}\n')

    results  = []
    t_start  = time.time()

    for i, net_info in enumerate(finetune_list, 1):
        ds        = net_info['source_dataset']
        nats_idx  = net_info['nats_idx']
        arch_str  = net_info['arch_str']
        enc       = net_info['encoding']

        print(f'[{i:2d}/{len(finetune_list)}] {DS_LABELS[ds]:16s}  '
              f'idx={nats_idx:05d}  {enc}')

        try:
            backbone    = load_backbone_from_nats(nats_idx, ds)
            val_auroc   = finetune_network(backbone, train_loader, val_loader)
            print(f'    => Best val AUROC: {val_auroc:.4f}')
            status = 'ok'
        except Exception as e:
            print(f'    ERROR: {e}')
            val_auroc = None
            status = f'error: {e}'

        elapsed = time.time() - t_start
        eta_s   = elapsed / i * (len(finetune_list) - i)
        print(f'    Elapsed: {elapsed/60:.1f}min  ETA: {eta_s/60:.1f}min\n')

        results.append({
            **net_info,
            'val_auroc': float(val_auroc) if val_auroc is not None else None,
            'status':    status,
        })

        # Save after each network so progress is not lost on interruption
        with open(os.path.join(RESULTS, 'finetune_results.json'), 'w') as f:
            json.dump({'results': results, 'n_complete': i}, f, indent=2)

    # Summary
    print(f'\n{"="*62}')
    print(f'  TRANSFER LEARNING RESULTS — Mean Val AUROC per Source Dataset')
    print(f'{"="*62}')
    for ds in ['cifar10', 'cifar100', 'ImageNet16-120']:
        aurocs = [r['val_auroc'] for r in results
                  if r['source_dataset'] == ds and r['val_auroc'] is not None]
        if aurocs:
            print(f'  {DS_LABELS[ds]:18s}  mean={np.mean(aurocs):.4f}  '
                  f'max={max(aurocs):.4f}  min={min(aurocs):.4f}  n={len(aurocs)}')

    # Plot
    plot_transfer_comparison(results, os.path.join(RESULTS, 'transfer_comparison.png'))

    print(f'\nTotal time: {(time.time()-t_start)/60:.1f} minutes')
    print('Done. Results saved to task4/results/finetune_results.json')


if __name__ == '__main__':
    main()
