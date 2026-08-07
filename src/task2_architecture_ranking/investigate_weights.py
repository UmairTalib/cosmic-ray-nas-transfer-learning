"""
TODO 2 - Investigate all weight files to find the best architecture per dataset.

What this script does
---------------------
  1. Scans every .pickle.pbz2 file (27 files, CIFAR-10 / CIFAR-100 / ImageNet16-120)
     and reads the final-epoch test accuracy for every (architecture, dataset, seed).

  2. Scans every .pth file (20 files, DarcyFlow / NinaPro) and reads the final-epoch
     validation metric:
       - DarcyFlow  : valid_losses  (lower  = better)
       - NinaPro    : valid_acc1es  (higher = better)

  3. For each dataset prints a ranked table and highlights the best architecture.

Key metric details
------------------
  Pickle  : params['200']['all_results'][(dataset, seed)]['eval_acc1es']['ori-test@199']
  PTH     : data[dataset_key]['valid_losses' | 'valid_acc1es']['ori-test@199']
"""

import os, pickle, bz2, warnings
import torch

warnings.filterwarnings('ignore')

WEIGHT_DIR = os.path.join(os.path.dirname(__file__), '..', 'all', 'all')
FINAL_KEY  = 'ori-test@199'


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mean(vals):
    return sum(vals) / len(vals) if vals else float('nan')


def _print_table(title, rows, col_headers, higher_is_better=True):
    """Print a ranked table.  rows = list of (arch_id, *values).
    Sorted by r[1] – the first metric column (mean for pickle, single value for pth)."""
    rows_sorted = sorted(rows, key=lambda r: r[1], reverse=higher_is_better)
    print(f'\n{"=" * 68}')
    print(f'  {title}')
    print(f'{"=" * 68}')
    hdr = f'  {"Rank":4s}  {"Arch-ID":8s}  ' + '  '.join(f'{h:>10s}' for h in col_headers)
    print(hdr)
    print('  ' + '-' * (len(hdr) - 2))
    for rank, row in enumerate(rows_sorted, 1):
        arch_id = row[0]
        vals    = row[1:]
        flag    = '  <-- BEST' if rank == 1 else ''
        vals_str = '  '.join(f'{v:>10.4f}' for v in vals)
        print(f'  {rank:4d}  {arch_id:8s}  {vals_str}{flag}')
    print()


# ── 1. Pickle files – CIFAR-10 / CIFAR-100 / ImageNet16-120 ──────────────────

print('\n' + '#' * 68)
print('# PART 1: Pickle files (.pickle.pbz2)')
print('# Datasets: cifar10, cifar10-valid, cifar100, ImageNet16-120')
print('# Metric: eval_acc1es[ori-test@199]  -- higher is better (% accuracy)')
print('#' * 68)

pickle_files = sorted(f for f in os.listdir(WEIGHT_DIR) if f.endswith('.pbz2'))
print(f'\nFound {len(pickle_files)} pickle files.\n')

# Accumulate per-arch, per-dataset scores
# Structure: acc_data[dataset][arch_id] = [seed1_acc, seed2_acc, ...]
acc_data = {}

for fname in pickle_files:
    arch_id = fname.replace('.pickle.pbz2', '')  # e.g. '001459'
    path    = os.path.join(WEIGHT_DIR, fname)

    with bz2.BZ2File(path, 'rb') as f:
        params = pickle.load(f)

    results = params['200']['all_results']

    for (dataset, seed), entry in results.items():
        accs = entry.get('eval_acc1es', {})
        if FINAL_KEY not in accs:
            # Fall back to last available key
            all_keys = sorted(accs.keys(), key=lambda k: int(k.split('@')[1]))
            if not all_keys:
                continue
            val = accs[all_keys[-1]]
        else:
            val = accs[FINAL_KEY]

        if dataset not in acc_data:
            acc_data[dataset] = {}
        if arch_id not in acc_data[dataset]:
            acc_data[dataset][arch_id] = []
        acc_data[dataset][arch_id].append(float(val))

# Print one table per dataset
for dataset in ['cifar10', 'cifar10-valid', 'cifar100', 'ImageNet16-120']:
    if dataset not in acc_data:
        continue
    rows = []
    for arch_id, seed_accs in acc_data[dataset].items():
        mean_acc = _mean(seed_accs)
        max_acc  = max(seed_accs)
        rows.append((arch_id, mean_acc, max_acc, len(seed_accs)))

    _print_table(
        f'Dataset: {dataset}  (sorted by mean accuracy, higher = better)',
        [(r[0], r[1], r[2]) for r in rows],
        col_headers=['mean_acc%', 'best_acc%'],
        higher_is_better=True
    )

    # Print the winner clearly
    best = max(rows, key=lambda r: r[1])
    print(f'  --> BEST for {dataset}: arch {best[0]}')
    print(f'      mean={best[1]:.4f}%  best_seed={best[2]:.4f}%  n_seeds={best[3]}')


# ── 2. PTH files – DarcyFlow / NinaPro ───────────────────────────────────────

print('\n\n' + '#' * 68)
print('# PART 2: PTH files (.pth)')
print('# DarcyFlow  : valid_losses[ori-test@199]   -- lower  is better')
print('# NinaPro    : valid_acc1es[ori-test@199]   -- higher is better')
print('#' * 68)

pth_files = sorted(f for f in os.listdir(WEIGHT_DIR) if f.endswith('.pth'))
print(f'\nFound {len(pth_files)} pth files.\n')

# Accumulate per-arch, per-dataset
# Structure: pth_data[dataset][arch_id] = float_metric
pth_scores = {}

for fname in pth_files:
    path = os.path.join(WEIGHT_DIR, fname)
    data = torch.load(path, map_location='cpu')
    ds   = data['all_dataset_keys'][0]

    # Extract arch_id from filename: arch-003391-seed-0777.pth -> 003391
    arch_id = fname.split('-')[1]

    inner = data[ds]

    if ds == 'darcyflow':
        val = float(inner['valid_losses'][FINAL_KEY])
        higher_better = False
    else:  # ninapro
        val = float(inner['valid_acc1es'][FINAL_KEY])
        higher_better = True

    if ds not in pth_scores:
        pth_scores[ds] = {}
    pth_scores[ds][arch_id] = val

# Print DarcyFlow table (lower loss = better)
if 'darcyflow' in pth_scores:
    rows = [(arch_id, val) for arch_id, val in pth_scores['darcyflow'].items()]
    rows_sorted = sorted(rows, key=lambda r: r[1])  # ascending for loss

    print(f'{"=" * 68}')
    print(f'  Dataset: darcyflow  (sorted by valid_loss, lower = better)')
    print(f'{"=" * 68}')
    print(f'  {"Rank":4s}  {"Arch-ID":8s}  {"valid_loss":>12s}')
    print('  ' + '-' * 32)
    for rank, (arch_id, val) in enumerate(rows_sorted, 1):
        flag = '  <-- BEST' if rank == 1 else ''
        print(f'  {rank:4d}  {arch_id:8s}  {val:>12.6f}{flag}')
    print()

    best = rows_sorted[0]
    print(f'  --> BEST for darcyflow: arch {best[0]}  (valid_loss={best[1]:.6f})')

# Print NinaPro table (higher acc = better)
if 'ninapro' in pth_scores:
    rows = [(arch_id, val) for arch_id, val in pth_scores['ninapro'].items()]
    rows_sorted = sorted(rows, key=lambda r: r[1], reverse=True)  # descending for acc

    print(f'\n{"=" * 68}')
    print(f'  Dataset: ninapro  (sorted by valid_acc%, higher = better)')
    print(f'{"=" * 68}')
    print(f'  {"Rank":4s}  {"Arch-ID":8s}  {"valid_acc%":>12s}')
    print('  ' + '-' * 32)
    for rank, (arch_id, val) in enumerate(rows_sorted, 1):
        flag = '  <-- BEST' if rank == 1 else ''
        print(f'  {rank:4d}  {arch_id:8s}  {val:>12.4f}{flag}')
    print()

    best = rows_sorted[0]
    print(f'  --> BEST for ninapro: arch {best[0]}  (valid_acc={best[1]:.4f}%)')


# ── 3. Summary ─────────────────────────────────────────────────────────────────

print('\n\n' + '#' * 68)
print('# SUMMARY: Best architecture per dataset')
print('#' * 68)
print()

# Pickle datasets
for dataset in ['cifar10', 'cifar10-valid', 'cifar100', 'ImageNet16-120']:
    if dataset not in acc_data:
        continue
    rows = [(arch_id, _mean(accs), max(accs)) for arch_id, accs in acc_data[dataset].items()]
    best = max(rows, key=lambda r: r[1])
    print(f'  {dataset:20s}  best_arch={best[0]}  mean_acc={best[1]:.4f}%  best_seed={best[2]:.4f}%')

# PTH datasets
if 'darcyflow' in pth_scores:
    best_id = min(pth_scores['darcyflow'], key=pth_scores['darcyflow'].get)
    print(f'  {"darcyflow":20s}  best_arch={best_id}  valid_loss={pth_scores["darcyflow"][best_id]:.6f}')

if 'ninapro' in pth_scores:
    best_id = max(pth_scores['ninapro'], key=pth_scores['ninapro'].get)
    print(f'  {"ninapro":20s}  best_arch={best_id}  valid_acc={pth_scores["ninapro"][best_id]:.4f}%')

print()
