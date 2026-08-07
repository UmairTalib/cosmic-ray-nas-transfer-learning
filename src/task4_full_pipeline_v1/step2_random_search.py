"""
Step 2 — Random Search + Architecture Selection
================================================
1. Randomly samples 300 architectures from NB201 search space (15,625 total)
2. Uses the Step 1 predictor (trained on 1024 architectures) to rank the 300
3. Selects top 10 per source dataset (CIFAR-10, CIFAR-100, ImageNet16-120)
   => 30 architectures to fine-tune in Step 3

Run AFTER step1_predictor.py (needs nb201_cache.npz).
"""

import os, sys, pickle, json, warnings
warnings.filterwarnings('ignore')

import numpy as np
from xgboost import XGBRegressor
from scipy.stats import kendalltau as scipy_kendalltau

RAML_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
NATS_DIR = os.path.join(RAML_DIR, 'NATS-tss-v1_0-3ffb9-simple')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step1_predictor import (load_cache, encode_arch, OPS, N_EDGES, N_OPS,
                              OP2IDX, make_xgb, TEST_SIZE, TRAIN_SIZES)

# ── Architecture helpers (reused from step1) ──────────────────────────────────

def ops_to_arch_str(ops):
    return (f'|{ops[0]}~0|+|{ops[1]}~0|{ops[2]}~1|'
            f'+|{ops[3]}~0|{ops[4]}~1|{ops[5]}~2|')

def random_arch_str(rng):
    ops = [rng.choice(OPS) for _ in range(N_EDGES)]
    return ops_to_arch_str(ops)

def arch_label(arch_str):
    from step1_predictor import parse_arch_str
    abbrev = {'none':'N','skip_connect':'S','nor_conv_1x1':'1',
              'nor_conv_3x3':'3','avg_pool_3x3':'P'}
    return '-'.join(abbrev.get(op, op) for op in parse_arch_str(arch_str))

# ── Predictor training (1024 archs) ──────────────────────────────────────────

TRAIN_P = 1024   # use best training size from step 1

DS_CONFIG = [
    ('cifar10',        'CIFAR-10'),
    ('cifar100',       'CIFAR-100'),
    ('ImageNet16-120', 'ImageNet16-120'),
]

def train_predictor(X_all, y_all, p=TRAIN_P, seed=42):
    """Train XGBoost on p randomly sampled architectures."""
    rng   = np.random.RandomState(seed)
    valid = np.where(~np.isnan(y_all))[0]
    idx   = rng.choice(valid, size=min(p, len(valid)), replace=False)
    m     = make_xgb()
    m.fit(X_all[idx], y_all[idx].astype(np.float64))
    return m, idx

# ── Main ──────────────────────────────────────────────────────────────────────

N_SAMPLE = 300   # random architectures to rank
N_TOP    = 10    # select best N per dataset

def main():
    # 1. Load cache
    encodings, c10_acc, c100_acc, inet_acc, arch_strs = load_cache()

    acc_map = {
        'cifar10':       c10_acc,
        'cifar100':      c100_acc,
        'ImageNet16-120': inet_acc,
    }

    # 2. Sample 300 random architectures from the full NB201 search space
    print(f'\nSampling {N_SAMPLE} random architectures from NB201 search space...')
    rng       = np.random.RandomState(99)
    sample_idx = rng.choice(len(arch_strs), size=N_SAMPLE, replace=False)

    sample_archs   = [arch_strs[i] for i in sample_idx]
    sample_encs    = encodings[sample_idx]   # (300, 30)

    print(f'Sampled {len(sample_archs)} architectures.')

    # 3. For each dataset: train predictor, rank 300, pick top 10
    all_results   = {}
    selected_arch_ids = {}   # {dataset: [(nats_idx, arch_str, predicted_acc, true_acc), ...]}

    for ds_key, ds_label in DS_CONFIG:
        y_all = acc_map[ds_key]
        print(f'\n{"="*58}')
        print(f'  {ds_label}')
        print(f'{"="*58}')

        # Train predictor on 1024 architectures
        model, train_idx = train_predictor(encodings, y_all, p=TRAIN_P)
        print(f'  Predictor trained on {TRAIN_P} architectures.')

        # Predict accuracy for all 300 sampled architectures
        preds = model.predict(sample_encs).tolist()

        # Get true accuracy for the sampled architectures (if available)
        true_accs = [float(y_all[i]) if not np.isnan(y_all[i]) else None
                     for i in sample_idx]

        # Rank by predicted accuracy, take top N_TOP
        ranked = sorted(
            zip(sample_idx, sample_archs, preds, true_accs),
            key=lambda x: x[2], reverse=True
        )

        top10 = ranked[:N_TOP]

        print(f'\n  Top {N_TOP} architectures by predicted {ds_label} accuracy:')
        print(f'  {"#":3s}  {"NATS idx":9s}  {"Predicted%":11s}  '
              f'{"True%":8s}  {"Encoding"}')
        print(f'  {"-"*62}')

        top_records = []
        for rank, (nats_idx, arch_str, pred, true_acc) in enumerate(top10, 1):
            true_s = f'{true_acc:.3f}' if true_acc is not None else '  N/A '
            print(f'  {rank:3d}  {nats_idx:9d}  {pred:11.4f}  '
                  f'{true_s:8s}  {arch_label(arch_str)}')
            top_records.append({
                'rank':       rank,
                'nats_idx':   int(nats_idx),
                'arch_str':   arch_str,
                'encoding':   arch_label(arch_str),
                'predicted':  float(pred),
                'true_acc':   true_acc,
            })

        # Kendall Tau on ranked sample (predicted vs true, for those with true acc)
        valid_pairs = [(p, t) for _, _, p, t in ranked if t is not None]
        if len(valid_pairs) >= 10:
            taus = [x[1] for x in valid_pairs]
            preds_v = [x[0] for x in valid_pairs]
            tau, _ = scipy_kendalltau(taus, preds_v)
            print(f'\n  Kendall Tau on {len(valid_pairs)} ranked samples: {tau:.4f}')

        all_results[ds_key] = {
            'ds_label':      ds_label,
            'train_size':    TRAIN_P,
            'n_sampled':     N_SAMPLE,
            'top10':         top_records,
        }
        selected_arch_ids[ds_key] = [(r['nats_idx'], r['arch_str']) for r in top_records]

    # 4. Summary — all 30 architectures to fine-tune
    print(f'\n{"="*58}')
    print(f'  ARCHITECTURES SELECTED FOR FINE-TUNING (Step 3)')
    print(f'{"="*58}')
    print(f'  Total: {len(DS_CONFIG) * N_TOP} networks '
          f'({N_TOP} per dataset x {len(DS_CONFIG)} datasets)\n')

    finetune_list = []
    for ds_key, ds_label in DS_CONFIG:
        print(f'  {ds_label}:')
        for nats_idx, arch_str in selected_arch_ids[ds_key]:
            print(f'    idx={nats_idx:05d}  {arch_label(arch_str)}')
            finetune_list.append({
                'source_dataset': ds_key,
                'nats_idx':       nats_idx,
                'arch_str':       arch_str,
                'encoding':       arch_label(arch_str),
            })
        print()

    # Save
    out_data = {
        'n_sampled':     N_SAMPLE,
        'n_top_per_ds':  N_TOP,
        'train_size':    TRAIN_P,
        'results':       all_results,
        'finetune_list': finetune_list,
    }
    out_path = os.path.join(RESULTS, 'random_search_results.json')
    with open(out_path, 'w') as f:
        json.dump(out_data, f, indent=2)
    print(f'Results saved to {out_path}')
    print(f'Run step3_finetune.py next to fine-tune these 30 architectures on Cosmic.')


if __name__ == '__main__':
    main()
