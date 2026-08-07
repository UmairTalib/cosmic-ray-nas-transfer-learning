"""
Step 2 — Random Search + Architecture Selection
================================================
1. Samples 300 random architectures from the NB201 search space (15,625 total)
2. Ranks them with the Step 1 predictor (SVR ensemble trained on 1024 archs)
3. Selects the top 10 per source dataset (CIFAR-10, CIFAR-100, ImageNet16-120)
   => 30 architectures to fine-tune in Step 3

Selection uses the rank-average SVR ensemble (our target-meeting predictor).
A single large-budget SVR is used only to display a human-readable predicted
accuracy alongside each architecture.

Each selected architecture is mapped back to its NATS-tss file index so that
Step 3 can load the matching backbone.

Run AFTER step1_predictor.py.
"""

import os, sys, json, pickle, warnings
warnings.filterwarnings('ignore')

import numpy as np
from scipy.stats import kendalltau as scipy_kendalltau

RAML_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from step1_predictor import (load_cache, encode_arch, OPS, N_EDGES, N_OPS,
                             parse_arch_str, make_predictor, make_svr_large)

# NATS-tss file index -> arch_str mapping (built in task4). Lets us recover the
# NATS file index for any architecture so Step 3 loads the right backbone.
NATS_STRS_PATH = os.path.join(RAML_DIR, 'task4', 'results',
                              'nb201_cache_arch_strs.pkl')

def load_nats_index():
    with open(NATS_STRS_PATH, 'rb') as f:
        nats_strs = pickle.load(f)          # list: nats_idx -> arch_str
    return {s: i for i, s in enumerate(nats_strs) if s}

def ops_to_arch_str(ops):
    return (f'|{ops[0]}~0|+|{ops[1]}~0|{ops[2]}~1|'
            f'+|{ops[3]}~0|{ops[4]}~1|{ops[5]}~2|')

def arch_label(arch_str):
    abbrev = {'none':'N','skip_connect':'S','nor_conv_1x1':'1',
              'nor_conv_3x3':'3','avg_pool_3x3':'P'}
    return '-'.join(abbrev.get(op, op) for op in parse_arch_str(arch_str))

TRAIN_P  = 1024
N_SAMPLE = 300
N_TOP    = 10

DS_CONFIG = [
    ('cifar10',        'CIFAR-10'),
    ('cifar100',       'CIFAR-100'),
    ('ImageNet16-120', 'ImageNet16-120'),
]

def train_models(X_all, y_all, p=TRAIN_P, seed=42):
    """Fit the ranking ensemble and an accuracy-scaled SVR on p archs."""
    rng   = np.random.RandomState(seed)
    valid = np.where(~np.isnan(y_all))[0]
    idx   = rng.choice(valid, size=min(p, len(valid)), replace=False)
    ranker = make_predictor().fit(X_all[idx], y_all[idx].astype(np.float64))
    accpred = make_svr_large().fit(X_all[idx], y_all[idx].astype(np.float64))
    return ranker, accpred

def main():
    encodings, c10_acc, c100_acc, inet_acc, arch_strs = load_cache()
    nats_map = load_nats_index()
    acc_map = {'cifar10': c10_acc, 'cifar100': c100_acc,
               'ImageNet16-120': inet_acc}

    print(f'\nSampling {N_SAMPLE} random architectures from NB201 ...')
    rng         = np.random.RandomState(99)
    sample_idx  = rng.choice(len(arch_strs), size=N_SAMPLE, replace=False)
    sample_encs = encodings[sample_idx]
    print('Done.')

    all_results  = {}
    selected_ids = {}

    for ds_key, ds_label in DS_CONFIG:
        y_all = acc_map[ds_key]
        print(f'\n{"="*58}\n  {ds_label}\n{"="*58}')

        ranker, accpred = train_models(encodings, y_all, p=TRAIN_P)
        rank_scores = ranker.predict(sample_encs)          # ensemble rank score
        acc_scores  = accpred.predict(sample_encs)         # predicted accuracy %
        true_accs   = [float(y_all[i]) if not np.isnan(y_all[i]) else None
                       for i in sample_idx]

        ranked = sorted(
            zip(sample_idx, [arch_strs[i] for i in sample_idx],
                rank_scores, acc_scores, true_accs),
            key=lambda x: x[2], reverse=True)
        top10 = ranked[:N_TOP]

        print(f'  Top {N_TOP} by predicted {ds_label} accuracy:')
        print(f'  {"#":3s}  {"nats_idx":8s}  {"PredAcc%":9s}  {"True%":8s}  Encoding')
        print(f'  {"-"*58}')
        top_records = []
        for rank, (samp_i, arch_str, _, pacc, tacc) in enumerate(top10, 1):
            nats_idx = nats_map.get(arch_str, -1)
            true_s   = f'{tacc:.3f}' if tacc is not None else '  N/A '
            print(f'  {rank:3d}  {nats_idx:8d}  {pacc:9.3f}  {true_s:8s}  '
                  f'{arch_label(arch_str)}')
            top_records.append({
                'rank': rank, 'nats_idx': int(nats_idx), 'arch_str': arch_str,
                'encoding': arch_label(arch_str), 'predicted_acc': float(pacc),
                'true_acc': tacc,
            })

        valid_pairs = [(rk, t) for _, _, rk, _, t in ranked if t is not None]
        if len(valid_pairs) >= 10:
            tau, _ = scipy_kendalltau([x[1] for x in valid_pairs],
                                      [x[0] for x in valid_pairs])
            print(f'\n  Kendall Tau on {len(valid_pairs)} ranked samples: {tau:.4f}')

        all_results[ds_key]  = {'ds_label': ds_label, 'train_size': TRAIN_P,
                                'n_sampled': N_SAMPLE, 'top10': top_records}
        selected_ids[ds_key] = [(r['nats_idx'], r['arch_str']) for r in top_records]

    print(f'\n{"="*58}\n  ARCHITECTURES FOR FINE-TUNING (Step 3)\n{"="*58}')
    finetune_list = []
    for ds_key, ds_label in DS_CONFIG:
        print(f'  {ds_label}:')
        for nats_idx, arch_str in selected_ids[ds_key]:
            print(f'    nats_idx={nats_idx:05d}  {arch_label(arch_str)}')
            finetune_list.append({'source_dataset': ds_key, 'nats_idx': nats_idx,
                                  'arch_str': arch_str,
                                  'encoding': arch_label(arch_str)})
        print()

    out_path = os.path.join(RESULTS, 'random_search_results.json')
    with open(out_path, 'w') as f:
        json.dump({'n_sampled': N_SAMPLE, 'n_top_per_ds': N_TOP,
                   'train_size': TRAIN_P, 'results': all_results,
                   'finetune_list': finetune_list}, f, indent=2)
    print(f'Saved: {out_path}')
    print('Run step3_finetune.py next.')


if __name__ == '__main__':
    main()
