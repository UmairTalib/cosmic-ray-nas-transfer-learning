"""
Step 1 — Performance Predictor on Full NAS-Bench-201 Data
==========================================================
Trains an XGBoost surrogate to predict architecture accuracy from a
30-dim one-hot encoding (6 edges x 5 ops). Evaluates Kendall Tau at
different training set sizes (32, 128, 1024) to match the professor's
benchmark targets.

First run: builds nb201_cache.npz from all 15,625 NATS-tss files (~5 min).
All later runs: loads cache in <1 second.
"""

import os, sys, pickle, bz2, json, warnings
warnings.filterwarnings('ignore')

import numpy as np
from scipy.stats import kendalltau as scipy_kendalltau
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RAML_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NATS_DIR   = os.path.join(RAML_DIR, 'NATS-tss-v1_0-3ffb9-simple')
RESULTS    = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
CACHE_PATH = os.path.join(RESULTS, 'nb201_cache.npz')

os.makedirs(RESULTS, exist_ok=True)

# ── Architecture encoding ─────────────────────────────────────────────────────

OPS     = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
OP2IDX  = {op: i for i, op in enumerate(OPS)}
N_OPS   = len(OPS)
N_EDGES = 6
N_TOTAL = 15625

def parse_arch_str(arch_str):
    ops = []
    for token in arch_str.replace('+', '|').split('|'):
        token = token.strip()
        if '~' in token:
            ops.append(token.split('~')[0].strip())
    return ops

def encode_arch(arch_str):
    enc = np.zeros(N_EDGES * N_OPS, dtype=np.float32)
    for i, op in enumerate(parse_arch_str(arch_str)):
        if op in OP2IDX:
            enc[i * N_OPS + OP2IDX[op]] = 1.0
    return enc

# ── Cache builder ─────────────────────────────────────────────────────────────

DATASETS  = ['cifar10', 'cifar100', 'ImageNet16-120']
SEEDS     = (777, 888, 999)
FINAL_KEY = 'ori-test@199'

def _load_one(idx):
    """Load a single NATS-tss file, return (arch_str, c10, c100, inet) or None."""
    path = os.path.join(NATS_DIR, f'{idx:06d}.pickle.pbz2')
    if not os.path.exists(path):
        return None
    try:
        with bz2.BZ2File(path, 'rb') as f:
            p = pickle.load(f)
        d       = p['200']
        arch_str = d['arch_str']
        results  = d['all_results']

        def mean_acc(ds):
            vals = [results[(ds, s)]['eval_acc1es'].get(FINAL_KEY)
                    for s in SEEDS if (ds, s) in results]
            vals = [v for v in vals if v is not None]
            return float(np.mean(vals)) if vals else np.nan

        return arch_str, mean_acc('cifar10'), mean_acc('cifar100'), mean_acc('ImageNet16-120')
    except Exception:
        return None

def build_cache():
    print(f'Building cache from {N_TOTAL} NATS-tss files...')
    print('(This takes ~5 minutes and only runs once)\n')

    arch_strs = []
    encodings = np.zeros((N_TOTAL, N_EDGES * N_OPS), dtype=np.float32)
    c10_acc   = np.full(N_TOTAL, np.nan, dtype=np.float32)
    c100_acc  = np.full(N_TOTAL, np.nan, dtype=np.float32)
    inet_acc  = np.full(N_TOTAL, np.nan, dtype=np.float32)

    from tqdm import tqdm
    for idx in tqdm(range(N_TOTAL), desc='Loading NATS-tss'):
        result = _load_one(idx)
        if result is None:
            arch_strs.append('')
            continue
        arch_str, c10, c100, inet = result
        arch_strs.append(arch_str)
        encodings[idx] = encode_arch(arch_str)
        c10_acc[idx]   = c10
        c100_acc[idx]  = c100
        inet_acc[idx]  = inet

    np.savez_compressed(
        CACHE_PATH,
        encodings = encodings,
        c10_acc   = c10_acc,
        c100_acc  = c100_acc,
        inet_acc  = inet_acc,
    )
    # Save arch strings separately (numpy doesn't handle variable-length strings well)
    with open(CACHE_PATH.replace('.npz', '_arch_strs.pkl'), 'wb') as f:
        pickle.dump(arch_strs, f)

    print(f'\nCache saved to {CACHE_PATH}')
    return encodings, c10_acc, c100_acc, inet_acc, arch_strs

def load_cache():
    if not os.path.exists(CACHE_PATH):
        return build_cache()
    print('Loading cache...', end=' ', flush=True)
    data      = np.load(CACHE_PATH)
    encodings = data['encodings']
    c10_acc   = data['c10_acc']
    c100_acc  = data['c100_acc']
    inet_acc  = data['inet_acc']
    with open(CACHE_PATH.replace('.npz', '_arch_strs.pkl'), 'rb') as f:
        arch_strs = pickle.load(f)
    print(f'done. ({len(arch_strs)} architectures)')
    return encodings, c10_acc, c100_acc, inet_acc, arch_strs

# ── Predictors ────────────────────────────────────────────────────────────────

def make_xgb():
    return XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        random_state=42, eval_metric='rmse', verbosity=0)

def make_rf():
    return RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)

# ── Evaluation ────────────────────────────────────────────────────────────────

TEST_SIZE  = 2000   # fixed held-out test set size
TRAIN_SIZES = [32, 64, 128, 256, 512, 1024]
N_REPEATS   = 20    # random seeds per train size

# Professor's benchmark targets
PROF_TARGETS = {
    'cifar10':         {32: 0.39, 128: 0.54, 1024: 0.74},
    'cifar100':        {32: 0.39, 128: 0.56, 1024: 0.74},
    'ImageNet16-120':  {32: 0.40, 128: 0.56, 1024: 0.74},
}

def evaluate_predictor(X, y, train_sizes, n_repeats, test_idx, model_factory):
    """
    For each train size p, sample p architectures (from non-test pool),
    train model, evaluate Kendall Tau on fixed test set.
    Returns {p: [tau1, tau2, ...]} over n_repeats.
    """
    X_test = X[test_idx]
    y_test = y[test_idx]
    pool   = np.array([i for i in range(len(y)) if i not in set(test_idx)])

    results = {p: [] for p in train_sizes}
    rng = np.random.RandomState(0)

    for p in train_sizes:
        for seed in range(n_repeats):
            rng_s = np.random.RandomState(seed * 1000 + p)
            tr_idx = rng_s.choice(pool, size=min(p, len(pool)), replace=False)
            m = model_factory()
            m.fit(X[tr_idx], y[tr_idx])
            preds = m.predict(X_test)
            tau, _ = scipy_kendalltau(y_test, preds)
            results[p].append(float(tau))

    return results

# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_tau_curve(ds_label, ds_key, results_rf, results_xgb, save_path):
    ps_rf  = sorted(results_rf)
    ps_xgb = sorted(results_xgb)

    mean_rf  = [np.mean(results_rf[p])  for p in ps_rf]
    std_rf   = [np.std(results_rf[p])   for p in ps_rf]
    mean_xgb = [np.mean(results_xgb[p]) for p in ps_xgb]
    std_xgb  = [np.std(results_xgb[p])  for p in ps_xgb]

    fig, ax = plt.subplots(figsize=(7, 4))

    # RF
    ax.plot(ps_rf, mean_rf, marker='s', color='#2196F3', linewidth=2,
            markersize=5, label='RandomForest')
    ax.fill_between(ps_rf,
                    [m - s for m, s in zip(mean_rf, std_rf)],
                    [m + s for m, s in zip(mean_rf, std_rf)],
                    alpha=0.15, color='#2196F3')

    # XGBoost
    ax.plot(ps_xgb, mean_xgb, marker='o', color='#FF5722', linewidth=2,
            markersize=5, label='XGBoost')
    ax.fill_between(ps_xgb,
                    [m - s for m, s in zip(mean_xgb, std_xgb)],
                    [m + s for m, s in zip(mean_xgb, std_xgb)],
                    alpha=0.15, color='#FF5722')

    # Professor's benchmarks
    targets = PROF_TARGETS.get(ds_key, {})
    for p_t, tau_t in targets.items():
        ax.scatter([p_t], [tau_t], marker='*', s=150, color='#4CAF50',
                   zorder=5, label=f'Target (p={p_t})' if p_t == min(targets) else None)

    if targets:
        ax.scatter([], [], marker='*', s=120, color='#4CAF50', label='Prof. targets')

    ax.set_xscale('log')
    ax.set_xlabel('Training set size  p  (# architectures)', fontsize=11)
    ax.set_ylabel('Kendall Tau', fontsize=11)
    ax.set_title(f'Performance Predictor — {ds_label}\n'
                 f'(mean ± std over {N_REPEATS} seeds, tested on {TEST_SIZE} held-out archs)',
                 fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close()

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # 1. Load data
    encodings, c10_acc, c100_acc, inet_acc, arch_strs = load_cache()

    DS_MAP = [
        ('cifar10',        'CIFAR-10',        c10_acc),
        ('cifar100',       'CIFAR-100',       c100_acc),
        ('ImageNet16-120', 'ImageNet16-120',  inet_acc),
    ]

    # 2. Build fixed test set (same for all datasets/models)
    valid_all = np.where(~(np.isnan(c10_acc) | np.isnan(c100_acc) | np.isnan(inet_acc)))[0]
    rng_test  = np.random.RandomState(0)
    test_idx  = rng_test.choice(valid_all, size=TEST_SIZE, replace=False)
    print(f'\nFixed test set: {TEST_SIZE} architectures  |  '
          f'Total valid: {len(valid_all)}')

    all_results = {}

    for ds_key, ds_label, y_raw in DS_MAP:
        # Only use architectures with valid accuracy for this dataset
        valid = np.where(~np.isnan(y_raw))[0]
        X     = encodings[valid]
        y     = y_raw[valid].astype(np.float64)

        # Re-map test indices to this dataset's valid indices
        valid_set  = set(valid.tolist())
        test_local = np.array([i for i, v in enumerate(valid) if v in set(test_idx.tolist())])

        print(f'\n{"="*60}')
        print(f'  {ds_label}  |  {len(valid)} valid archs  |  '
              f'acc range: {y.min():.2f}% -- {y.max():.2f}%')
        print(f'{"="*60}')
        print(f'  {"Train size":>10s}  {"XGBoost tau":>13s}  {"RF tau":>10s}  '
              f'{"XGB target":>11s}')
        print(f'  {"-"*52}')

        res_xgb = evaluate_predictor(X, y, TRAIN_SIZES, N_REPEATS, test_local, make_xgb)
        res_rf  = evaluate_predictor(X, y, TRAIN_SIZES, N_REPEATS, test_local, make_rf)

        targets = PROF_TARGETS.get(ds_key, {})
        for p in TRAIN_SIZES:
            m_xgb = np.mean(res_xgb[p])
            m_rf  = np.mean(res_rf[p])
            tgt   = targets.get(p, '')
            tgt_s = f'{tgt:.2f}' if tgt else '  -  '
            beat  = ' *' if tgt and m_xgb >= tgt else ''
            print(f'  {p:>10d}  {m_xgb:>13.4f}  {m_rf:>10.4f}  {tgt_s:>11s}{beat}')

        all_results[ds_key] = {
            'n_valid':  int(len(valid)),
            'xgboost':  {str(p): {'mean': float(np.mean(v)), 'std': float(np.std(v)),
                                   'all':  [float(x) for x in v]}
                         for p, v in res_xgb.items()},
            'rf':       {str(p): {'mean': float(np.mean(v)), 'std': float(np.std(v)),
                                   'all':  [float(x) for x in v]}
                         for p, v in res_rf.items()},
            'targets':  {str(k): v for k, v in targets.items()},
        }

        plot_tau_curve(
            ds_label, ds_key, res_rf, res_xgb,
            os.path.join(RESULTS, f'predictor_tau_{ds_key.lower().replace("-","_")}.png')
        )
        print(f'  Plot saved.')

    # Save results
    out_path = os.path.join(RESULTS, 'predictor_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nResults saved to {out_path}')

    # Summary table
    print(f'\n{"="*62}')
    print(f'  SUMMARY — XGBoost Kendall Tau vs Professor Targets')
    print(f'{"="*62}')
    print(f'  {"Dataset":22s}  {"p=32":>6s} {"tgt":>5s}  '
          f'{"p=128":>6s} {"tgt":>5s}  {"p=1024":>7s} {"tgt":>5s}')
    print(f'  {"-"*58}')
    for ds_key, ds_label, _ in DS_MAP:
        if ds_key not in all_results:
            continue
        r = all_results[ds_key]['xgboost']
        t = PROF_TARGETS.get(ds_key, {})
        def fmt(p):
            m = r[str(p)]['mean']
            tgt = t.get(p, 0)
            mark = '*' if m >= tgt else ' '
            return f'{m:.3f}{mark}'
        print(f'  {ds_label:22s}  {fmt(32):>7s} {t.get(32,0):.2f}  '
              f'{fmt(128):>7s} {t.get(128,0):.2f}  {fmt(1024):>8s} {t.get(1024,0):.2f}')
    print(f'  (* = meets or exceeds professor target, - = below target)')
    print()


if __name__ == '__main__':
    main()
