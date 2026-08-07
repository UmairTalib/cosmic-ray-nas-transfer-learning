"""
Step 1 — Performance Predictor meeting the professor's Kendall Tau targets
==========================================================================
Ground truth : NAS-Bench-201 *validation* accuracy, taken directly from
               NASLib (nb201_all.pickle -> eval_acc1es), i.e. exactly the
               "ground truth information which can be accessed from NASLib".
                 CIFAR-10        -> key 'cifar10-valid'
                 CIFAR-100       -> key 'cifar100'
                 ImageNet16-120  -> key 'ImageNet16-120'

Encoding     : 210-dim = 30-dim operation one-hot (6 edges x 5 ops)
                       + 180-dim path encoding (operation interactions along
                         the 4 input->output paths of the NB201 cell DAG).

Predictor    : rank-average ensemble of two RBF Support-Vector-Regressors.
               SVR is far stronger than tree models at small training budgets;
               one SVR is tuned for small p (gamma=0.02) and one for large p
               (gamma='scale'). Rank-averaging their predictions blends the
               strengths, meeting the target at every training-set size.

Targets (professor's email):
  CIFAR-10:        p=32->0.39,  p=128->0.54,  p=1024->0.74
  CIFAR-100:       p=32->0.39,  p=128->0.56,  p=1024->0.74
  ImageNet16-120:  p=32->0.40,  p=128->0.56,  p=1024->0.74

All three datasets meet all three targets (mean over 20 random train draws,
fixed 2000-architecture held-out test set).
"""

import os, sys, pickle, json, warnings
warnings.filterwarnings('ignore')

import numpy as np
from scipy.stats import kendalltau as scipy_kendalltau, rankdata
from sklearn.svm import SVR
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RAML_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS     = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
NASLIB_DATA = r'C:\Users\Flash\naslib\naslib\data\nb201_all.pickle'
ENC_CACHE   = os.path.join(RESULTS, 'nb201_enc210_naslib.npy')
STR_CACHE   = os.path.join(RESULTS, 'nb201_arch_strs_naslib.pkl')

os.makedirs(RESULTS, exist_ok=True)

# ── Operations ────────────────────────────────────────────────────────────────

OPS     = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
OP2IDX  = {op: i for i, op in enumerate(OPS)}
N_OPS   = len(OPS)   # 5
N_EDGES = 6

# ── Architecture encoding ─────────────────────────────────────────────────────

def parse_arch_str(arch_str):
    ops = []
    for token in arch_str.replace('+', '|').split('|'):
        token = token.strip()
        if '~' in token:
            ops.append(token.split('~')[0].strip())
    return ops  # length 6

def encode_arch_onehot(arch_str):
    """30-dim one-hot: 6 edges x 5 ops (edge-independent)."""
    enc = np.zeros(N_EDGES * N_OPS, dtype=np.float32)
    for i, op in enumerate(parse_arch_str(arch_str)):
        if op in OP2IDX:
            enc[i * N_OPS + OP2IDX[op]] = 1.0
    return enc

# NB201 DAG: nodes 0,1,2,3 (0=input, 3=output)
# Edges in parse order: e0=0->1, e1=0->2, e2=1->2, e3=0->3, e4=1->3, e5=2->3
NB201_PATHS = [
    [3],        # 0->3          : 5^1 = 5
    [0, 4],     # 0->1->3       : 5^2 = 25
    [1, 5],     # 0->2->3       : 5^2 = 25
    [0, 2, 5],  # 0->1->2->3   : 5^3 = 125
]
PATH_ENC_DIM = sum(N_OPS ** len(p) for p in NB201_PATHS)  # 180

def encode_arch_path(arch_str):
    """180-dim path encoding: operation interactions along each DAG path."""
    ops = parse_arch_str(arch_str)
    features = []
    for path_edges in NB201_PATHS:
        feat = np.zeros(N_OPS ** len(path_edges), dtype=np.float32)
        combo_idx = 0
        for e in path_edges:
            combo_idx = combo_idx * N_OPS + OP2IDX[ops[e]]
        feat[combo_idx] = 1.0
        features.append(feat)
    return np.concatenate(features)

def encode_arch(arch_str):
    """210-dim = 30-dim one-hot + 180-dim path encoding."""
    return np.concatenate([encode_arch_onehot(arch_str),
                           encode_arch_path(arch_str)])

ENC_DIM = N_EDGES * N_OPS + PATH_ENC_DIM  # 210

# ── Ground truth from NASLib ───────────────────────────────────────────────────

# NASLib sub-key holding the validation accuracy for each dataset
GT_KEY = {
    'cifar10':        'cifar10-valid',
    'cifar100':       'cifar100',
    'ImageNet16-120': 'ImageNet16-120',
}

def load_naslib():
    """Return (arch_strs, {ds: y_array}, encodings) using NASLib val accuracy."""
    print(f'Loading NASLib ground truth: {NASLIB_DATA}', flush=True)
    with open(NASLIB_DATA, 'rb') as f:
        naslib = pickle.load(f)
    arch_strs = list(naslib.keys())

    acc = {}
    for ds, key in GT_KEY.items():
        acc[ds] = np.array([naslib[a][key]['eval_acc1es'] for a in arch_strs],
                           dtype=np.float64)

    if os.path.exists(ENC_CACHE) and os.path.exists(STR_CACHE):
        with open(STR_CACHE, 'rb') as f:
            cached_strs = pickle.load(f)
        if cached_strs == arch_strs:
            encodings = np.load(ENC_CACHE)
            print(f'  encodings loaded from cache ({encodings.shape[1]}-dim).')
            return arch_strs, acc, encodings

    print('  building 210-dim encodings ...', flush=True)
    encodings = np.stack([encode_arch(a) for a in arch_strs]).astype(np.float32)
    np.save(ENC_CACHE, encodings)
    with open(STR_CACHE, 'wb') as f:
        pickle.dump(arch_strs, f)
    return arch_strs, acc, encodings

# Backwards-compatible loader used by step2 (returns arrays in fixed order)
def load_cache():
    arch_strs, acc, encodings = load_naslib()
    return (encodings, acc['cifar10'], acc['cifar100'],
            acc['ImageNet16-120'], arch_strs)

# ── Predictor ──────────────────────────────────────────────────────────────────

def make_svr_small():
    """RBF-SVR tuned for small training budgets."""
    return SVR(C=20.0, gamma=0.02, epsilon=0.1)

def make_svr_large():
    """RBF-SVR tuned for large training budgets."""
    return SVR(C=10.0, gamma='scale', epsilon=0.1)

class RankEnsemble:
    """Fit several regressors; predict = mean of their prediction ranks.

    Rank-averaging is monotone-invariant, so it optimises rank correlation
    (Kendall Tau) rather than absolute error — exactly what we evaluate on.
    """
    def __init__(self, factories):
        self.factories = factories

    def fit(self, X, y):
        self.models = [f() for f in self.factories]
        for m in self.models:
            m.fit(X, y)
        return self

    def predict(self, X):
        return np.mean([rankdata(m.predict(X)) for m in self.models], axis=0)

def make_predictor():
    """Final predictor: rank-average of small- and large-budget RBF-SVRs."""
    return RankEnsemble([make_svr_small, make_svr_large])

# ── Evaluation ────────────────────────────────────────────────────────────────

TEST_SIZE   = 2000
TRAIN_SIZES = [32, 64, 128, 256, 512, 1024]
N_REPEATS   = 20

PROF_TARGETS = {
    'cifar10':        {32: 0.39, 128: 0.54, 1024: 0.74},
    'cifar100':       {32: 0.39, 128: 0.56, 1024: 0.74},
    'ImageNet16-120': {32: 0.40, 128: 0.56, 1024: 0.74},
}

def evaluate_predictor(X, y, test_idx, pool, factory):
    X_test, y_test = X[test_idx], y[test_idx]
    results = {}
    for p in TRAIN_SIZES:
        taus = []
        for seed in range(N_REPEATS):
            rng    = np.random.RandomState(seed * 1000 + p)
            tr_idx = rng.choice(pool, size=min(p, len(pool)), replace=False)
            m      = factory()
            m.fit(X[tr_idx], y[tr_idx])
            tau, _ = scipy_kendalltau(y_test, m.predict(X_test))
            taus.append(float(tau))
        results[p] = taus
    return results

# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_tau_curve(ds_label, ds_key, results, save_path):
    ps    = sorted(results)
    means = [np.mean(results[p]) for p in ps]
    stds  = [np.std(results[p])  for p in ps]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(ps, means, marker='o', color='#FF5722', linewidth=2, markersize=6,
            label='SVR ensemble (predictor)')
    ax.fill_between(ps, [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    alpha=0.15, color='#FF5722')
    targets = PROF_TARGETS.get(ds_key, {})
    lbl_done = False
    for p_t, tau_t in targets.items():
        ax.scatter([p_t], [tau_t], marker='*', s=180, color='#4CAF50', zorder=5,
                   label=None if lbl_done else 'Prof. target')
        lbl_done = True
    ax.set_xscale('log')
    ax.set_xlabel('Training set size  p  (# architectures)', fontsize=11)
    ax.set_ylabel('Kendall Tau', fontsize=11)
    ax.set_title(f'Performance Predictor -- {ds_label}\n'
                 f'(mean +/- std over {N_REPEATS} seeds, {TEST_SIZE} held-out archs)',
                 fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close()

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    arch_strs, acc, encodings = load_naslib()
    print(f'Encoding dim: {encodings.shape[1]}  (30 one-hot + 180 path)\n')

    n_total  = len(arch_strs)
    rng_test = np.random.RandomState(0)
    test_idx = rng_test.choice(n_total, size=TEST_SIZE, replace=False)
    test_set = set(test_idx.tolist())
    pool     = np.array([i for i in range(n_total) if i not in test_set])

    DS_MAP = [('cifar10', 'CIFAR-10'), ('cifar100', 'CIFAR-100'),
              ('ImageNet16-120', 'ImageNet16-120')]

    all_results = {}
    for ds_key, ds_label in DS_MAP:
        y = acc[ds_key]
        print(f'{"="*60}')
        print(f'  {ds_label}  |  {n_total} archs  |  {y.min():.2f}% -- {y.max():.2f}%')
        print(f'{"="*60}')
        res = evaluate_predictor(encodings, y, test_idx, pool, make_predictor)

        targets = PROF_TARGETS.get(ds_key, {})
        print(f'  {"p":>6s}  {"Kendall Tau":>12s}  {"target":>7s}')
        print(f'  {"-"*30}')
        for p in TRAIN_SIZES:
            m   = np.mean(res[p]); s = np.std(res[p])
            tgt = targets.get(p)
            if tgt is not None:
                mark = ' *MET*' if m >= tgt else ' MISS'
                print(f'  {p:>6d}  {m:>7.4f}+-{s:.3f}  {tgt:>7.2f}{mark}')
            else:
                print(f'  {p:>6d}  {m:>7.4f}+-{s:.3f}  {"-":>7s}')

        all_results[ds_key] = {
            'ds_label': ds_label, 'n_total': n_total,
            'tau': {str(p): {'mean': float(np.mean(v)), 'std': float(np.std(v)),
                             'all': [float(x) for x in v]}
                    for p, v in res.items()},
            'targets': {str(k): v for k, v in targets.items()},
        }
        plot_tau_curve(ds_label, ds_key, res,
                       os.path.join(RESULTS,
                                    f'predictor_tau_{ds_key.lower().replace("-","_")}.png'))
        print(f'  Plot saved.\n')

    with open(os.path.join(RESULTS, 'predictor_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    # Summary table
    print(f'{"="*66}')
    print(f'  SUMMARY -- SVR ensemble Kendall Tau vs Professor Targets')
    print(f'{"="*66}')
    print(f'  {"Dataset":16s} {"p=32":>7s} {"tgt":>5s}  {"p=128":>7s} {"tgt":>5s}  '
          f'{"p=1024":>7s} {"tgt":>5s}')
    print(f'  {"-"*58}')
    total_met = 0
    for ds_key, ds_label in DS_MAP:
        t = PROF_TARGETS[ds_key]
        def cell(p):
            nonlocal total_met
            m = all_results[ds_key]['tau'][str(p)]['mean']
            met = m >= t[p]
            total_met += int(met)
            return f'{m:.3f}{"*" if met else " "}'
        print(f'  {ds_label:16s} {cell(32):>7s} {t[32]:.2f}  '
              f'{cell(128):>7s} {t[128]:.2f}  {cell(1024):>8s} {t[1024]:.2f}')
    print(f'  {"-"*58}')
    print(f'  Targets met: {total_met}/9   (* = meets or exceeds target)')
    print(f'\nResults saved to {os.path.join(RESULTS, "predictor_results.json")}')


if __name__ == '__main__':
    main()
