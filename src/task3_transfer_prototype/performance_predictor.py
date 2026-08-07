"""
Performance Predictor for NAS-Bench-201 Architectures
======================================================
Input  : architecture encoding (30-dim one-hot: 6 edges x 5 ops)
Target : test accuracy on CIFAR-10, CIFAR-100, or ImageNet16-120
Metric : Kendall Tau -- rank correlation between predicted and true rankings

Two surrogate models: Random Forest, XGBoost.

Also runs a training-size sweep (the 'p' from the meeting):
  Train predictor on p labelled architectures -> evaluate Kendall Tau on the rest.
  Repeated 50 times per p to average out sampling noise.
  Answers: how few evaluations do you need to rank architectures reliably?

Data: 47 architectures from all/all/*.pickle.pbz2
"""

import os, sys, pickle, bz2, json, warnings
warnings.filterwarnings('ignore')

import numpy as np
from scipy.stats     import kendalltau as scipy_kendalltau
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

RAML_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALL_DIR  = os.path.join(RAML_DIR, 'all', 'all')
OUT_DIR  = os.path.dirname(os.path.abspath(__file__))

# ── Architecture encoding ─────────────────────────────────────────────────────

OPS    = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
OP2IDX = {op: i for i, op in enumerate(OPS)}
N_OPS   = len(OPS)    # 5
N_EDGES = 6           # NB201 has 4 nodes -> 6 directed edges

def parse_arch_str(arch_str):
    """Parse NB201 arch string -> list of 6 op names.
    Format: |op~0|+|op~0|op~1|+|op~0|op~1|op~2|
    """
    ops = []
    for token in arch_str.replace('+', '|').split('|'):
        token = token.strip()
        if '~' in token:
            ops.append(token.split('~')[0].strip())
    assert len(ops) == N_EDGES, f"Expected 6 ops, got {len(ops)}: {arch_str}"
    return ops

def encode_arch(arch_str):
    """30-dim one-hot vector (6 edges x 5 ops)."""
    enc = np.zeros(N_EDGES * N_OPS, dtype=np.float32)
    for i, op in enumerate(parse_arch_str(arch_str)):
        enc[i * N_OPS + OP2IDX[op]] = 1.0
    return enc

def arch_to_readable(arch_str):
    """Short readable form: comma-separated op abbreviations."""
    abbrev = {'none': 'N', 'skip_connect': 'S',
              'nor_conv_1x1': '1', 'nor_conv_3x3': '3', 'avg_pool_3x3': 'P'}
    return '-'.join(abbrev.get(op, op) for op in parse_arch_str(arch_str))

# ── Data loading ──────────────────────────────────────────────────────────────

SEEDS     = (777, 888, 999)
FINAL_KEY = 'ori-test@199'

def load_all_data():
    """Load all pickle files; return list of record dicts."""
    fnames  = sorted(f for f in os.listdir(ALL_DIR) if f.endswith('.pickle.pbz2'))
    records = []

    for fname in fnames:
        with bz2.BZ2File(os.path.join(ALL_DIR, fname), 'rb') as f:
            p = pickle.load(f)

        d200     = p['200']
        arch_str = d200['arch_str']
        results  = d200['all_results']

        def mean_acc(dataset):
            vals = []
            for s in SEEDS:
                key = (dataset, s)
                if key in results:
                    acc = results[key]['eval_acc1es'].get(FINAL_KEY)
                    if acc is not None:
                        vals.append(float(acc))
            return float(np.mean(vals)) if vals else None

        c10  = mean_acc('cifar10')
        c100 = mean_acc('cifar100')
        inet = mean_acc('ImageNet16-120')

        if c10 is None and c100 is None and inet is None:
            continue

        records.append({
            'arch_id':  fname.replace('.pickle.pbz2', ''),
            'arch_str': arch_str,
            'enc':      encode_arch(arch_str),
            'cifar10':  c10,
            'cifar100': c100,
            'imagenet': inet,
        })

    print(f"Loaded {len(records)} architectures with benchmark data.")
    return records

# ── Surrogate models ──────────────────────────────────────────────────────────

def make_rf():
    return RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)

def make_xgb():
    return XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        random_state=42, eval_metric='rmse', verbosity=0)

MODELS = [('RandomForest', make_rf)]
if HAS_XGB:
    MODELS.append(('XGBoost', make_xgb))

# ── Leave-One-Out evaluation ──────────────────────────────────────────────────

def loo_eval(X, y, model_factory):
    """Leave-one-out CV: returns (preds array, Kendall Tau)."""
    preds = np.empty_like(y)
    for tr, te in LeaveOneOut().split(X):
        m = model_factory()
        m.fit(X[tr], y[tr])
        preds[te] = m.predict(X[te])
    tau, pval = scipy_kendalltau(y, preds)
    return preds, tau, pval

# ── Training-size sweep ───────────────────────────────────────────────────────

def p_sweep(X, y, model_factory, p_values, n_repeats=50, seed=42):
    """
    For each p: randomly sample p arches as train, evaluate on rest.
    Returns dict {p -> mean_kendall_tau}.
    """
    n   = len(y)
    rng = np.random.RandomState(seed)
    out = {}

    for p in p_values:
        taus = []
        iters = n_repeats if p < n else 1
        for _ in range(iters):
            idx   = rng.permutation(n)
            tr, te = idx[:p], idx[p:]
            if len(te) < 3:
                continue
            m = model_factory()
            m.fit(X[tr], y[tr])
            pred      = m.predict(X[te])
            tau, _    = scipy_kendalltau(y[te], pred)
            taus.append(tau)
        out[p] = float(np.mean(taus)) if taus else float('nan')

    return out

# ── Plotting ──────────────────────────────────────────────────────────────────

COLORS = {'RandomForest': '#2196F3', 'XGBoost': '#FF5722'}

def plot_p_sweep(ds_label, sweeps_by_model, n_total, save_path):
    """sweeps_by_model: {model_name -> {p -> tau}}"""
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, sweep in sweeps_by_model.items():
        ps   = sorted(sweep)
        taus = [sweep[p] for p in ps]
        ax.plot(ps, taus, marker='o', label=name,
                color=COLORS.get(name), linewidth=2, markersize=5)

    ax.axhline(0,   color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.axhline(0.5, color='green', linestyle=':', linewidth=0.8, alpha=0.6,
               label='tau=0.5 (target)')
    ax.set_xlabel('Training set size  p  (# architectures labelled)', fontsize=11)
    ax.set_ylabel('Kendall Tau', fontsize=11)
    ax.set_title(f'Performance Predictor — {ds_label}\n'
                 f'(evaluated on held-out arches, 50 repeats per p)', fontsize=11)
    ax.set_ylim(-0.2, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close()

def plot_scatter(ds_label, true_y, preds_by_model, save_path):
    """True vs predicted scatter (LOO)."""
    n_models = len(preds_by_model)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4), squeeze=False)
    for ax, (name, preds) in zip(axes[0], preds_by_model.items()):
        ax.scatter(true_y, preds, alpha=0.7, s=30, color=COLORS.get(name, 'steelblue'))
        mn, mx = min(true_y.min(), preds.min()), max(true_y.max(), preds.max())
        ax.plot([mn, mx], [mn, mx], 'r--', linewidth=1)
        tau, _ = scipy_kendalltau(true_y, preds)
        ax.set_title(f'{name}  (tau={tau:.3f})', fontsize=10)
        ax.set_xlabel('True accuracy (%)')
        ax.set_ylabel('Predicted accuracy (%)')
        ax.grid(True, alpha=0.2)
    fig.suptitle(f'LOO Predictions — {ds_label}', fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close()

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

    records = load_all_data()
    X       = np.stack([r['enc'] for r in records])   # (N, 30)

    DS_INFO = [
        ('cifar10',  'CIFAR-10'),
        ('cifar100', 'CIFAR-100'),
        ('imagenet', 'ImageNet16-120'),
    ]

    all_results = {}

    for ds_key, ds_label in DS_INFO:
        y_raw = [r[ds_key] for r in records]
        mask  = [v is not None for v in y_raw]
        X_ds  = X[mask]
        y_ds  = np.array([v for v in y_raw if v is not None], dtype=np.float64)
        arch_ids = [r['arch_id'] for r, m in zip(records, mask) if m]

        n = len(y_ds)
        if n < 10:
            print(f"\nSkipping {ds_label}: only {n} samples.")
            continue

        print(f"\n{'='*62}")
        print(f"  {ds_label}  ({n} architectures)")
        print(f"  Accuracy range: {y_ds.min():.2f}% -- {y_ds.max():.2f}%")
        print(f"{'='*62}")

        loo_results   = {}
        preds_for_scatter = {}
        sweeps        = {}

        p_values = sorted(set(
            list(range(5, min(n, 41), 5)) + [n]
        ))

        for name, factory in MODELS:
            preds, tau, pval = loo_eval(X_ds, y_ds, factory)
            print(f"  {name:15s}  LOO Kendall Tau = {tau:.4f}  (p={pval:.4f})")

            sweep = p_sweep(X_ds, y_ds, factory, p_values=p_values, n_repeats=50)
            sweep[n] = tau   # replace nan at p=n with the proper LOO result
            sweeps[name] = sweep
            loo_results[name] = {'tau': tau, 'pval': pval}
            preds_for_scatter[name] = preds

            print(f"               p-sweep (mean tau): " +
                  "  ".join(f"p={p}:{sweep[p]:.2f}" for p in p_values[::2]))

        all_results[ds_key] = {
            'n_archs': n,
            'acc_min': float(y_ds.min()),
            'acc_max': float(y_ds.max()),
            'loo': loo_results,
            'p_sweep': {name: {str(k): v for k, v in sw.items()}
                        for name, sw in sweeps.items()},
        }

        # Plots
        plot_p_sweep(
            ds_label, sweeps, n,
            os.path.join(OUT_DIR, f'predictor_psweep_{ds_key}.png')
        )
        plot_scatter(
            ds_label, y_ds, preds_for_scatter,
            os.path.join(OUT_DIR, f'predictor_scatter_{ds_key}.png')
        )
        print(f"  Plots saved: predictor_psweep_{ds_key}.png  +  predictor_scatter_{ds_key}.png")

        # Top/bottom architectures according to true ranking
        order = np.argsort(y_ds)[::-1]
        print(f"\n  True ranking (top 5 / bottom 5):")
        print(f"  {'Rank':4s}  {'Arch':8s}  {'Acc%':7s}  {'Encoding'}")
        for rank_i in list(range(5)) + list(range(n-5, n)):
            i = order[rank_i]
            sep = '...' if rank_i == 5 else ''
            if sep:
                print(f"  {'':4s}  {'...':8s}")
            print(f"  {rank_i+1:4d}  {arch_ids[i]:8s}  {y_ds[i]:7.3f}  "
                  f"{arch_to_readable(records[[r['arch_id'] for r in records].index(arch_ids[i])]['arch_str'])}")

    # Save JSON
    out_path = os.path.join(OUT_DIR, 'predictor_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nResults saved to {out_path}")

    # Final summary table
    print(f"\n{'='*62}")
    print(f"  SUMMARY: Leave-One-Out Kendall Tau")
    print(f"{'='*62}")
    print(f"  {'Dataset':20s}  {'RandomForest':14s}  {'XGBoost':14s}")
    print(f"  {'-'*54}")
    for ds_key, ds_label in DS_INFO:
        if ds_key not in all_results:
            continue
        loo = all_results[ds_key]['loo']
        rf  = loo.get('RandomForest', {}).get('tau', float('nan'))
        xgb = loo.get('XGBoost',      {}).get('tau', float('nan'))
        print(f"  {ds_label:20s}  {rf:14.4f}  {xgb:14.4f}")
    print()


if __name__ == '__main__':
    main()
