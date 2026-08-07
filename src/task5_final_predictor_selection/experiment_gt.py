"""
Diagnostic: compare Kendall Tau using NASLib validation accuracy (nb201_all.pickle)
vs the NATS-tss test accuracy we used before. Same encoding + XGBoost.
"""
import os, sys, pickle, warnings
warnings.filterwarnings('ignore')
import numpy as np
from scipy.stats import kendalltau
from xgboost import XGBRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step1_predictor import encode_arch, make_xgb

NASLIB_DATA = r'C:\Users\Flash\naslib\naslib\data\nb201_all.pickle'

# Which NASLib sub-key holds the ground-truth val accuracy for each dataset
GT_KEY = {
    'cifar10':        'cifar10-valid',
    'cifar100':       'cifar100',
    'ImageNet16-120': 'ImageNet16-120',
}
TARGETS = {
    'cifar10':        {32: 0.39, 128: 0.54, 1024: 0.74},
    'cifar100':       {32: 0.39, 128: 0.56, 1024: 0.74},
    'ImageNet16-120': {32: 0.40, 128: 0.56, 1024: 0.74},
}
TRAIN_SIZES = [32, 128, 1024]
N_REPEATS   = 20
TEST_SIZE   = 2000

print('Loading NASLib nb201_all.pickle ...', end=' ', flush=True)
with open(NASLIB_DATA, 'rb') as f:
    naslib = pickle.load(f)
arch_strs = list(naslib.keys())
print(f'{len(arch_strs)} archs.')

# Build encodings once
print('Encoding ...', end=' ', flush=True)
X = np.stack([encode_arch(a) for a in arch_strs])
print('done.')

def build_y(ds):
    return np.array([naslib[a][GT_KEY[ds]]['eval_acc1es'] for a in arch_strs],
                    dtype=np.float64)

rng_test = np.random.RandomState(0)
test_idx = rng_test.choice(len(arch_strs), size=TEST_SIZE, replace=False)
test_set = set(test_idx.tolist())
pool     = np.array([i for i in range(len(arch_strs)) if i not in test_set])

def eval_ds(ds):
    y = build_y(ds)
    X_test, y_test = X[test_idx], y[test_idx]
    out = {}
    for p in TRAIN_SIZES:
        taus = []
        for seed in range(N_REPEATS):
            r = np.random.RandomState(seed * 1000 + p)
            tr = r.choice(pool, size=p, replace=False)
            m = make_xgb()
            m.fit(X[tr], y[tr])
            tau, _ = kendalltau(y_test, m.predict(X_test))
            taus.append(tau)
        out[p] = np.mean(taus)
    return y, out

print(f'\n{"="*64}')
print(f'  NASLib VALIDATION accuracy  |  XGBoost(d6)  |  {N_REPEATS} seeds')
print(f'{"="*64}')
print(f'  {"Dataset":18s} {"range":16s} {"p=32":>10s} {"p=128":>10s} {"p=1024":>10s}')
for ds in ['cifar10', 'cifar100', 'ImageNet16-120']:
    y, res = eval_ds(ds)
    t = TARGETS[ds]
    def cell(p):
        m = res[p]; tgt = t[p]
        return f'{m:.3f}{"*" if m>=tgt else " "}'
    print(f'  {ds:18s} {y.min():5.1f}-{y.max():5.1f}%   '
          f'{cell(32):>10s} {cell(128):>10s} {cell(1024):>10s}')
print(f'  targets:           {"":16s} '
      f'{"~.39":>10s} {"~.55":>10s} {"0.74":>10s}')
print('  (* meets/exceeds target)')
