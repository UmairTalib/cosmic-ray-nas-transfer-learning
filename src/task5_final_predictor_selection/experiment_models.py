"""
Model sweep: find predictors that hit the professor's Kendall Tau targets,
especially at small training budgets (p=32, 128). Ground truth = NASLib
validation accuracy (nb201_all.pickle).
"""
import os, sys, pickle, warnings, time
warnings.filterwarnings('ignore')
import numpy as np
from scipy.stats import kendalltau

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step1_predictor import (encode_arch, encode_arch_onehot, encode_arch_path)

from sklearn.linear_model import Ridge, BayesianRidge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from xgboost import XGBRegressor

NASLIB_DATA = r'C:\Users\Flash\naslib\naslib\data\nb201_all.pickle'
GT_KEY = {'cifar10':'cifar10-valid','cifar100':'cifar100','ImageNet16-120':'ImageNet16-120'}
TARGETS = {
    'cifar10':        {32:0.39,128:0.54,1024:0.74},
    'cifar100':       {32:0.39,128:0.56,1024:0.74},
    'ImageNet16-120': {32:0.40,128:0.56,1024:0.74},
}
TRAIN_SIZES = [32,128,1024]
N_REPEATS   = 10
TEST_SIZE   = 2000

print('Loading NASLib ...', end=' ', flush=True)
with open(NASLIB_DATA,'rb') as f: naslib = pickle.load(f)
arch_strs = list(naslib.keys())
X_onehot = np.stack([encode_arch_onehot(a) for a in arch_strs])
X_path   = np.stack([encode_arch_path(a)   for a in arch_strs])
X_comb   = np.concatenate([X_onehot, X_path], axis=1)
print(f'{len(arch_strs)} archs.')

def build_y(ds):
    return np.array([naslib[a][GT_KEY[ds]]['eval_acc1es'] for a in arch_strs], dtype=np.float64)

rng_test = np.random.RandomState(0)
test_idx = rng_test.choice(len(arch_strs), size=TEST_SIZE, replace=False)
test_set = set(test_idx.tolist())
pool = np.array([i for i in range(len(arch_strs)) if i not in test_set])

def gp_kernel():
    return ConstantKernel(1.0) * RBF(length_scale=5.0) + WhiteKernel(noise_level=1.0)

MODELS = {
    'Ridge(1h)':     ('onehot', lambda: Ridge(alpha=1.0)),
    'Ridge(comb)':   ('comb',   lambda: Ridge(alpha=1.0)),
    'BayesRidge(1h)':('onehot', lambda: BayesianRidge()),
    'BayesRidge(cb)':('comb',   lambda: BayesianRidge()),
    'GP(comb)':      ('comb',   lambda: GaussianProcessRegressor(kernel=gp_kernel(), normalize_y=True, alpha=1e-6)),
    'SVR(comb)':     ('comb',   lambda: SVR(C=10.0, kernel='rbf', gamma='scale')),
    'RF(cb,d10)':    ('comb',   lambda: RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1)),
    'GBRT(cb)':      ('comb',   lambda: GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.1, subsample=0.8, random_state=42)),
    'XGB(cb,d3)':    ('comb',   lambda: XGBRegressor(n_estimators=200,max_depth=3,learning_rate=0.1,subsample=0.8,colsample_bytree=0.6,reg_alpha=0.5,reg_lambda=2.0,random_state=42,verbosity=0)),
}
XMAP = {'onehot':X_onehot,'comb':X_comb,'path':X_path}

def eval_model(xkey, factory, y):
    X = XMAP[xkey]
    Xte, yte = X[test_idx], y[test_idx]
    out = {}
    for p in TRAIN_SIZES:
        taus=[]
        for seed in range(N_REPEATS):
            r=np.random.RandomState(seed*1000+p)
            tr=r.choice(pool,size=p,replace=False)
            m=factory(); m.fit(X[tr],y[tr])
            tau,_=kendalltau(yte,m.predict(Xte)); taus.append(tau)
        out[p]=np.mean(taus)
    return out

for ds in ['cifar10','cifar100','ImageNet16-120']:
    y=build_y(ds); t=TARGETS[ds]
    print(f'\n{"="*60}\n  {ds}\n{"="*60}')
    print(f'  {"model":16s} {"p=32":>10s} {"p=128":>10s} {"p=1024":>10s}')
    print(f'  {"target":16s} {t[32]:>10.2f} {t[128]:>10.2f} {t[1024]:>10.2f}')
    print('  '+'-'*50)
    for name,(xk,fac) in MODELS.items():
        ts=time.time()
        r=eval_model(xk,fac,y)
        def c(p): return f'{r[p]:.3f}{"*" if r[p]>=t[p] else " "}'
        print(f'  {name:16s} {c(32):>10s} {c(128):>10s} {c(1024):>10s}   ({time.time()-ts:.0f}s)')
