"""
Final predictor validation. Test ensemble strategies to hit all 9 targets.
Rank-average ensembles blend a small-budget SVR (gamma=0.02) with a
large-budget SVR (gamma='scale') and optionally deep XGBoost.
"""
import os, sys, pickle, warnings
warnings.filterwarnings('ignore')
import numpy as np
from scipy.stats import kendalltau, rankdata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step1_predictor import encode_arch_onehot, encode_arch_path, make_xgb
from sklearn.svm import SVR

NASLIB_DATA = r'C:\Users\Flash\naslib\naslib\data\nb201_all.pickle'
GT_KEY = {'cifar10':'cifar10-valid','cifar100':'cifar100','ImageNet16-120':'ImageNet16-120'}
TARGETS = {
    'cifar10':        {32:0.39,128:0.54,1024:0.74},
    'cifar100':       {32:0.39,128:0.56,1024:0.74},
    'ImageNet16-120': {32:0.40,128:0.56,1024:0.74},
}
TRAIN_SIZES=[32,64,128,256,512,1024]; N_REPEATS=20; TEST_SIZE=2000

with open(NASLIB_DATA,'rb') as f: naslib=pickle.load(f)
arch_strs=list(naslib.keys())
X=np.concatenate([np.stack([encode_arch_onehot(a) for a in arch_strs]),
                  np.stack([encode_arch_path(a)   for a in arch_strs])],axis=1)
def build_y(ds): return np.array([naslib[a][GT_KEY[ds]]['eval_acc1es'] for a in arch_strs],dtype=np.float64)
rng=np.random.RandomState(0); test_idx=rng.choice(len(arch_strs),TEST_SIZE,replace=False)
ts=set(test_idx.tolist()); pool=np.array([i for i in range(len(arch_strs)) if i not in ts])

def svr_small(): return SVR(C=20, gamma=0.02)
def svr_large(): return SVR(C=10, gamma='scale')

class RankEnsemble:
    """Rank-average predictions of several fitted models."""
    def __init__(self, factories): self.factories=factories
    def fit(self, Xtr, ytr):
        self.models=[f() for f in self.factories]
        for m in self.models: m.fit(Xtr,ytr)
        return self
    def predict(self, Xte):
        ranks=[rankdata(m.predict(Xte)) for m in self.models]
        return np.mean(ranks,axis=0)

PREDICTORS={
    'SVR-small':          lambda: svr_small(),
    'SVR-large':          lambda: svr_large(),
    'ENS 2xSVR':          lambda: RankEnsemble([svr_small, svr_large]),
    'ENS SVR+SVR+XGB':    lambda: RankEnsemble([svr_small, svr_large, make_xgb]),
}

def evalp(factory,y):
    Xte,yte=X[test_idx],y[test_idx]; out={}
    for p in TRAIN_SIZES:
        taus=[]
        for s in range(N_REPEATS):
            r=np.random.RandomState(s*1000+p); tr=r.choice(pool,size=p,replace=False)
            m=factory(); m.fit(X[tr],y[tr]); taus.append(kendalltau(yte,m.predict(Xte))[0])
        out[p]=np.mean(taus)
    return out

for ds in ['cifar10','cifar100','ImageNet16-120']:
    y=build_y(ds); t=TARGETS[ds]
    print(f'\n{"="*70}\n  {ds}  (targets: p32={t[32]} p128={t[128]} p1024={t[1024]})\n{"="*70}')
    print(f'  {"predictor":18s} {"p=32":>8s} {"p=64":>8s} {"p=128":>8s} {"p=256":>8s} {"p=512":>8s} {"p=1024":>8s} {"hits":>5s}')
    for name,fac in PREDICTORS.items():
        r=evalp(fac,y)
        hits=sum(r[p]>=t[p] for p in [32,128,1024])
        def c(p):
            star='*' if (p in t and r[p]>=t[p]) else ' '
            return f'{r[p]:.3f}{star}'
        print(f'  {name:18s} {c(32):>8s} {c(64):>8s} {c(128):>8s} {c(256):>8s} {c(512):>8s} {c(1024):>8s} {hits:>4d}/3')
