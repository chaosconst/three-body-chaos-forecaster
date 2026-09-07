import numpy as np, json, sys
sys.argv=["x"]
src=open("scripts/train_chaos.py").read()
exec(src.split("perm = np.random.permutation(N)")[0].replace("a = ap.parse_args()","a = ap.parse_args([])"))
Y = np.stack([y_div, y_esc, y_col], 1); W = np.stack([mask_div, np.ones(N), np.ones(N)], 1)
perm = np.random.permutation(N); te = perm[:a.ntest]
names = ["E","L","Q","r_min_sorted","r_mid","r_max","pair_e_min","pair_e_mid","pair_e_max","rdot_min","rdot_mid","rdot_max","m_min","m_mid","m_max","r_min","log r_min"]
F = features(m, pos, vel)[:, :len(names)]
def auc(y, p):
    o = np.argsort(p); r = np.empty(len(p)); r[o] = np.arange(1, len(p)+1); npos = y.sum(); nneg = len(y)-npos
    return (r[y==1].sum() - npos*(npos+1)/2) / (npos*nneg)
out = {}
for k, task in enumerate(["diverge","escape","collision"]):
    sel = W[te,k] > 0; y = Y[te,k][sel]
    rows = []
    for j, nm in enumerate(names):
        a_ = auc(y, F[te][sel, j]); rows.append((nm, max(a_, 1-a_), "high" if a_>=0.5 else "low"))
    rows.sort(key=lambda r: -r[1]); out[task] = rows[:5]
    print(task, " | ".join(f"{nm}: {a_:.3f} ({d})" for nm, a_, d in rows[:5]))
json.dump(out, open("data/rule_baselines.json","w"))
