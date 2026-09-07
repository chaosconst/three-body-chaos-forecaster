#!/usr/bin/env python
"""v2: proper protocol (train / val / large test) and stronger models.
Test = original 1 500 (10k file, seed-0 split) + last 10 000 of the 100k file + fresh 3 000 = 14 500 systems.
Val  = 5 000 systems from the training pool (early stopping only). Train = the rest (~133 000).
Models: logistic, HistGradientBoosting (per task), MLP early-stopped on val (5 seeds, ensembled), GBDT+MLP average.
"""
import json, time, sys
import numpy as np, torch, torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
src = open("scripts/train_chaos.py").read()
exec(src[src.index("def features"):src.index("perm = np.random.permutation")])   # features, rand_rot, augment
torch.manual_seed(0); np.random.seed(0)
FILES = ["data/three_body_3d_10k_T30.npz", "data/three_body_3d_40k_T30.npz", "data/three_body_3d_100k_T30.npz", "data/three_body_3d_fresh3k.npz"]
parts = []
for f in FILES:
    d = np.load(f); POS = d["pos"]; n = len(POS); T = float(d["times"][-1])
    last = (~np.isnan(POS[:, :, 0, 0])).sum(1) - 1
    parts.append(dict(m=d["masses"], pos=POS[:, 0], vel=d["vel"][:, 0], st=d["status"], t_div=d["t_div"], n=n,
                      rlast=np.linalg.norm(POS[np.arange(n), last], axis=2).max(1)))
cat = lambda k: np.concatenate([p[k] for p in parts])
m, pos, vel, st, rlast, t_div = cat("m"), cat("pos"), cat("vel"), cat("st"), cat("rlast"), cat("t_div")
N = len(m); off = np.cumsum([0] + [p["n"] for p in parts])
Y = np.stack([(t_div < T) & (st == 0), rlast > 20, st == 1], 1).astype(np.float32)
W = np.stack([st == 0, np.ones(N, bool), np.ones(N, bool)], 1)
perm0 = np.random.permutation(parts[0]["n"])
test_idx = np.concatenate([perm0[:1500], np.arange(off[2] + 90000, off[3]), np.arange(off[3], off[4])])
pool = np.setdiff1d(np.arange(N), test_idx); pool = np.random.permutation(pool)
val_idx, tr_idx = pool[:5000], pool[5000:]
print(f"train {len(tr_idx)}  val {len(val_idx)}  test {len(test_idx)}", flush=True)
F = features(m, pos, vel); mu, sd = F[tr_idx].mean(0), F[tr_idx].std(0) + 1e-6; Z = (F - mu) / sd
TASKS = ["diverge", "escape", "collision"]
def aucs(P, idx):
    return [roc_auc_score(Y[idx, k][W[idx, k]], P[W[idx, k], k]) if True else 0 for k in range(3)]
def report(name, P):
    a = aucs(P, test_idx); sub = {}
    for lab, sl in [("orig1500", test_idx[:1500]), ("held10k", test_idx[1500:11500]), ("fresh3k", test_idx[11500:])]:
        pos_ = np.searchsorted(test_idx, sl) if False else np.isin(test_idx, sl)
        sub[lab] = aucs(P[pos_], sl)
    print(f"{name:34s} test AUC {a[0]:.3f} {a[1]:.3f} {a[2]:.3f}   | orig1500 {' '.join(f'{x:.3f}' for x in sub['orig1500'])} | fresh3k {' '.join(f'{x:.3f}' for x in sub['fresh3k'])}", flush=True)
    return dict(test=a, **sub)
results = {}
# ---- logistic ----
P = np.zeros((len(test_idx), 3))
for k in range(3):
    sel = W[tr_idx, k]; lr = LogisticRegression(max_iter=2000, C=1.0).fit(Z[tr_idx][sel], Y[tr_idx, k][sel]); P[:, k] = lr.predict_proba(Z[test_idx])[:, 1]
results["logistic"] = report("logistic", P)
# ---- GBDT ----
t0 = time.time(); P_gb = np.zeros((len(test_idx), 3)); best_iters = []
for k in range(3):
    sel = W[tr_idx, k]; vsel = W[val_idx, k]
    gb = HistGradientBoostingClassifier(max_iter=2000, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0,
                                        early_stopping=True, validation_fraction=None, n_iter_no_change=50, random_state=0)
    # use our own val split: sklearn early stopping needs validation_fraction; emulate by concatenating and letting it split is not exact,
    # so instead do a manual staged search on val
    gb.set_params(early_stopping=False, max_iter=1500); gb.fit(F[tr_idx][sel], Y[tr_idx, k][sel])
    stages = list(gb.staged_predict_proba(F[val_idx][vsel])); va = [roc_auc_score(Y[val_idx, k][vsel], s[:, 1]) for s in stages[9::10]]
    bi = (int(np.argmax(va)) + 1) * 10; best_iters.append(bi)
    Pt = list(gb.staged_predict_proba(F[test_idx]))[bi - 1][:, 1]; P_gb[:, k] = Pt
print(f"GBDT best iterations per task {best_iters}  ({time.time()-t0:.0f}s)")
results["gbdt"] = report("GBDT (val-tuned iterations)", P_gb)
# ---- MLP: early stopping on val, 5 seeds ----
class Net(nn.Module):
    def __init__(s, nf, h=128):
        super().__init__(); s.f = nn.Sequential(nn.Linear(nf, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 3))
    def forward(s, z): return s.f(z)
def train_mlp(seed, epochs=60):
    torch.manual_seed(seed); np.random.seed(seed); net = Net(F.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    Yt, Wt = torch.tensor(Y[tr_idx]), torch.tensor(W[tr_idx].astype(np.float32)); Zv = torch.tensor(Z[val_idx], dtype=torch.float32)
    best, best_state, best_ep = -1, None, 0
    for ep in range(epochs):
        net.train(); mm, xx, vv = augment(m[tr_idx], pos[tr_idx], vel[tr_idx]); X = torch.tensor((features(mm, xx, vv) - mu) / sd, dtype=torch.float32)
        p = torch.randperm(len(tr_idx))
        for b in range(0, len(tr_idx), 256):
            i = p[b:b+256]; loss = (nn.functional.binary_cross_entropy_with_logits(net(X[i]), Yt[i], reduction="none") * Wt[i]).sum() / Wt[i].sum()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step(); net.eval()
        with torch.no_grad(): Pv = torch.sigmoid(net(Zv)).numpy()
        score = np.mean(aucs(Pv, val_idx))
        if score > best: best, best_state, best_ep = score, {k: v.clone() for k, v in net.state_dict().items()}, ep + 1
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad(): Pt = torch.sigmoid(net(torch.tensor(Z[test_idx], dtype=torch.float32))).numpy()
    print(f"  mlp seed {seed}: best val epoch {best_ep}, val mean AUC {best:.3f}", flush=True); return Pt
t0 = time.time(); Ps = [train_mlp(s) for s in range(5)]; print(f"  5 MLPs in {time.time()-t0:.0f}s")
results["mlp_single"] = report("MLP, val early-stopped, 1 seed", Ps[0])
P_mlp = np.mean(Ps, 0); results["mlp_ens5"] = report("MLP ensemble, 5 seeds", P_mlp)
results["gbdt+mlp"] = report("GBDT + MLP ensemble average", 0.5 * (P_gb + P_mlp))
json.dump(dict(results=results, n_train=len(tr_idx), n_val=len(val_idx), n_test=len(test_idx), gbdt_iters=best_iters), open("data/v2_results.json", "w"))
np.savez_compressed("data/v2_test_preds.npz", test_idx=test_idx, P_gb=P_gb, P_mlp=P_mlp)
print("saved data/v2_results.json")
