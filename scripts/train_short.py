#!/usr/bin/env python
"""Short-integration forecaster: features from the state at t_short (plus min pair distance
so far and the shadow-run separation at t_short), same labels, same test split."""
import argparse, json, time, sys
import numpy as np, torch, torch.nn as nn
src = open("scripts/train_chaos.py").read()
head = src.split("files = a.data.split")[0]      # imports, argparse, feature/rot/aug helpers come later; grab pieces
ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/three_body_3d_10k_T30.npz,data/three_body_3d_40k_T30.npz")
ap.add_argument("--t_short", type=float, default=3.0)
ap.add_argument("--epochs", type=int, default=80); ap.add_argument("--hidden", type=int, default=256); ap.add_argument("--ntest", type=int, default=1500)
ap.add_argument("--out", default="data/short_model.json")
a = ap.parse_args(); torch.manual_seed(0); np.random.seed(0)
# reuse feature/augmentation functions from the main script
exec(src[src.index("def features"):src.index("perm = np.random.permutation")])
parts = []
for f in a.data.split(","):
    d = np.load(f); POS, VEL, SEP, times = d["pos"], d["vel"], d["sep"], d["times"]; n = len(POS); T = float(times[-1])
    k = int(np.argmin(np.abs(times - a.t_short)))
    last = (~np.isnan(POS[:, :, 0, 0])).sum(1) - 1
    kk = np.minimum(k, last)                       # runs stopped before t_short use their last frame
    stopped = (last < k).astype(np.float32)
    pr = [np.linalg.norm(POS[:, :k+1, i] - POS[:, :k+1, j], axis=2) for i, j in [(0,1),(0,2),(1,2)]]
    dmin = np.nanmin(np.stack(pr, 1), axis=(1, 2))
    sep = SEP[np.arange(n), kk]; sep = np.where(np.isnan(sep), 1e-10, sep)
    parts.append(dict(m=d["masses"], pos=POS[np.arange(n), kk], vel=VEL[np.arange(n), kk], st=d["status"], t_div=d["t_div"],
        rlast=np.linalg.norm(POS[np.arange(n), last], axis=2).max(1), n=n,
        extra=np.stack([np.log(dmin), np.log10(np.clip(sep, 1e-13, 1e3)), stopped, np.log(np.clip(d["e_err"],1e-17,1))*0], 1)[:, :3].astype(np.float32),
        cost=(times[k] / T)))
cat = lambda key: np.concatenate([p[key] for p in parts])
m, pos, vel, st, rlast, t_div, extra = cat("m"), cat("pos"), cat("vel"), cat("st"), cat("rlast"), cat("t_div"), cat("extra")
N = len(m); N0 = parts[0]["n"]
y_col = (st == 1).astype(np.float32); y_esc = (rlast > 20).astype(np.float32)
y_div = ((t_div < T) & (st == 0)).astype(np.float32); mask_div = (st == 0).astype(np.float32)
Y = np.stack([y_div, y_esc, y_col], 1); W = np.stack([mask_div, np.ones(N), np.ones(N)], 1).astype(np.float32)
perm = np.random.permutation(N0); te = perm[:a.ntest]; tr = np.concatenate([perm[a.ntest:], np.arange(N0, N)])
feat = lambda mm, xx, vv, ex: np.concatenate([features(mm, xx, vv), ex], 1)
F0 = feat(m, pos, vel, extra); nf = F0.shape[1]; mu, sd = F0[tr].mean(0), F0[tr].std(0) + 1e-6
class Net(nn.Module):
    def __init__(s, nf, h):
        super().__init__(); s.f = nn.Sequential(nn.Linear(nf, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 3))
    def forward(s, z): return s.f(z)
def auc(y, p):
    o = np.argsort(p); r = np.empty(len(p)); r[o] = np.arange(1, len(p)+1); npos = y.sum(); nneg = len(y)-npos
    return (r[y==1].sum() - npos*(npos+1)/2) / (npos*nneg)
def evaluate(model, tag):
    model.eval()
    with torch.no_grad(): P = torch.sigmoid(model(torch.tensor((F0[te]-mu)/sd))).numpy()
    res = {}
    for k, name in enumerate(["diverge","escape","collision"]):
        sel = W[te,k] > 0; y, p = Y[te,k][sel], P[sel,k]; acc = ((p>0.5)==(y>0.5)).mean()
        res[name] = dict(auc=float(auc(y,p)), acc=float(acc), majority=float(max(y.mean(),1-y.mean())))
        print(f"[{tag}] {name:9s} AUC {auc(y,p):.3f} acc {acc:.3f} (majority {res[name]['majority']:.3f})", flush=True)
    return res
def run(model, aug, epochs, tag):
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    Yt, Wt = torch.tensor(Y[tr]), torch.tensor(W[tr])
    for ep in range(epochs):
        model.train(); mm, xx, vv = (augment(m[tr], pos[tr], vel[tr]) if aug else (m[tr], pos[tr], vel[tr]))
        X = torch.tensor((feat(mm, xx, vv, extra[tr]) - mu) / sd); p = torch.randperm(len(tr))
        for b in range(0, len(tr), 256):
            i = p[b:b+256]; loss = (nn.functional.binary_cross_entropy_with_logits(model(X[i]), Yt[i], reduction="none") * Wt[i]).sum() / Wt[i].sum()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    return model
# rule: shadow separation alone
for k, name in enumerate(["diverge","escape","collision"]):
    sel = W[te,k] > 0; a_ = auc(Y[te,k][sel], extra[te][sel,1]); print(f"[rule log-sep@t={a.t_short}] {name:9s} AUC {max(a_,1-a_):.3f}")
    sel = W[te,k] > 0; a_ = auc(Y[te,k][sel], -extra[te][sel,0]); print(f"[rule -log dmin@t={a.t_short}] {name:9s} AUC {max(a_,1-a_):.3f}")
results = {}
lin = nn.Sequential(nn.Linear(nf, 3)); run(lin, False, 40, "logistic"); results["logistic"] = evaluate(lin, f"logistic t={a.t_short}")
net = Net(nf, a.hidden); run(net, True, a.epochs, "mlp"); results["mlp_aug"] = evaluate(net, f"mlp-aug t={a.t_short}")
results["cost_fraction"] = float(parts[0]["cost"]); results["t_short"] = a.t_short
json.dump(results, open(a.out, "w")); print("saved", a.out, "cost fraction of full run:", results["cost_fraction"])
