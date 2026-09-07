#!/usr/bin/env python
"""v3: same protocol as v2, MLP with ordinal auxiliary heads on t_div (<10, <20, <30) + escape + collision.
Compares 'diverge' AUC (the <30 head) against the plain 3-head MLP, same seeds, same early stopping."""
import json, time, numpy as np, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
src = open("scripts/train_v2.py").read()
exec(src[src.index("import json"):src.index("TASKS = ")].replace("from sklearn.ensemble import HistGradientBoostingClassifier\n", ""))  # data, splits, features, Z
Y5 = np.stack([(t_div < 10) & (st == 0), (t_div < 20) & (st == 0), (t_div < 30) & (st == 0), rlast > 20, st == 1], 1).astype(np.float32)
W5 = np.stack([st == 0, st == 0, st == 0, np.ones(N, bool), np.ones(N, bool)], 1).astype(np.float32)
class Net(nn.Module):
    def __init__(s, nf, nout, h=128):
        super().__init__(); s.f = nn.Sequential(nn.Linear(nf, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, nout))
    def forward(s, z): return s.f(z)
def train(seed, Yall, Wall, heads_eval, epochs=60):
    torch.manual_seed(seed); np.random.seed(seed); net = Net(F.shape[1], Yall.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    Yt, Wt = torch.tensor(Yall[tr_idx]), torch.tensor(Wall[tr_idx]); Zv = torch.tensor(Z[val_idx], dtype=torch.float32)
    best, best_state = -1, None
    for ep in range(epochs):
        net.train(); mm, xx, vv = augment(m[tr_idx], pos[tr_idx], vel[tr_idx]); X = torch.tensor((features(mm, xx, vv) - mu) / sd, dtype=torch.float32)
        p = torch.randperm(len(tr_idx))
        for b in range(0, len(tr_idx), 256):
            i = p[b:b+256]; loss = (nn.functional.binary_cross_entropy_with_logits(net(X[i]), Yt[i], reduction="none") * Wt[i]).sum() / Wt[i].sum()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step(); net.eval()
        with torch.no_grad(): Pv = torch.sigmoid(net(Zv)).numpy()
        score = np.mean([roc_auc_score(Yall[val_idx, k][Wall[val_idx, k] > 0], Pv[Wall[val_idx, k] > 0, k]) for k in heads_eval])
        if score > best: best, best_state = score, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad(): return torch.sigmoid(net(torch.tensor(Z[test_idx], dtype=torch.float32))).numpy()
def test_auc(P, Yall, Wall, k): return roc_auc_score(Yall[test_idx, k][Wall[test_idx, k] > 0], P[Wall[test_idx, k] > 0, k])
out = {}
for name, Yall, Wall, heads, main in [("3-head", Y, W.astype(np.float32), [0, 1, 2], [0, 1, 2]), ("5-head ordinal", Y5, W5, [2, 3, 4], [2, 3, 4])]:
    t0 = time.time(); Ps = [train(s, Yall, Wall, heads) for s in range(5)]; Pm = np.mean(Ps, 0)
    a1 = [test_auc(Ps[0], Yall, Wall, k) for k in main]; a5 = [test_auc(Pm, Yall, Wall, k) for k in main]
    print(f"{name:16s} single seed {' '.join(f'{x:.3f}' for x in a1)} | 5-seed ensemble {' '.join(f'{x:.3f}' for x in a5)}  ({time.time()-t0:.0f}s)", flush=True)
    out[name] = dict(single=a1, ens5=a5)
json.dump(out, open("data/v3_results.json", "w")); print("saved data/v3_results.json")
