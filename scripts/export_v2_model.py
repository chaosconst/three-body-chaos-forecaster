#!/usr/bin/env python
"""Retrain the v2 protocol MLP (val early-stopped, seed 0) and export weights + big-test predictions for the page."""
import json, numpy as np, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
src = open("scripts/train_v2.py").read()
exec(src[src.index("import json"):src.index("TASKS = ")].replace("from sklearn.ensemble import HistGradientBoostingClassifier\n", ""))
import time
def aucs(P, idx):
    return [roc_auc_score(Y[idx, k][W[idx, k]], P[W[idx, k], k]) for k in range(3)]
class Net(nn.Module):
    def __init__(s, nf, h=128):
        super().__init__(); s.f = nn.Sequential(nn.Linear(nf, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 3))
    def forward(s, z): return s.f(z)
def train_keep(seed, epochs=60):
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
    net.load_state_dict(best_state); net.eval(); return net, best_ep
net, best_ep = train_keep(0)
with torch.no_grad(): Pt = torch.sigmoid(net(torch.tensor(Z[test_idx], dtype=torch.float32))).numpy()
a = aucs(Pt, test_idx); print(f"exported model: best val epoch {best_ep}, test AUC {a[0]:.3f} {a[1]:.3f} {a[2]:.3f}")
layers = [l for l in net.f if isinstance(l, nn.Linear)]
X1 = torch.tensor(Z[test_idx[:1]], dtype=torch.float32)
with torch.no_grad():
    t0 = time.time(); [net(X1) for _ in range(1000)]; single = (time.time() - t0) / 1000
    Xb = torch.tensor(Z[test_idx], dtype=torch.float32); t0 = time.time(); net(Xb); batched = (time.time() - t0) / len(test_idx)
old = json.load(open("data/chaos_model.json"))
export = dict(mu=mu.tolist(), sd=sd.tolist(), W=[l.weight.detach().numpy().tolist() for l in layers], b=[l.bias.detach().numpy().tolist() for l in layers],
              results=old["results"], timing=dict(single_us=single*1e6, batched_us=batched*1e6, ias15_ms=old["timing"]["ias15_ms"]),
              test_pred=Pt.tolist(), test_true=Y[test_idx].tolist(), test_mask=W[test_idx].astype(int).tolist(),
              n_train=len(tr_idx), n_val=len(val_idx), n_test=len(test_idx), best_epoch=best_ep, test_auc=a, protocol="v2")
json.dump(export, open("data/chaos_model_v2.json", "w")); print("saved data/chaos_model_v2.json")
