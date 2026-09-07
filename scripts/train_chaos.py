#!/usr/bin/env python
"""Chaos forecaster: from a three-body initial condition predict
  (1) P(diverges before T)  -- 1e-10 perturbation grows to 1e-3 before t=30
  (2) P(escape by T)        -- a body beyond r=20 at the end (or at truncation)
  (3) P(near collision)     -- two bodies within 1e-3 before T (run truncated)
Inputs: masses, positions, velocities in the CoM frame + physics features.
Training uses SO(3) rotation + body-permutation augmentation (exact symmetries).
"""
import argparse, json, time
import numpy as np, torch, torch.nn as nn

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/three_body_3d_10k_T30.npz", help="comma-separated npz files; test set is drawn from the first")
ap.add_argument("--ntrain", type=int, default=0, help="cap on training systems (0 = all)")
ap.add_argument("--skip_noaug", action="store_true")
ap.add_argument("--eval_every", type=int, default=0, help="print test AUC every k epochs")
ap.add_argument("--skip_logistic", action="store_true")
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--hidden", type=int, default=128)
ap.add_argument("--ntest", type=int, default=1500)
ap.add_argument("--out", default="data/chaos_model.json")
a = ap.parse_args(); torch.manual_seed(0); np.random.seed(0)

files = a.data.split(","); parts = []
for f in files:
    d = np.load(f); POS = d["pos"]; n = len(POS); T = float(d["times"][-1])
    last = (~np.isnan(POS[:, :, 0, 0])).sum(1) - 1
    parts.append(dict(m=d["masses"], pos=POS[:, 0], vel=d["vel"][:, 0], st=d["status"], t_div=d["t_div"], wall=d["wall"],
                      rlast=np.linalg.norm(POS[np.arange(n), last], axis=2).max(1), n=n))
cat = lambda k: np.concatenate([p[k] for p in parts])
m, pos, vel, st, rlast, t_div, wall = cat("m"), cat("pos"), cat("vel"), cat("st"), cat("rlast"), cat("t_div"), cat("wall")
N = len(m); N0 = parts[0]["n"]; d = {"wall": wall}
y_col = (st == 1).astype(np.float32)
y_esc = (rlast > 20).astype(np.float32)
y_div = ((t_div < T) & (st == 0)).astype(np.float32)   # only defined for complete runs
mask_div = (st == 0).astype(np.float32)
print(f"N={N}  P(collision)={y_col.mean():.3f}  P(escape)={y_esc.mean():.3f}  P(diverge|complete)={y_div[st==0].mean():.3f}")

def features(m, x, v):
    """m (B,3), x (B,3,3), v (B,3,3) -> (B,F). Rotation/permutation-invariant physics + raw."""
    B = len(m)
    KE = 0.5 * (m * (v ** 2).sum(-1)).sum(1)
    r = np.stack([np.linalg.norm(x[:, i] - x[:, j], axis=1) for i, j in [(0, 1), (0, 2), (1, 2)]], 1)
    PE = -(m[:, 0] * m[:, 1] / r[:, 0] + m[:, 0] * m[:, 2] / r[:, 1] + m[:, 1] * m[:, 2] / r[:, 2])
    E = KE + PE
    L = np.linalg.norm((m[..., None] * np.cross(x, v)).sum(1), axis=1)
    Q = KE / np.abs(PE)
    # pair binding energies (which pair is closest to bound binary)
    pair_e = []
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        mu = m[:, i] * m[:, j] / (m[:, i] + m[:, j]); dv = np.linalg.norm(v[:, i] - v[:, j], axis=1)
        pair_e.append(0.5 * mu * dv ** 2 - m[:, i] * m[:, j] / np.linalg.norm(x[:, i] - x[:, j], axis=1))
    pair_e = np.stack(pair_e, 1)
    rdot = np.stack([((x[:, i] - x[:, j]) * (v[:, i] - v[:, j])).sum(1) for i, j in [(0, 1), (0, 2), (1, 2)]], 1)
    phys = np.concatenate([E[:, None], L[:, None], Q[:, None], np.sort(r, 1), np.sort(pair_e, 1), np.sort(rdot, 1),
                           np.sort(m, 1), r.min(1, keepdims=True), np.log(r.min(1, keepdims=True))], 1)
    raw = np.concatenate([m, x.reshape(B, -1), v.reshape(B, -1)], 1)
    return np.concatenate([phys, raw], 1).astype(np.float32)

def rand_rot(B):
    q = np.random.normal(size=(B, 4)); q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([np.stack([1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], 1),
                     np.stack([2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], 1),
                     np.stack([2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)], 1)], 1)

def augment(m, x, v):
    R = rand_rot(len(m)); x = np.einsum("bij,bkj->bki", R, x); v = np.einsum("bij,bkj->bki", R, v)
    perm = np.array([np.random.permutation(3) for _ in range(len(m))])
    idx = np.arange(len(m))[:, None]
    return m[idx, perm], x[idx, perm], v[idx, perm]

perm = np.random.permutation(N0); te = perm[:a.ntest]; tr = np.concatenate([perm[a.ntest:], np.arange(N0, N)])
if a.ntrain: tr = tr[:a.ntrain]
print(f"train {len(tr)}  test {len(te)}")
F0 = features(m, pos, vel); nf = F0.shape[1]
mu, sd = F0[tr].mean(0), F0[tr].std(0) + 1e-6
Y = np.stack([y_div, y_esc, y_col], 1); W = np.stack([mask_div, np.ones(N), np.ones(N)], 1).astype(np.float32)

def auc(y, p):
    o = np.argsort(p); r = np.empty(len(p)); r[o] = np.arange(1, len(p) + 1)
    npos = y.sum(); nneg = len(y) - npos
    return (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)
class Net(nn.Module):
    def __init__(s, nf, h):
        super().__init__()
        s.f = nn.Sequential(nn.Linear(nf, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 3))
    def forward(s, z): return s.f(z)
def run(model, aug, epochs, tag):
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    Yt, Wt = torch.tensor(Y[tr]), torch.tensor(W[tr]); t0 = time.time()
    for ep in range(epochs):
        model.train()
        mm, xx, vv = (augment(m[tr], pos[tr], vel[tr]) if aug else (m[tr], pos[tr], vel[tr]))
        X = torch.tensor((features(mm, xx, vv) - mu) / sd)
        p = torch.randperm(len(tr)); tot = 0
        for b in range(0, len(tr), 256):
            i = p[b:b+256]; logit = model(X[i])
            loss = (nn.functional.binary_cross_entropy_with_logits(logit, Yt[i], reduction="none") * Wt[i]).sum() / Wt[i].sum()
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(i)
        sched.step()
        if (ep + 1) % 10 == 0 or ep == epochs - 1:
            print(f"[{tag}] epoch {ep+1}/{epochs} loss {tot/len(tr):.4f} {time.time()-t0:.0f}s", flush=True)
        if a.eval_every and ((ep + 1) % a.eval_every == 0 or ep == epochs - 1):
            model.eval()
            with torch.no_grad(): Pt = torch.sigmoid(model(torch.tensor((F0[te] - mu) / sd))).numpy()
            aucs = [auc(Y[te, k][W[te, k] > 0], Pt[W[te, k] > 0, k]) for k in range(3)]
            print(f"[{tag}] EVAL epoch {ep+1} train_loss {tot/len(tr):.4f} test_auc " + " ".join(f"{x:.3f}" for x in aucs), flush=True)
    return model

def auc(y, p):
    o = np.argsort(p); r = np.empty(len(p)); r[o] = np.arange(1, len(p) + 1)
    npos = y.sum(); nneg = len(y) - npos
    return (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)
def evaluate(model, tag):
    model.eval()
    with torch.no_grad():
        P = torch.sigmoid(model(torch.tensor((F0[te] - mu) / sd))).numpy()
    res = {}
    for k, name in enumerate(["diverge", "escape", "collision"]):
        sel = W[te, k] > 0; y, p = Y[te, k][sel], P[sel, k]
        acc = ((p > 0.5) == (y > 0.5)).mean(); base = max(y.mean(), 1 - y.mean())
        res[name] = dict(auc=float(auc(y, p)), acc=float(acc), majority=float(base), n=int(sel.sum()))
        print(f"[{tag}] {name:9s} n={sel.sum():4d}  AUC {auc(y,p):.3f}  acc {acc:.3f}  (majority {base:.3f})")
    return res, P

results = {}
if not a.skip_logistic:
    lin = nn.Sequential(nn.Linear(nf, 3)); run(lin, False, 40, "logistic"); results["logistic"], _ = evaluate(lin, "logistic")
if not a.skip_noaug:
    net0 = Net(nf, a.hidden); run(net0, False, a.epochs, "mlp-noaug"); results["mlp_noaug"], _ = evaluate(net0, "mlp-noaug")
net = Net(nf, a.hidden); run(net, True, a.epochs, "mlp-aug"); results["mlp_aug"], P = evaluate(net, "mlp-aug")

# timing: batched and single
X1 = torch.tensor((F0[:1] - mu) / sd); net.eval()
with torch.no_grad():
    t0 = time.time(); [net(X1) for _ in range(1000)]; single = (time.time() - t0) / 1000
    Xb = torch.tensor((F0 - mu) / sd); t0 = time.time(); net(Xb); batched = (time.time() - t0) / N
print(f"inference: single {single*1e6:.0f} us, batched {batched*1e6:.2f} us/system; IAS15+shadow truth {np.median(d['wall'])*1e3:.0f} ms/system")

# export weights for the browser
layers = [l for l in net.f if isinstance(l, nn.Linear)]
export = dict(mu=mu.tolist(), sd=sd.tolist(), W=[l.weight.detach().numpy().tolist() for l in layers], b=[l.bias.detach().numpy().tolist() for l in layers],
              results=results, timing=dict(single_us=single*1e6, batched_us=batched*1e6, ias15_ms=float(np.median(d["wall"])*1e3)),
              test_ids=te.tolist(), test_pred=P.tolist(), test_true=Y[te].tolist(), test_mask=W[te].tolist(), n_train=len(tr), n_test=len(te), epochs=a.epochs, hidden=a.hidden)
json.dump(export, open(a.out, "w")); print("saved", a.out)
