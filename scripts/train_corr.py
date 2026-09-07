#!/usr/bin/env python
"""Correction network: cheap fixed-step leapfrog + MLP that predicts the residual
to the IAS15 reference over one output interval (dt_out = times[1]-times[0]).

Compared at evaluation: leapfrog alone, leapfrog+NN, both rolled out
autoregressively from t=0 on held-out trajectories, against the IAS15 truth.
"""
import argparse, time, json
import numpy as np
import torch, torch.nn as nn

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/three_body_3d_10k_T30.npz")
ap.add_argument("--substeps", type=int, default=20)
ap.add_argument("--epochs", type=int, default=15)
ap.add_argument("--hidden", type=int, default=256)
ap.add_argument("--batch", type=int, default=4096)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--ntest", type=int, default=500)
ap.add_argument("--out", default="data/corr_results.json")
ap.add_argument("--device", default="cpu")  # MPS unavailable on macOS 12
a = ap.parse_args()
dev = torch.device(a.device)
torch.manual_seed(0); np.random.seed(0)

# ---------------- data ----------------
d = np.load(a.data)
times = d["times"]; dt = float(times[1] - times[0])
ok = d["status"] == 0
pos, vel, m = d["pos"][ok], d["vel"][ok], d["masses"][ok]
t_div = d["t_div"][ok]
N, T = pos.shape[:2]
print(f"trajectories ok={N}/{len(ok)}  T={T} steps dt={dt}")
perm = np.random.permutation(N)
test_idx, train_idx = perm[:a.ntest], perm[a.ntest:]

# ---------------- leapfrog (vectorised, torch) ----------------
def accel(x, m):
    # x: (B,3,3), m: (B,3) -> (B,3,3)
    r = x[:, None, :, :] - x[:, :, None, :]          # r_ij = x_j - x_i
    d2 = (r ** 2).sum(-1) + torch.eye(3, device=x.device)  # avoid /0 on diagonal
    inv3 = d2.pow(-1.5) * (1 - torch.eye(3, device=x.device))
    return (r * (m[:, None, :] * inv3)[..., None]).sum(2)

def leapfrog(x, v, m, h, n):
    a_ = accel(x, m)
    for _ in range(n):
        v = v + 0.5 * h * a_
        x = x + h * v
        a_ = accel(x, m)
        v = v + 0.5 * h * a_
    return x, v

h = dt / a.substeps

def state(x, v):  # (B,3,3),(B,3,3) -> (B,18)
    return torch.cat([x.reshape(len(x), -1), v.reshape(len(v), -1)], 1)

# ---------------- build training pairs ----------------
def pairs(idx):
    X0 = torch.tensor(pos[idx, :-1], dtype=torch.float32).reshape(-1, 3, 3)
    V0 = torch.tensor(vel[idx, :-1], dtype=torch.float32).reshape(-1, 3, 3)
    X1 = torch.tensor(pos[idx, 1:], dtype=torch.float32).reshape(-1, 3, 3)
    V1 = torch.tensor(vel[idx, 1:], dtype=torch.float32).reshape(-1, 3, 3)
    M = torch.tensor(m[idx], dtype=torch.float32).repeat_interleave(T - 1, 0)
    return X0, V0, X1, V1, M

X0, V0, X1, V1, M = pairs(train_idx)
with torch.no_grad():
    XL, VL = leapfrog(X0, V0, M, h, a.substeps)
inp = torch.cat([M, state(XL, VL)], 1)               # 21 features
tgt = state(X1, V1) - state(XL, VL)                  # residual, 18
print(f"train pairs {len(inp)}  |residual| median {tgt.norm(dim=1).median():.2e} "
      f"p99 {tgt.norm(dim=1).quantile(0.99):.2e}")
in_mu, in_sd = inp.mean(0), inp.std(0) + 1e-8
tg_sd = tgt.std(0) + 1e-12

# ---------------- model ----------------
class Net(nn.Module):
    def __init__(s, hdim):
        super().__init__()
        s.f = nn.Sequential(nn.Linear(21, hdim), nn.SiLU(), nn.Linear(hdim, hdim), nn.SiLU(),
                            nn.Linear(hdim, hdim), nn.SiLU(), nn.Linear(hdim, 18))
    def forward(s, z):
        return s.f((z - s.mu) / s.sd) * s.tsd
net = Net(a.hidden).to(dev)
net.mu, net.sd, net.tsd = in_mu.to(dev), in_sd.to(dev), tg_sd.to(dev)
opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
inp_d, tgt_d = inp.to(dev), tgt.to(dev)
nb = len(inp_d) // a.batch
t0 = time.time()
for ep in range(a.epochs):
    p = torch.randperm(len(inp_d), device=dev)
    tot = 0.0
    for b in range(nb):
        i = p[b * a.batch:(b + 1) * a.batch]
        pred = net(inp_d[i])
        loss = nn.functional.huber_loss(pred / net.tsd, tgt_d[i] / net.tsd, delta=1.0)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item()
    sched.step()
    print(f"epoch {ep+1}/{a.epochs} loss {tot/nb:.4e}  {time.time()-t0:.0f}s", flush=True)
train_time = time.time() - t0

# ---------------- rollout evaluation ----------------
@torch.no_grad()
def rollout(idx, use_nn, device):
    x = torch.tensor(pos[idx, 0], dtype=torch.float32, device=device)
    v = torch.tensor(vel[idx, 0], dtype=torch.float32, device=device)
    mm = torch.tensor(m[idx], dtype=torch.float32, device=device)
    out = [x.clone()]
    for k in range(T - 1):
        x, v = leapfrog(x, v, mm, h, a.substeps)
        if use_nn:
            r = net.to(device)(torch.cat([mm, state(x, v)], 1))
            x = x + r[:, :9].reshape(-1, 3, 3); v = v + r[:, 9:].reshape(-1, 3, 3)
        out.append(x.clone())
    return torch.stack(out, 1).cpu().numpy()   # (B,T,3,3)

truth = pos[test_idx]
res = {}
for name, use in [("leapfrog", False), ("leapfrog+nn", True)]:
    t1 = time.time(); P = rollout(test_idx, use, dev); wall = time.time() - t1
    err = np.linalg.norm((P - truth).reshape(len(test_idx), T, -1), axis=2)   # (B,T)
    med = np.nanmedian(err, 0)
    first = [(times[np.where(e > 1e-2)[0][0]] if (e > 1e-2).any() else times[-1]) for e in err]
    res[name] = dict(median_err_vs_t=med.tolist(), t_fail_median=float(np.median(first)),
                     t_fail_mean=float(np.mean(first)),
                     wall_per_traj_batched=wall / len(test_idx))
    print(f"{name:12s} median err @t=5 {med[np.searchsorted(times,5)]:.2e} @t=10 {med[np.searchsorted(times,10)]:.2e} "
          f"@t=30 {med[-1]:.2e} | t_fail(1e-2) median {np.median(first):.2f} | wall/traj batched {wall/len(test_idx)*1e3:.2f} ms")

# single-trajectory CPU timing (fair vs IAS15, which runs one trajectory at a time)
net_cpu = net.to("cpu")
for name, use in [("leapfrog", False), ("leapfrog+nn", True)]:
    t1 = time.time()
    for j in range(10): rollout(test_idx[j:j+1], use, torch.device("cpu"))
    res[name]["wall_per_traj_single_cpu"] = (time.time() - t1) / 10
    print(f"{name:12s} single-traj CPU wall {res[name]['wall_per_traj_single_cpu']*1e3:.1f} ms")
ias = d["wall"][ok][test_idx] / 2   # gen script ran main + shadow sim
res["ias15"] = dict(wall_per_traj_single_cpu=float(np.median(ias)))
print(f"ias15        single-traj CPU wall {np.median(ias)*1e3:.1f} ms (median, from generation log)")

# error grouped by predictability label t_div (intrinsic horizon from 1e-10 perturbation)
td = t_div[test_idx]
groups = {"chaotic (t_div<15)": td < 15, "regular (t_div=30)": td >= 30}
res["groups"] = {}
for name, use in [("leapfrog", False), ("leapfrog+nn", True)]:
    P = rollout(test_idx, use, torch.device("cpu"))
    err = np.linalg.norm((P - truth).reshape(len(test_idx), T, -1), axis=2)
    first = np.array([(times[np.where(e > 1e-2)[0][0]] if (e > 1e-2).any() else times[-1]) for e in err])
    res[name]["t_fail_all"] = first.tolist()
    for g, sel in groups.items():
        res["groups"].setdefault(g, dict(n=int(sel.sum())))[name] = float(np.median(first[sel]))
        print(f"{name:12s} {g:20s} n={sel.sum():3d} t_fail median {np.median(first[sel]):.2f}  (intrinsic t_div median {np.median(td[sel]):.2f})")
res["t_div_test"] = td.tolist()
res.update(dict(train_time=train_time, n_train=len(train_idx), n_test=len(test_idx), epochs=a.epochs,
                substeps=a.substeps, hidden=a.hidden, device=a.device, times=times.tolist()))
torch.save(net_cpu.state_dict(), a.out.replace(".json", ".pt"))
json.dump(res, open(a.out, "w"))
print("saved", a.out)
