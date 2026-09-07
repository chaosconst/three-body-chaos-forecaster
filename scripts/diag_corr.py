import numpy as np, torch, torch.nn as nn, json, sys
sys.argv = ["x"]; 
exec(open("scripts/train_corr.py").read().split("# ---------------- build training pairs")[0].replace("a = ap.parse_args()", "a = ap.parse_args([])"))
# rebuild net and load weights
class Net(nn.Module):
    def __init__(s, hdim):
        super().__init__()
        s.f = nn.Sequential(nn.Linear(21, hdim), nn.SiLU(), nn.Linear(hdim, hdim), nn.SiLU(), nn.Linear(hdim, hdim), nn.SiLU(), nn.Linear(hdim, 18))
    def forward(s, z): return s.f((z - s.mu) / s.sd) * s.tsd
# recompute normalisation exactly as training did (same seed/perm)
def pairs(idx):
    X0 = torch.tensor(pos[idx, :-1], dtype=torch.float32).reshape(-1,3,3); V0 = torch.tensor(vel[idx, :-1], dtype=torch.float32).reshape(-1,3,3)
    X1 = torch.tensor(pos[idx, 1:], dtype=torch.float32).reshape(-1,3,3); V1 = torch.tensor(vel[idx, 1:], dtype=torch.float32).reshape(-1,3,3)
    M = torch.tensor(m[idx], dtype=torch.float32).repeat_interleave(T-1, 0); return X0,V0,X1,V1,M
with torch.no_grad():
    X0,V0,X1,V1,M = pairs(train_idx); XL,VL = leapfrog(X0,V0,M,h,a.substeps)
    inp = torch.cat([M, state(XL,VL)],1); tgt = state(X1,V1)-state(XL,VL)
    net = Net(a.hidden); net.mu, net.sd, net.tsd = inp.mean(0), inp.std(0)+1e-8, tgt.std(0)+1e-12
    net.load_state_dict(torch.load("data/corr_results.pt"), strict=False)
    # held-out one-step test
    X0,V0,X1,V1,M = pairs(test_idx); XL,VL = leapfrog(X0,V0,M,h,a.substeps)
    inp = torch.cat([M, state(XL,VL)],1); tgt = state(X1,V1)-state(XL,VL)
    pred = net(inp)
    r0 = tgt[:, :9].norm(dim=1); r1 = (tgt-pred)[:, :9].norm(dim=1)
    q = lambda r, p: r.quantile(p).item()
    print("held-out one-step position residual (leapfrog vs leapfrog+nn):")
    for p in [0.1, 0.5, 0.9, 0.99]: print(f"  p{int(p*100):2d}: {q(r0,p):.2e} -> {q(r1,p):.2e}")
    print(f"  frac improved: {(r1<r0).float().mean():.3f}; frac>1e-2: {(r0>1e-2).float().mean():.3f} -> {(r1>1e-2).float().mean():.3f}")
    # float64 leapfrog rollout on regular test trajectories: does float32 matter?
    td = t_div[test_idx]; reg = test_idx[td>=30][:60]
    for dtype in [torch.float32, torch.float64]:
        x = torch.tensor(pos[reg,0], dtype=dtype); v = torch.tensor(vel[reg,0], dtype=dtype); mm = torch.tensor(m[reg], dtype=dtype)
        firsts = np.full(len(reg), times[-1]); 
        for k in range(1, T):
            x, v = leapfrog(x, v, mm, h, a.substeps)
            e = (x.numpy() - pos[reg, k]).reshape(len(reg), -1); e = np.linalg.norm(e, axis=1)
            new = (e > 1e-2) & (firsts == times[-1]); firsts[new] = times[k]
        print(f"leapfrog-only rollout, regular group, {str(dtype)[6:]}: t_fail median {np.median(firsts):.2f}")
    # how tight are binaries? min pairwise distance over the dataset per trajectory
    P = pos[test_idx]; dmin = np.full(len(test_idx), np.inf)
    for i,j in [(0,1),(0,2),(1,2)]:
        dmin = np.minimum(dmin, np.nanmin(np.linalg.norm(P[:,:,i]-P[:,:,j],axis=2),axis=1))
    print(f"min pair distance over test trajectories: p10 {np.percentile(dmin,10):.3f} median {np.median(dmin):.3f} p90 {np.percentile(dmin,90):.3f}; substep h={h:.4f}")
