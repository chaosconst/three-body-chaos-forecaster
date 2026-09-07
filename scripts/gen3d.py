#!/usr/bin/env python
"""Generate random 3D three-body trajectories with REBOUND IAS15 (G=1).

Each trajectory also gets a shadow run with a 1e-10 position perturbation;
the time at which the two runs separate by more than DIV_THRESH is stored
as t_div, a cheap per-trajectory predictability-horizon label.
"""
import argparse, time, sys
import numpy as np
import rebound
from multiprocessing import Pool

PERTURB = 1e-10
DIV_THRESH = 1e-3
MIN_DIST = 1e-3   # stop if two bodies get closer than this (point-mass near-collision)


def sample_ic(rng, mmin, mmax, q_lo, q_hi):
    m = rng.uniform(mmin, mmax, size=3)
    # positions uniform in unit sphere
    while True:
        x = rng.uniform(-1, 1, size=(3, 3))
        if np.all(np.linalg.norm(x, axis=1) <= 1):
            break
    v = rng.normal(size=(3, 3))
    # centre of mass frame
    x -= (m[:, None] * x).sum(0) / m.sum()
    v -= (m[:, None] * v).sum(0) / m.sum()
    # scale velocities to target virial ratio Q = T/|U|
    U = 0.0
    for i in range(3):
        for j in range(i + 1, 3):
            U -= m[i] * m[j] / np.linalg.norm(x[i] - x[j])
    T = 0.5 * (m * (v ** 2).sum(1)).sum()
    Q = rng.uniform(q_lo, q_hi)
    v *= np.sqrt(Q * abs(U) / T)
    return m, x, v


def make_sim(m, x, v):
    sim = rebound.Simulation()
    sim.G = 1.0
    sim.integrator = "ias15"
    sim.exit_min_distance = MIN_DIST
    for i in range(3):
        sim.add(m=m[i], x=x[i, 0], y=x[i, 1], z=x[i, 2],
                vx=v[i, 0], vy=v[i, 1], vz=v[i, 2])
    return sim


def run_one(args):
    seed, T, nout, mmin, mmax, q_lo, q_hi = args
    rng = np.random.default_rng(seed)
    m, x, v = sample_ic(rng, mmin, mmax, q_lo, q_hi)
    times = np.linspace(0, T, nout)
    pos = np.full((nout, 3, 3), np.nan)
    vel = np.full((nout, 3, 3), np.nan)
    sep = np.full(nout, np.nan)
    status = 0
    t_end = T

    sim = make_sim(m, x, v)
    dx = rng.normal(size=(3, 3)); dx *= PERTURB / np.linalg.norm(dx)
    sim2 = make_sim(m, x + dx, v)
    E0 = sim.energy()
    t0 = time.time()
    for k, t in enumerate(times):
        try:
            sim.integrate(t, exact_finish_time=1)
            sim2.integrate(t, exact_finish_time=1)
        except rebound.Encounter:
            status = 1; t_end = sim.t; break
        p = np.array([[pp.x, pp.y, pp.z] for pp in sim.particles])
        p2 = np.array([[pp.x, pp.y, pp.z] for pp in sim2.particles])
        pos[k] = p
        vel[k] = np.array([[pp.vx, pp.vy, pp.vz] for pp in sim.particles])
        sep[k] = np.linalg.norm(p - p2)
    e_err = abs((sim.energy() - E0) / E0)
    above = np.where(sep > DIV_THRESH)[0]
    t_div = times[above[0]] if len(above) else T
    return dict(seed=seed, m=m, pos=pos, vel=vel, sep=sep, status=status,
                t_end=t_end, e_err=e_err, t_div=t_div, wall=time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--T", type=float, default=10.0)
    ap.add_argument("--nout", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--mmin", type=float, default=0.5)
    ap.add_argument("--mmax", type=float, default=1.5)
    ap.add_argument("--qlo", type=float, default=0.1)
    ap.add_argument("--qhi", type=float, default=0.5)
    ap.add_argument("--out", default="data/three_body_3d.npz")
    a = ap.parse_args()

    jobs = [(a.seed + i, a.T, a.nout, a.mmin, a.mmax, a.qlo, a.qhi) for i in range(a.n)]
    res = []
    t0 = time.time()
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap(run_one, jobs, chunksize=4)):
            res.append(r)
            if (i + 1) % max(1, a.n // 20) == 0 or i + 1 == a.n:
                print(f"{i+1}/{a.n}  elapsed {time.time()-t0:.0f}s", flush=True)

    out = dict(
        times=np.linspace(0, a.T, a.nout),
        seeds=np.array([r["seed"] for r in res]),
        masses=np.array([r["m"] for r in res]),
        pos=np.array([r["pos"] for r in res]),
        vel=np.array([r["vel"] for r in res]),
        sep=np.array([r["sep"] for r in res]),
        status=np.array([r["status"] for r in res]),
        t_end=np.array([r["t_end"] for r in res]),
        e_err=np.array([r["e_err"] for r in res]),
        t_div=np.array([r["t_div"] for r in res]),
        wall=np.array([r["wall"] for r in res]),
        meta=np.array(str(vars(a))),
    )
    np.savez_compressed(a.out, **out)
    st = out["status"]; ok = st == 0
    print(f"saved {a.out}: n={a.n} ok={ok.sum()} close_encounter={(st==1).sum()}")
    print(f"energy err (ok): median {np.median(out['e_err'][ok]):.2e} max {out['e_err'][ok].max():.2e}")
    print(f"t_div: median {np.median(out['t_div']):.2f} min {out['t_div'].min():.2f} frac<T {np.mean(out['t_div']<a.T):.2f}")
    print(f"wall per traj: median {np.median(out['wall']):.3f}s max {out['wall'].max():.2f}s")


if __name__ == "__main__":
    main()
