# Export a compact JSON for the web page: stats over all trajectories + a sample of full trajectories.
import numpy as np, json, sys
d = np.load(sys.argv[1]); out = sys.argv[2]
times = d["times"]; st = d["status"]; ok = st == 0
t_div = d["t_div"]; e_err = d["e_err"]; T = float(times[-1])
rng = np.random.default_rng(0)
# sample: 16 chaotic (t_div<12), 16 mid, 16 regular (t_div==T), all ok
cats = {"chaotic": np.where(ok & (t_div < 12))[0], "mid": np.where(ok & (t_div >= 12) & (t_div < T))[0], "regular": np.where(ok & (t_div >= T))[0]}
sample = []
for c, idx in cats.items():
    for i in rng.choice(idx, 16, replace=False):
        sample.append(dict(id=int(i), cat=c, m=np.round(d["masses"][i], 3).tolist(), t_div=float(t_div[i]),
            e_err=float(e_err[i]), pos=np.round(d["pos"][i], 4).tolist(), sep=[float(f"{s:.3g}") for s in d["sep"][i]]))
hist_td = np.histogram(t_div[ok], bins=np.arange(0, T + 1, 1))
hist_ee = np.histogram(np.log10(np.clip(e_err[ok], 1e-17, 1)), bins=np.arange(-17, -6, 0.5))
# escape fraction at end: any body farther than 20 from com
p = d["pos"][ok][:, -1]; esc = (np.linalg.norm(p, axis=2).max(1) > 20).mean()
js = dict(n=int(len(st)), n_ok=int(ok.sum()), n_ce=int((st == 1).sum()), T=T, nout=len(times), dt=float(times[1]-times[0]),
  t_div_median=float(np.median(t_div[ok])), frac_div=float((t_div[ok] < T).mean()),
  e_err_median=float(np.median(e_err[ok])), e_err_max=float(e_err[ok].max()), esc_frac=float(esc),
  wall_median=float(np.median(d["wall"])), t_end_ce=np.round(d["t_end"][st==1],2).tolist(),
  hist_td=dict(counts=hist_td[0].tolist(), edges=hist_td[1].tolist()),
  hist_ee=dict(counts=hist_ee[0].tolist(), edges=hist_ee[1].tolist()),
  mass_range=[float(d["masses"].min()), float(d["masses"].max())], sample=sample, times=np.round(times,3).tolist())
json.dump(js, open(out, "w"), separators=(",", ":"))
print(out, "bytes", len(open(out).read()), {k: js[k] for k in ["n","n_ok","n_ce","t_div_median","frac_div","e_err_median","e_err_max","esc_frac","wall_median"]})
