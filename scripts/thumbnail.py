import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
W = json.load(open(__import__("sys").argv[1])); out = __import__("sys").argv[2]
s = [x for x in W["sample"] if x["cat"] == "chaotic"][0]
P = np.array(s["pos"]); n = np.isfinite(P[:, 0, 0]).sum(); P = P[:n]
fig = plt.figure(figsize=(12, 8), dpi=100); fig.patch.set_facecolor("#0F1424")
ax = fig.add_axes([0, 0, 1, 1], projection="3d"); ax.set_facecolor("#0F1424"); ax.set_axis_off()
cols = ["#F2B441", "#F0715C", "#7FD3E6"]
for b in range(3):
    ax.plot(P[:, b, 0], P[:, b, 1], P[:, b, 2], color=cols[b], lw=1.2, alpha=0.8)
    ax.scatter(*P[-1, b], color=cols[b], s=60 + 40 * s["m"][b], edgecolor="#0F1424", zorder=5)
c = P.reshape(-1, 3); r = np.percentile(np.linalg.norm(c - c.mean(0), axis=1), 92); m = c.mean(0)
ax.set_xlim(m[0]-r, m[0]+r); ax.set_ylim(m[1]-r, m[1]+r); ax.set_zlim(m[2]-r, m[2]+r); ax.view_init(22, 35)
fig.text(0.05, 0.90, "Three-Body Chaos Forecaster", color="#E9ECF5", fontsize=30, fontweight="bold", family="DejaVu Sans")
fig.text(0.05, 0.84, "Will this system stay predictable?  A neural net answers in 60 µs.  The reference run takes 150 ms.", color="#9AA3C0", fontsize=14)
fig.text(0.05, 0.07, f"sample #{s['id']}  ·  t_div = {s['t_div']:.1f}  ·  IAS15, G = 1, t ≤ 30  ·  10 000 random 3D systems", color="#9AA3C0", fontsize=12, family="DejaVu Sans Mono")
fig.savefig(out, facecolor=fig.get_facecolor()); print("saved", out)
