# Three-body trajectory datasets — inventory

Downloaded 2026-09-06 into `data/`. Total on disk: ~107 MB.

| Dataset | Directory | Status | Size on disk |
|---|---|---|---|
| Breen et al. 2020 "Newton vs the machine" (NVM) | `data/NVM` | OK (git clone --depth 1) | 61 MB (19 MB of it is `.git`) |
| Zenodo 10.5281/zenodo.18403805 (arXiv 2601.09843) | `data/zenodo_18403805` | **FAILED — record does not resolve (404)** | 0 |
| SJTU Liao group periodic orbits | `data/sjtu_liao_three_body` | Partial: repo is ~748 MB (>200 MB), so NOT cloned; the non-image data/text/code/paper files (~46 MB) were fetched individually | 46 MB |

---

## 1. NVM — Breen, Foley, Boekholt & Portegies Zwart (2020), "Newton vs the machine"

- **Source:** https://github.com/pgbreen/NVM (paper: https://arxiv.org/abs/1910.07291)
- **Contents:** Brutus (arbitrary-precision) integrations of the planar, equal-mass (m=1, G=1), zero-initial-velocity three-body problem, plus the trained Keras network (`NN.h5`) and a script (`nvm.py`) that evaluates it. Only a 180-trajectory sample of the paper's 10 000 runs is in the repo.
- **Initial-condition family (verified from the data):** body 1 at (1, 0); body 2 at (x, y) with x ∈ (−0.5, 0), y ∈ (0, 1) (observed ranges x ∈ [−0.498, −0.007], y ∈ [0.009, 0.914]); body 3 at −(r1 + r2) so the centre of mass is at the origin; all velocities zero. Because of this symmetry only the position of body 2 varies between runs; `nvm.py` samples it uniformly in that rectangle intersected with the unit disk.
- **Files:**
  - `trainex/data/run10.dat … run99.dat` — 90 training-example trajectories (run ids 10–99).
  - `validex/data/run9910.dat … run9999.dat` — 90 validation-example trajectories (run ids 9910–9999).
  - `trainex/*.pdf`, `validex/*.pdf` — plots of the trajectories, 9 runs per PDF.
  - `NN.h5` (1.9 MB) — trained network: input (t, x, y) → output (x1, y1, x2, y2).
  - `output.dat` / `output.pdf` — example network output, 500 rows × 6 columns (x1 y1 x2 y2 x3 y3), t ∈ [0, 3.9].
- **Format of `run*.dat`:** whitespace-separated ASCII, 4 columns, **no time column**:

  | col | meaning |
  |---|---|
  | 0 | x of body 1 |
  | 1 | y of body 1 |
  | 2 | x of body 2 |
  | 3 | y of body 2 |

  Body 3 is recovered as `r3 = -(r1 + r2)`. Rows are uniformly spaced in time with **dt = 1/256 = 0.00390625** (checked: the first displacements grow as 1:3:5 and dt² = 2·|Δr|/|a| from the Newtonian acceleration gives 0.0039062 for both bodies), so `t = row_index / 256`. A full run has **2561 rows, t ∈ [0, 10]**.
- **Truncated runs:** 64/90 training and 72/90 validation files have the full 2561 rows. The remaining 26 + 18 files stop early (from 5 rows up to 2547 rows), always heading into a close encounter (e.g. `run96.dat` stops after 5 rows with bodies 2–3 at separation 0.028) — Brutus apparently aborted at collisions / failed convergence. The paper itself only uses t ≤ 3.9 (rows 0–998) for training.
- **Load one sample:**

```python
import numpy as np
a = np.loadtxt('data/NVM/trainex/data/run10.dat')   # shape (2561, 4)
t  = np.arange(a.shape[0]) / 256.0                   # time, dt = 1/256
r1 = a[:, 0:2]; r2 = a[:, 2:4]; r3 = -(r1 + r2)       # positions of the 3 bodies
print(a.shape); print(a[:3])
# [[ 1.0000e+00 -0.0000e+00 -1.6916e-01  1.9183e-02]
#  [ 9.9999e-01  6.7697e-08 -1.6917e-01  1.9182e-02]
#  [ 9.9997e-01  2.7078e-07 -1.6920e-01  1.9179e-02]]
```

---

## 2. Zenodo 10.5281/zenodo.18403805 — FAILED

- **Source:** https://doi.org/10.5281/zenodo.18403805 , cited in the "Data availability" section of arXiv 2601.09843 (*"The formation of periodic three-body orbits for Newtonian systems"*, Portegies Zwart et al., A&A, doi 10.1051/0004-6361/202558230): "All runscripts and data generated for this paper are available on Zenodo under doi 10.5281/zenodo.18403805."
- **What happened:** `https://zenodo.org/api/records/18403805` returns HTTP 404 `"The persistent identifier is not registered."`; `https://zenodo.org/records/18403805` and the doi.org redirect also give 404. Neighbouring id 18403806 resolves to an unrelated record, so the id is valid-looking but the deposit was never published (or is still a private draft). A Zenodo search for the arXiv id, for "braid three-body", and for author Portegies Zwart found no matching record. Nothing downloaded; `data/zenodo_18403805` was not created.
- **Note:** the paper is about *braids* (periodic N-body orbits) formed/dissolved in 4-body encounters simulated with AMUSE, not a periodic-orbit catalogue. A possibly useful alternative from the same group is Zenodo 11032352 "Flow map data of the single pendulum, double pendulum and 3-body problem" (not downloaded).

---

## 3. SJTU Liao group — planar and 3D periodic three-body orbits (partial)

- **Source:** https://github.com/sjtu-liao/three-body (Xiaoming Li & Shijun Liao). Papers included in `paper/` (2017 SCPMA, 2018 PASJ, 2019 New Astronomy, 2021 SCPMA, 2022 New Astronomy).
- **Why not cloned:** GitHub reports the repo at ~710 MB packed / 748 MB of blobs: 4721 PNG orbit pictures (154 MB), 47 GIFs (372 MB), four Keras `.h5` models (179 MB). Everything else (catalogue text files, tables, code, PDFs; 23 files, 46 MB) was downloaded file-by-file from `raw.githubusercontent.com` into `data/sjtu_liao_three_body/` with the same relative paths. **Not downloaded:** `data/*.png`, `unequal-mass-data/*.png`, `free-fall-data/*.png`, `readme_gif/`, `roadmap/Model/*.h5`.
- **Contents are catalogues of initial conditions + periods, not time series.** To get a trajectory you integrate the initial condition yourself (G = 1). Files:

  | File | Orbits | Format / columns |
  |---|---|---|
  | `three-body-free-group-word.md` | 695 planar equal-mass (m=1) families; r = (−1,0),(1,0),(0,0); v = (v1,v2),(v1,v2),(−2v1,−2v2) | HTML table: class/number, free-group word. Periods and (v1,v2) are in the companion `three-body-pictures.md` |
  | `three-body-unequal-mass-free-group-word.md` / `-pictures.md` | 1349 planar unequal-mass families | HTML tables |
  | `free-fall-3b-free-group-word.md` / `-pictures.md` | 316 collisionless free-fall orbits | HTML tables |
  | `non-hierarchical-3b-supplementary_data.txt` | 135 445 unequal-mass non-hierarchical periodic orbits (13 315 stable) | 7-line header then whitespace columns: `m1 m2 m3 x1 v1 v2 T stability(S/U)`; r1=(x1,0), r2=(1,0), r3=(0,0), v1=(0,v1), v2=(0,v2), v3=(0,−(m1v1+m2v2)/m3) |
  | `initial-condition-of-3D-periodic-orbits.txt` | 10 059 3D periodic orbits (m1=m2=1, m3 varies) | 6-line header then `O_{index}(m3) z0 vx vy vz T stability`; r1=(−1,0,0), r2=(1,0,0), r3=(0,0,z0); v1=(vx,vy,vz), v2=(vx,vy,−vz), v3=(−2vx/m3, −2vy/m3, 0) |
  | `initial-condition-of-piano-trio-orbits-...txt` | 273 "piano-trio" 3D orbits | same layout as above |
  | `topological-sequences-of-3D-periodic-orbits.txt` | 10 059 | `O_{n}(m3) <tab> topological sequence <tab> reduced sequence` |
  | `roadmap/Data/case1_periodic_orbit.dat` | 29 150 rows | 8 cols: `m1 m2 m3 x1 v1 v2 T ?` (BHH-satellite family; training data for the 2022 ANN paper) |
  | `roadmap/Data/case2_periodic_orbit.dat` | 35 895 rows | 9 cols: `m1 m2 m3 x1 v1 v2 T ? flag(0/1)` |
  | `roadmap/Data/case{1,2}_classification.dat` | 32 278 / 41 795 rows | `m1 m2 onehot(3)` orbit-type classification labels |
  | `roadmap/Code/*.py` | — | Keras scripts used with the (not downloaded) `roadmap/Model/*.h5` |

- **Load one sample (non-hierarchical catalogue):**

```python
import numpy as np
rows = np.genfromtxt('data/sjtu_liao_three_body/non-hierarchical-3b-supplementary_data.txt',
                     skip_header=7, dtype=None, encoding='utf-8')
m1, m2, m3, x1, v1, v2, T, stab = rows[0]          # e.g. 0.8 0.75 1.0 -0.1324 2.543 0.3066 5.038 'U'
# initial state (G = 1): r1=(x1,0) r2=(1,0) r3=(0,0); v1=(0,v1) v2=(0,v2) v3=(0,-(m1*v1+m2*v2)/m3)
```

  and for the 3D catalogue:

```python
import numpy as np
d = np.loadtxt('data/sjtu_liao_three_body/initial-condition-of-3D-periodic-orbits.txt',
               skiprows=6, dtype=str)               # shape (10059, 7)
name, z0, vx, vy, vz, T, stab = d[0]                 # 'O_{1}(0.1)' ... ; m3 is inside the name
```
