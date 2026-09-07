# Three-Body Chaos Forecaster

**Microsecond neural forecasts of chaos, escape and collision for random 3D three-body systems.**

Live demo: see the project page (artifact link). Built at RSI Hack #3: co-scientist, Sundai Club San Francisco, 2026-09-06.

## The question

Given only the initial condition of a three-body system (three masses, positions, velocities), can a network tell you, before you integrate anything, whether the system will:

1. **diverge** — become unpredictable (a 1e-10 nudge grows to 1e-3) before t = 30,
2. **eject a body** (a body beyond r = 20 at the end),
3. **suffer a near collision** (two bodies within 1e-3)?

The reference answer costs a full IAS15 integration plus a shadow run, about 150 ms per system on one core. The network answers in about 60 µs, or 1 µs batched.

## What we built

- `scripts/gen3d.py` — dataset generator. REBOUND IAS15, G = 1, masses U(0.5, 1.5), positions uniform in the unit sphere, isotropic velocities scaled to virial ratio U(0.1, 0.5), integrated to t = 30. Every run has a shadow run with a 1e-10 perturbation, giving a per-system predictability horizon `t_div`. 153 000 systems generated in total.
- `scripts/train_chaos.py` — the forecaster: 38 physics + raw features → MLP (3 hidden layers) → 3 logits, trained with exact SO(3) rotation and body-permutation augmentation. Includes logistic-regression and single-feature rule baselines.
- `scripts/train_short.py` — SPOCK-style variant: integrate to t = 3 first (10 % of the cost), then forecast.
- `scripts/train_corr.py` — a first attempt (leapfrog + correction network) that failed; kept as a record.
- The web page runs the trained MLP in the browser and contains a Dormand-Prince 5(4) reference integrator (tol 1e-13 + shadow run), so a freshly drawn or pasted system is forecast and then checked live.

## Results (AUC, held-out systems)

| model | cost | diverge | escape | collision |
|---|---|---|---|---|
| best single physical rule | 0 | 0.675 | 0.654 | 0.672 |
| logistic regression, physics features | 0 | 0.730 | 0.723 | 0.714 |
| MLP + symmetry augmentation (page model) | 0 | 0.769 | 0.768 | 0.731 |
| same model, 3 000 fresh never-touched systems | 0 | 0.772 | 0.778 | 0.696 |
| integrate to t = 3, then logistic regression | 10 % | 0.823 | 0.826 | 0.817 |

What we learned, in order:

1. **Zero initial velocity is not a 3D problem.** Three points share a plane; with no velocity the motion never leaves it.
2. **A one-step correction network does not work here.** The median closest approach is 0.05, so a binary completes two orbits inside one output step. The network added noise to 93 % of steps.
3. **More data does not move the zero-cost ceiling.** Same test set, training set grown 2 000 → 48 500 → 148 500: peak AUC 0.77–0.78 throughout (+0.005 for 3× data, inside test-set noise).
4. **Longer training overfits, width does not matter.** Per-epoch test AUC peaks at epoch 10–40 for both 128- and 384-wide models on both dataset sizes (≈0.77–0.78 / 0.78–0.80 / 0.72), when the training loss reaches ≈0.38, then decays as the loss keeps falling. More data only slows the decay. The label noise is intrinsic: chaotic decision boundaries are fractal.
5. **Paying 10 % of the integration cost buys the next 0.05 of AUC.**

## Round 2: proper protocol, stronger models (AUC on a 14 500-system test set, ±0.005)

133 500 training systems, 5 000 validation systems used only for early stopping, test set never used for selection.

| model | diverge | escape | collision |
|---|---|---|---|
| logistic regression | 0.761 | 0.738 | 0.706 |
| gradient-boosted trees (val-tuned) | 0.787 | 0.788 | 0.704 |
| MLP, early-stopped on validation (page model) | 0.796 | 0.797 | 0.711 |
| MLP, 5-seed ensemble | 0.799 | 0.801 | 0.713 |
| MLP + ordinal t_div heads, 5-seed ensemble | 0.799 | 0.801 | 0.715 |

Early stopping on a validation set (all seeds stop at epoch 12–19) is the single largest gain. Trees, ensembles, and richer supervision each add ≤ 0.003. The zero-cost ceiling is 0.80 / 0.80 / 0.71.

## Honest caveats

- The 1 500-system test set was used to compare model variants, so the page model carries a small selection bias; the fresh 3 000-system numbers are the ones to quote.
- Round 1 had no validation split for early stopping; round 2 fixes that and is the one to quote.
- Labels are finite-time, thresholded outcomes, not dynamical stability in the strict sense.

## Reproduce

```bash
python3 -m venv .venv && .venv/bin/pip install rebound numpy torch
.venv/bin/python scripts/gen3d.py --n 10000 --T 30 --nout 300 --seed 1000 --out data/three_body_3d_10k_T30.npz
.venv/bin/python scripts/train_chaos.py --data data/three_body_3d_10k_T30.npz
```

## Team

Xingyuan Yuan (launch lead), Shicong Yang (builder), Claude Fable 5.1 via Claude Code (co-scientist: literature check, code, experiments, and the page).
