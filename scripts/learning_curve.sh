#!/bin/bash
DATA=data/three_body_3d_10k_T30.npz,data/three_body_3d_40k_T30.npz
for n in 2000 5000 8500 20000 48500; do
  .venv/bin/python -u scripts/train_chaos.py --data $DATA --ntrain $n --skip_noaug --epochs 60 --out data/lc_$n.json 2>&1 | grep -E "^\[(logistic|mlp-aug)\] (diverge|escape|collision)|^train" | sed "s/^/[n=$n] /"
done
