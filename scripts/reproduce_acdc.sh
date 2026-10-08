#!/usr/bin/env bash
# Main ACDC result: three seeds of the full model, per-seed and ensemble evaluation.
# Usage: bash scripts/reproduce_acdc.sh data/acdc
set -euo pipefail
DATA=${1:-data/acdc}
PRE=pretrained/vssmsmall_dp03_ckpt_epoch_238.pth
for SEED in 42 123 256; do
  python train.py --dataset acdc --data_root "$DATA" --pretrained "$PRE" \
    --config full --seed "$SEED" --output_dir "runs/acdc/full/seed$SEED"
done
python test.py --dataset acdc --data_root "$DATA" \
  --checkpoints runs/acdc/full/seed{42,123,256}/best.pth --names 42 123 256 \
  --output_dir results/acdc/full --save_preds
