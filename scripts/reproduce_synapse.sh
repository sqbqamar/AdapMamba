#!/usr/bin/env bash
# Main Synapse result: three seeds of the full model, per-seed and ensemble evaluation.
# Usage: bash scripts/reproduce_synapse.sh data/synapse
set -euo pipefail
DATA=${1:-data/synapse}
PRE=pretrained/vssmsmall_dp03_ckpt_epoch_238.pth
for SEED in 42 123 256; do
  python train.py --dataset synapse --data_root "$DATA" --pretrained "$PRE" \
    --config full --seed "$SEED" --output_dir "runs/synapse/full/seed$SEED"
done
python test.py --dataset synapse --data_root "$DATA" \
  --checkpoints runs/synapse/full/seed{42,123,256}/best.pth --names 42 123 256 \
  --output_dir results/synapse/full --save_preds
