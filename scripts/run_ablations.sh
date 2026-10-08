#!/usr/bin/env bash
# Every ablation configuration on Synapse, three seeds each, same protocol as the full model.
# Usage: bash scripts/run_ablations.sh data/synapse [config ...]
set -euo pipefail
DATA=${1:-data/synapse}; shift || true
PRE=pretrained/vssmsmall_dp03_ckpt_epoch_238.pth
CONFIGS=${@:-"baseline step_pau step_pau_hcag step_pau_hcag_apfa_fixed \
  fact_p0_h0_a0 fact_p0_h0_a1 fact_p0_h1_a0 fact_p0_h1_a1 fact_p1_h0_a0 fact_p1_h0_a1 fact_p1_h1_a0 fact_p1_h1_a1 \
  up_bilinear up_transposed up_patch_expand up_carafe \
  gate_none gate_channel gate_spatial gate_both_d1 gate_both_d4 \
  apfa_d1 apfa_d12 apfa_d136 apfa_d1248"}
for CFG in $CONFIGS; do
  for SEED in 42 123 256; do
    OUT="runs/synapse/$CFG/seed$SEED"
    [ -f "$OUT/COMPLETE.json" ] && continue   # resume: skip finished runs
    python train.py --dataset synapse --data_root "$DATA" --pretrained "$PRE" \
      --config "$CFG" --seed "$SEED" --output_dir "$OUT"
  done
  python test.py --dataset synapse --data_root "$DATA" \
    --checkpoints runs/synapse/$CFG/seed{42,123,256}/best.pth --names 42 123 256 \
    --output_dir "results/synapse/$CFG"
done
