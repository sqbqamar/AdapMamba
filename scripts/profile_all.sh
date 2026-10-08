#!/usr/bin/env bash
# Cost tables: parameters, FLOPs, latency and peak memory (GPU with mamba_ssm required for latency).
set -euo pipefail
mkdir -p results
python profile_model.py --configs baseline step_pau step_pau_hcag step_pau_hcag_apfa_fixed full \
  up_bilinear up_transposed up_patch_expand up_carafe gate_none gate_channel gate_spatial \
  gate_both_d1 gate_both_d4 apfa_d1 apfa_d12 apfa_d136 apfa_d1248 \
  fact_p0_h0_a1 fact_p0_h1_a0 fact_p0_h1_a1 fact_p1_h0_a1 \
  --resolutions 224 --out results/cost_224.csv
python profile_model.py --configs baseline full --resolutions 224 320 448 512 --out results/cost_resolution.csv
