#!/usr/bin/env bash
# Loss-weight grid search on the ACDC validation split (Section 3.6). Selection uses only
# the validation DSC recorded in each run's log; the test split is never touched.
# Usage: bash scripts/grid_search_acdc.sh data/acdc
set -euo pipefail
DATA=${1:-data/acdc}
PRE=pretrained/vssmsmall_dp03_ckpt_epoch_238.pth
GRID="0.4 0.5 0.6 0.7 1.0"
for L1 in $GRID; do for L2 in $GRID; do
  python train.py --dataset acdc --data_root "$DATA" --pretrained "$PRE" --config full --seed 42 \
    --lambda_dice "$L1" --lambda_ce "$L2" --output_dir "runs/acdc_grid/d${L1}_c${L2}"
done; done
python - << 'PY'
import csv, glob, re
rows = []
for log in glob.glob("runs/acdc_grid/*/log.csv"):
    best = max(float(r["val_dsc"]) for r in csv.DictReader(open(log)) if r["val_dsc"] != "nan")
    l1, l2 = re.search(r"d([\d.]+)_c([\d.]+)", log).groups()
    rows.append((best, float(l1), float(l2)))
rows.sort(reverse=True)
with open("runs/acdc_grid/grid_results.csv", "w") as f:
    f.write("val_dsc,lambda_dice,lambda_ce\n" + "".join(f"{b:.3f},{a},{c}\n" for b, a, c in rows))
print("best:", rows[0])
PY
