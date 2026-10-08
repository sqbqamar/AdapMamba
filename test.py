"""
Evaluate on the test split.



Columns per class c: dice_c, hd95mm_c, assdmm_c (spacing-aware, NaN if undefined)
and hd95tu_c (TransUNet rule, voxel units), plus the case means over classes.

Baselines: predictions produced by another codebase can be scored with exactly the
same code by passing --pred_dir (one <volume_id>.npz per test volume, key "pred",
same grid as the ground truth).


    python test.py --dataset synapse --data_root data/synapse \
        --checkpoints runs/synapse/full/seed42/best.pth runs/synapse/full/seed123/best.pth \
                      runs/synapse/full/seed256/best.pth \
        --names 42 123 256 --output_dir results/synapse/full
"""

import argparse
import csv
import json
import os
import warnings

import numpy as np
import torch

from adapmamba.data import (SYNAPSE_CLASSES, ACDC_CLASSES, read_split, volume_ids, load_volume)
from adapmamba.metrics import evaluate_volume, predict_volume_probs, probs_to_label
from adapmamba.utils import build_from_checkpoint, environment_info

IMG_SIZE = {"synapse": 224, "acdc": 256}


def case_of(dataset, vid):
    return vid if dataset == "synapse" else vid.split("_frame")[0]


def score(dataset, vid, pred, gt, spacing, n_cls):
    mm = evaluate_volume(pred, gt, n_cls, spacing, "mm")
    tu = evaluate_volume(pred, gt, n_cls, None, "transunet")
    return {"dice": mm["dice"], "hd95mm": mm["hd95"], "assdmm": mm["assd"], "hd95tu": tu["hd95"]}


def aggregate_cases(dataset, per_volume, classes):
    """Average volumes of the same case (ACDC: ED and ES) -> one row per case."""
    rows = {}
    for vid, m in per_volume.items():
        rows.setdefault(case_of(dataset, vid), []).append(m)
    out = []
    for case, ms in sorted(rows.items()):
        row = {"case": case}
        for key in ["dice", "hd95mm", "assdmm", "hd95tu"]:
            arr = np.array([m[key] for m in ms], dtype=float)          # (n_vol, n_cls)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                per_cls = np.nanmean(arr, axis=0)
            for c, name in enumerate(classes):
                row[f"{key}_{name}"] = per_cls[c]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                row[f"{key}_mean"] = float(np.nanmean(per_cls))
        out.append(row)
    return out


def write_cases(path, rows):
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})


def means(rows):
    """Mean over cases; NaN entries (undefined HD95/ASSD) are excluded and counted."""
    keys = [k for k in rows[0] if k != "case"]
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for k in keys:
            vals = np.array([r[k] for r in rows], dtype=float)
            out[k] = float(np.nanmean(vals)) if not np.all(np.isnan(vals)) else float("nan")
            if k.startswith(("hd95mm_", "assdmm_")):
                out[f"{k}__n_undefined"] = int(np.isnan(vals).sum())
    return out


def main():
    p = argparse.ArgumentParser("Evaluate AdapMamba-UNet")
    p.add_argument("--dataset", required=True, choices=["synapse", "acdc"])
    p.add_argument("--data_root", required=True)
    p.add_argument("--splits_dir", default="splits")
    p.add_argument("--split", default="test")
    p.add_argument("--checkpoints", nargs="*", default=[])
    p.add_argument("--names", nargs="*", default=None)
    p.add_argument("--pred_dir", default=None, help="score external predictions instead of checkpoints")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--save_preds", action="store_true")
    p.add_argument("--scan_backend", default="auto")
    p.add_argument("--img_size", type=int, default=None, help="network input size (default: 224 Synapse, 256 ACDC)")
    a = p.parse_args()
    img_size = a.img_size or IMG_SIZE[a.dataset]
    os.makedirs(a.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classes = SYNAPSE_CLASSES if a.dataset == "synapse" else ACDC_CLASSES
    n_cls = len(classes) + 1
    vids = volume_ids(a.dataset, a.data_root, read_split(os.path.join(a.splits_dir, f"{a.dataset}_{a.split}.txt")))

    summary = {"dataset": a.dataset, "split": a.split, "n_volumes": len(vids),
               "environment": environment_info()}

    if a.pred_dir:
        per_vol = {}
        for vid in vids:
            _, gt, sp = load_volume(a.data_root, vid)
            pred = np.load(os.path.join(a.pred_dir, f"{vid}.npz"))["pred"].astype(np.uint8)
            per_vol[vid] = score(a.dataset, vid, pred, gt, sp, n_cls)
        rows = aggregate_cases(a.dataset, per_vol, classes)
        write_cases(os.path.join(a.output_dir, "external_cases.csv"), rows)
        summary["external"] = means(rows)
    else:
        if not a.checkpoints:
            p.error("give --checkpoints or --pred_dir")
        names = a.names or [f"run{i}" for i in range(len(a.checkpoints))]
        models = [build_from_checkpoint(c, device, a.scan_backend)[0] for c in a.checkpoints]
        per_seed = {n: {} for n in names}
        ens = {}
        for vid in vids:
            img, gt, sp = load_volume(a.data_root, vid)
            probs_each = [predict_volume_probs([m], img, img_size, device) for m in models]
            for n, pr in zip(names, probs_each):
                per_seed[n][vid] = score(a.dataset, vid, probs_to_label(pr, gt.shape[1:]), gt, sp, n_cls)
            pred_e = probs_to_label(sum(probs_each) / len(probs_each), gt.shape[1:])
            ens[vid] = score(a.dataset, vid, pred_e, gt, sp, n_cls)
            if a.save_preds:
                os.makedirs(os.path.join(a.output_dir, "preds"), exist_ok=True)
                np.savez_compressed(os.path.join(a.output_dir, "preds", f"{vid}.npz"), pred=pred_e)
            print(f"{vid}: ensemble mean DSC {100 * np.mean(ens[vid]['dice']):.2f}")

        summary["per_seed"] = {}
        for n in names:
            rows = aggregate_cases(a.dataset, per_seed[n], classes)
            write_cases(os.path.join(a.output_dir, f"seed_{n}_cases.csv"), rows)
            summary["per_seed"][n] = means(rows)
        keys = [k for k in summary["per_seed"][names[0]] if "__n_undefined" not in k]
        summary["across_seeds"] = {k: {"mean": float(np.mean([summary["per_seed"][n][k] for n in names])),
                                       "std": float(np.std([summary["per_seed"][n][k] for n in names], ddof=1))
                                       if len(names) > 1 else 0.0}
                                   for k in keys}
        rows = aggregate_cases(a.dataset, ens, classes)
        write_cases(os.path.join(a.output_dir, "ensemble_cases.csv"), rows)
        summary["ensemble"] = means(rows)
        summary["checkpoints"] = dict(zip(names, a.checkpoints))

    with open(os.path.join(a.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    if "across_seeds" in summary:
        s = summary["across_seeds"]
        print(f"\nmean DSC across seeds: {100 * s['dice_mean']['mean']:.2f} +/- {100 * s['dice_mean']['std']:.2f}")
        print(f"mean HD95 (mm):        {s['hd95mm_mean']['mean']:.2f} +/- {s['hd95mm_mean']['std']:.2f}")
    print(f"wrote {a.output_dir}/summary.json")


if __name__ == "__main__":
    main()
