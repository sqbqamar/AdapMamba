# -*- coding: utf-8 -*-
"""
Created on Feb 28 11:04:38 2026

@author: drsaq


test.py — Evaluation for AdapMamba-UNet.

Synapse : Volume-level DSC + HD95 per organ (matches published protocol).
ACDC    : 2-D slice DSC per cardiac structure.

Usage
─────
# Synapse
python test.py --dataset synapse \
    --data_root data/Synapse --num_classes 9 --img_size 224 \
    --checkpoint runs/synapse/ckpt_best.pth

# ACDC
python test.py --dataset acdc \
    --data_root data/ACDC --num_classes 4 --img_size 256 \
    --checkpoint runs/acdc/ckpt_best.pth
"""

import os
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
from torch.utils.data import DataLoader

from model   import adapmamba_unet_small
from dataset import (SynapseTestDataset, ACDCDataset,
                     SYNAPSE_CLASSES, ACDC_CLASSES, normalise)
from utils   import eval_volume, predict_volume, load_ckpt


# ── CLI ─────

def get_args():
    p = argparse.ArgumentParser("Evaluate AdapMamba-UNet")
    p.add_argument("--dataset",     default="synapse",
                   choices=["synapse", "acdc"])
    p.add_argument("--data_root",   default="data/Synapse")
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--num_classes", type=int, default=9)
    p.add_argument("--img_size",    type=int, default=224)
    p.add_argument("--save_preds",  action="store_true",
                   help="Save *.npy prediction masks")
    p.add_argument("--save_dir",    default="predictions")
    return p.parse_args()


# ── Synapse ────────

def eval_synapse(model, args, device):
    test_ds = SynapseTestDataset(
        root      = os.path.join(args.data_root, "test_vol_h5"),
        list_file = os.path.join(args.data_root, "test_vol.txt"),
        img_size  = args.img_size,
    )
    cls_names = SYNAPSE_CLASSES[1:]          # skip background
    all_dsc   = {c: [] for c in cls_names}
    all_hd95  = {c: [] for c in cls_names}

    W = 16   # column width for alignment
    print(f"\n  Synapse Test  —  {len(test_ds)} volumes\n")
    print(f"  {'Case':<14}  {'DSC':>6}  {'HD95':>8}")
    print(f"  {'-'*34}")

    for i in range(len(test_ds)):
        vol, gt, name = test_ds[i]
        vol  = normalise(vol)
        pred = predict_volume(vol, model, patch_size=args.img_size,
                               num_classes=args.num_classes, device=device)

        m = eval_volume(pred, gt, args.num_classes)

        for j, cn in enumerate(cls_names):
            all_dsc[cn].append(m["dsc"][j]  * 100)
            all_hd95[cn].append(m["hd95"][j])

        print(f"  {name:<14}  {m['mean_dsc']*100:>6.2f}  {m['mean_hd95']:>8.2f}")

        if args.save_preds:
            os.makedirs(args.save_dir, exist_ok=True)
            np.save(os.path.join(args.save_dir, f"{name}.npy"), pred)

    # ── Summary ────────
    print(f"\n  {'═'*46}")
    print(f"  {'Class':<18}  {'DSC (%)':>8}  {'HD95 (mm)':>10}")
    print(f"  {'─'*46}")
    grand_dsc, grand_hd = [], []
    for cn in cls_names:
        d = float(np.mean(all_dsc[cn]))
        h = float(np.mean([v for v in all_hd95[cn] if v > 0] or [0]))
        grand_dsc.append(d);  grand_hd.append(h)
        print(f"  {cn:<18}  {d:>8.2f}  {h:>10.2f}")
    print(f"  {'─'*46}")
    print(f"  {'Mean':<18}  {np.mean(grand_dsc):>8.2f}  {np.mean(grand_hd):>10.2f}")
    print(f"  {'═'*46}\n")


# ── ACDC ───────

@torch.no_grad()
def eval_acdc(model, args, device):
    test_ds = ACDCDataset(os.path.join(args.data_root, "test"),
                           img_size=args.img_size, augment=False)
    loader  = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=4)

    cls_names = ACDC_CLASSES[1:]
    dice_sum  = torch.zeros(len(cls_names), device=device)
    n = 0

    model.eval()
    for imgs, lbls in loader:
        imgs  = imgs.to(device, non_blocking=True)
        lbls  = lbls.to(device, non_blocking=True)
        preds = model(imgs).argmax(1)
        for i, c in enumerate(range(1, args.num_classes)):
            p = (preds == c).float()
            t = (lbls  == c).float()
            dice_sum[i] += 2.0 * (p * t).sum() / (p.sum() + t.sum() + 1e-5)
        n += imgs.size(0)

    per_cls = (dice_sum / n * 100).cpu().numpy()

    print(f"\n  {'═'*36}")
    print(f"  ACDC Test Results")
    print(f"  {'─'*36}")
    for name, d in zip(cls_names, per_cls):
        print(f"  {name:<16}  {d:>8.2f} %")
    print(f"  {'─'*36}")
    print(f"  {'Mean DSC':<16}  {per_cls.mean():>8.2f} %")
    print(f"  {'═'*36}\n")


# ── Main ──────────

def main():
    args   = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device     : {device}")
    print(f"  Checkpoint : {args.checkpoint}")

    model = adapmamba_unet_small(num_classes=args.num_classes).to(device)
    load_ckpt(args.checkpoint, model)
    model.eval()

    if args.dataset == "synapse":
        eval_synapse(model, args, device)
    else:
        eval_acdc(model, args, device)


if __name__ == "__main__":
    main()
