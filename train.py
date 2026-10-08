"""
Train AdapMamba-UNet .

Protocol (Section 4.2)
    Synapse : train on the 15 training cases in splits/synapse_train.txt; the 3 cases
              in splits/synapse_val.txt are the internal validation subset used for
              checkpoint selection; the 12 test cases are never loaded here.
              300 epochs, batch 24, AdamW lr 3e-4 with cosine annealing, 224x224,
              lambda_Dice = lambda_CE = 1.0.
    ACDC    : 70 / 10 / 20 patients (splits/acdc_*.txt), both annotated frames.
              200 epochs, batch 32, OneCycleLR with peak lr 1e-4, 256x256,
              lambda_Dice = 0.6, lambda_CE = 0.4.
    Both    : AdamW betas (0.9, 0.999), eps 1e-8, weight decay 1e-2 (A_logs and D
              excluded, as in VMamba), gradient clipping at norm 1.0, FP32, encoder
              initialised from ImageNet-1K VMamba-Small, remaining layers Kaiming.
              After every epoch the model is evaluated on the validation volumes and
              the checkpoint with the highest mean volumetric DSC is kept (best.pth).

Example
    python train.py --dataset synapse --data_root data/synapse \
        --pretrained pretrained/vssmsmall_dp03_ckpt_epoch_238.pth \
        --config full --seed 42 --output_dir runs/synapse/full/seed42
"""

import argparse
import csv
import os
import time

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.utils.data import DataLoader

from adapmamba.configs import get_config
from adapmamba.data import SliceDataset, read_split, volume_ids, load_volume, seed_worker
from adapmamba.metrics import dice, predict_volume_probs, probs_to_label
from adapmamba.model import AdapMambaUNet, param_groups
from adapmamba.utils import (CompositeLoss, set_determinism, save_checkpoint,
                             environment_info, file_sha256, write_json)

DEFAULTS = {
    "synapse": dict(num_classes=9, epochs=300, batch_size=24, lr=3e-4, img_size=224,
                    lambda_dice=1.0, lambda_ce=1.0),
    "acdc": dict(num_classes=4, epochs=200, batch_size=32, lr=1e-4, img_size=256,
                 lambda_dice=0.6, lambda_ce=0.4),
}


def get_args():
    p = argparse.ArgumentParser("Train AdapMamba-UNet")
    p.add_argument("--dataset", required=True, choices=["synapse", "acdc"])
    p.add_argument("--data_root", required=True)
    p.add_argument("--splits_dir", default="splits")
    p.add_argument("--config", default="full")
    p.add_argument("--pretrained", default=None, help="VMamba-Small checkpoint")
    p.add_argument("--no_pretrained", action="store_true", help="for smoke tests only; departs from the paper")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", required=True)
    for k in ["epochs", "batch_size", "img_size"]:
        p.add_argument(f"--{k}", type=int, default=None)
    for k in ["lr", "lambda_dice", "lambda_ce"]:
        p.add_argument(f"--{k}", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--amp", action="store_true", help="mixed precision (off by default: the paper trains in FP32)")
    p.add_argument("--no_deterministic", action="store_true")
    p.add_argument("--scan_backend", default="auto", choices=["auto", "cuda", "torch"])
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--val_every", type=int, default=1)
    p.add_argument("--max_train_iters", type=int, default=0, help="cap iterations per epoch (smoke tests)")
    a = p.parse_args()
    for k, v in DEFAULTS[a.dataset].items():
        if getattr(a, k, None) is None:
            setattr(a, k, v)
    if a.pretrained is None and not a.no_pretrained:
        p.error("--pretrained is required (the paper initialises the encoder from VMamba-Small)")
    return a


@torch.no_grad()
def validate(model, root, vol_ids, num_classes, img_size, device):
    scores = []
    for vid in vol_ids:
        img, lbl, _ = load_volume(root, vid)
        probs = predict_volume_probs([model], img, img_size, device)
        pred = probs_to_label(probs, lbl.shape[1:])
        scores.append(np.mean([dice(pred == c, lbl == c) for c in range(1, num_classes)]))
    model.train()
    return float(np.mean(scores)) * 100.0


def main():
    a = get_args()
    set_determinism(a.seed, deterministic=not a.no_deterministic)
    os.makedirs(a.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = get_config(a.config)
    model = AdapMambaUNet(a.num_classes, scan_backend=a.scan_backend, **cfg)
    pre_report = None
    if not a.no_pretrained:
        pre_report = model.load_pretrained_encoder(a.pretrained)
        print(f"encoder initialised from {a.pretrained}: {pre_report['loaded']}/{pre_report['encoder_tensors']} tensors")
    model.to(device)

    train_ids = volume_ids(a.dataset, a.data_root, read_split(os.path.join(a.splits_dir, f"{a.dataset}_train.txt")))
    val_ids = volume_ids(a.dataset, a.data_root, read_split(os.path.join(a.splits_dir, f"{a.dataset}_val.txt")))
    train_ds = SliceDataset(a.data_root, train_ids, a.img_size, augment_data=True,
                            swap_kidneys=(a.dataset == "synapse"))
    g = torch.Generator()
    g.manual_seed(a.seed)
    loader = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True, drop_last=True,
                        num_workers=a.num_workers, pin_memory=device.type == "cuda",
                        worker_init_fn=seed_worker, generator=g)
    steps_per_epoch = len(loader) if not a.max_train_iters else min(len(loader), a.max_train_iters)

    criterion = CompositeLoss(a.num_classes, a.lambda_dice, a.lambda_ce)
    optimizer = AdamW(param_groups(model, a.weight_decay), lr=a.lr, betas=(0.9, 0.999), eps=1e-8)
    if a.dataset == "synapse":
        scheduler, per_batch = CosineAnnealingLR(optimizer, T_max=a.epochs), False
    else:
        scheduler = OneCycleLR(optimizer, max_lr=a.lr, epochs=a.epochs, steps_per_epoch=steps_per_epoch, pct_start=0.3)
        per_batch = True
    use_scaler = a.amp and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)      # PyTorch >= 2.3
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)         # PyTorch 2.1 / 2.2

    write_json(os.path.join(a.output_dir, "run_config.json"), {
        "args": vars(a), "model_config": model.config, "pretrained": pre_report,
        "train_volumes": train_ids, "val_volumes": val_ids, "n_train_slices": len(train_ds),
        "release_version": "1.0.0",
        "pretrained_checkpoint": {"path": a.pretrained, "sha256": file_sha256(a.pretrained) if a.pretrained else None},
        "environment": environment_info()})

    log_path = os.path.join(a.output_dir, "log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "lr", "train_loss", "val_dsc", "best_val_dsc", "seconds"])

    best = -1.0
    for epoch in range(1, a.epochs + 1):
        t0, model_loss, n = time.time(), 0.0, 0
        model.train()
        for it, (x, y) in enumerate(loader):
            if a.max_train_iters and it >= a.max_train_iters:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=a.amp):
                loss = criterion(model(x), y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if per_batch:
                scheduler.step()
            model_loss += loss.item() * x.size(0)
            n += x.size(0)
        lr_now = optimizer.param_groups[0]["lr"]
        if not per_batch:
            scheduler.step()

        val = float("nan")
        if epoch % a.val_every == 0 or epoch == a.epochs:
            val = validate(model, a.data_root, val_ids, a.num_classes, a.img_size, device)
            if val > best:
                best = val
                save_checkpoint(os.path.join(a.output_dir, "best.pth"), model, epoch, val)
        save_checkpoint(os.path.join(a.output_dir, "last.pth"), model, epoch, val, optimizer, scheduler)
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{lr_now:.3e}", f"{model_loss / max(n, 1):.5f}",
                                    f"{val:.3f}", f"{best:.3f}", f"{time.time() - t0:.1f}"])
        print(f"epoch {epoch:3d}  lr {lr_now:.2e}  loss {model_loss / max(n, 1):.4f}  "
              f"val DSC {val:.2f}  best {best:.2f}  ({time.time() - t0:.0f}s)")
    write_json(os.path.join(a.output_dir, "COMPLETE.json"), {"best_val_dsc": best, "epochs": a.epochs})
    print(f"done. best validation DSC {best:.2f}; checkpoint {a.output_dir}/best.pth")


if __name__ == "__main__":
    main()
