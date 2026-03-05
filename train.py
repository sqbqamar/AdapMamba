# -*- coding: utf-8 -*-
"""
Created on Feb 24 11:04:38 2026

@author: drsaq


train.py - Training script for AdapMamba-UNet.

Quick start

# Synapse multi-organ CT
python train.py \
    --dataset synapse --data_root data/Synapse \
    --output_dir runs/synapse --num_classes 9 \
    --epochs 300 --batch_size 24 --lr 3e-4 \
    --img_size 224

# ACDC cardiac MRI
python train.py \
    --dataset acdc --data_root data/ACDC \
    --output_dir runs/acdc --num_classes 4 \
    --epochs 200 --batch_size 32 --lr 1e-4 \
    --img_size 256 --lambda_dice 0.6 --lambda_ce 0.4
"""

import os
import time
import argparse
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.optim            import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.cuda.amp         import GradScaler, autocast

from model   import adapmamba_unet_small
from dataset import get_synapse_loaders, get_acdc_loaders
from utils   import CombinedLoss, AverageMeter, param_count, save_ckpt, load_ckpt



def get_args():
    p = argparse.ArgumentParser("Train AdapMamba-UNet")
    # Data
    p.add_argument("--dataset",      default="synapse",
                   choices=["synapse", "acdc"])
    p.add_argument("--data_root",    default="data/Synapse")
    p.add_argument("--num_classes",  type=int,   default=9)
    p.add_argument("--img_size",     type=int,   default=224)
    p.add_argument("--num_workers",  type=int,   default=4)
    # Model
    p.add_argument("--embed_dim",    type=int,   default=96)
    p.add_argument("--drop_path",    type=float, default=0.1)
    # Training
    p.add_argument("--epochs",       type=int,   default=300)
    p.add_argument("--batch_size",   type=int,   default=24)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--grad_clip",    type=float, default=1.0)
    p.add_argument("--warmup_epochs",type=int,   default=10)
    # Loss
    p.add_argument("--lambda_dice",  type=float, default=1.0)
    p.add_argument("--lambda_ce",    type=float, default=1.0)
    # I/O
    p.add_argument("--output_dir",   default="runs/default")
    p.add_argument("--resume",       default="",
                   help="Path to checkpoint to resume from")
    p.add_argument("--log_freq",     type=int,   default=20,
                   help="Print every N steps")
    p.add_argument("--seed",         type=int,   default=42)
    return p.parse_args()



def set_seed(seed):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# One epoch 

def train_one_epoch(model, loader, criterion, optimizer,
                    scaler, scheduler, args, device, epoch,
                    step_per_batch: bool):
    model.train()
    meter = AverageMeter()
    t0    = time.time()

    for step, (imgs, lbls) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        lbls = lbls.to(device, non_blocking=True)

        with autocast():
            loss = criterion(model(imgs), lbls)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if step_per_batch:
            scheduler.step()

        meter.update(loss.item(), imgs.size(0))

        if (step + 1) % args.log_freq == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  E{epoch:03d}  step {step+1:4d}/{len(loader)}"
                  f"  loss {meter.avg:.4f}  lr {lr_now:.2e}"
                  f"  {time.time()-t0:.1f}s")
            t0 = time.time()

    return meter.avg


# Quick 2-D val DSC 

@torch.no_grad()
def validate(model, loader, num_classes, device) -> float:
    """Fast 2-D Dice proxy (avoids full-volume evaluation during training)."""
    model.eval()
    dice_sum = torch.zeros(num_classes - 1, device=device)
    n = 0
    for imgs, lbls in loader:
        imgs = imgs.to(device, non_blocking=True)
        lbls = lbls.to(device, non_blocking=True)
        preds = model(imgs).argmax(1)
        for c in range(1, num_classes):
            p = (preds == c).float()
            t = (lbls  == c).float()
            dice_sum[c-1] += 2.0 * (p * t).sum() / (p.sum() + t.sum() + 1e-5)
        n += imgs.size(0)
    return (dice_sum / n).mean().item() * 100.0


# Main 

def main():
    args = get_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  AdapMamba-UNet  |  {args.dataset.upper()}  |  {device}")
    print(f"  Output: {args.output_dir}")
    print(f"{'='*60}\n")

    # ── Model
    model = adapmamba_unet_small(
        num_classes    = args.num_classes,
        drop_path_rate = args.drop_path,
    ).to(device)
    print(f"  Params: {param_count(model)}\n")

    # ── Data
    val_loader = None
    if args.dataset == "synapse":
        train_loader, _ = get_synapse_loaders(
            args.data_root, args.img_size, args.batch_size, args.num_workers)
    else:
        train_loader, val_loader, _ = get_acdc_loaders(
            args.data_root, args.img_size, args.batch_size, args.num_workers)

    # Loss
    criterion = CombinedLoss(
        args.num_classes, args.lambda_dice, args.lambda_ce,
        ignore_bg=True,
    )

    # Optimiser
    optimizer = AdamW(model.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay, betas=(0.9, 0.999))

    # Scheduler
    if args.dataset == "acdc":
        scheduler = OneCycleLR(
            optimizer, max_lr=args.lr,
            steps_per_epoch=len(train_loader),
            epochs=args.epochs, pct_start=0.1,
        )
        step_per_batch = True
    else:
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max   = args.epochs - args.warmup_epochs,
            eta_min = args.lr * 1e-2,
        )
        step_per_batch = False

    scaler = GradScaler()

    # Resume
    start = 0
    if args.resume:
        start = load_ckpt(args.resume, model, optimizer, scheduler)

    best_dsc = 0.0

    for epoch in range(start + 1, args.epochs + 1):

        # Linear warm-up for cosine schedule
        if not step_per_batch and epoch <= args.warmup_epochs:
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr * (epoch / args.warmup_epochs)

        loss = train_one_epoch(
            model, train_loader, criterion,
            optimizer, scaler, scheduler, args,
            device, epoch, step_per_batch,
        )

        if not step_per_batch and epoch > args.warmup_epochs:
            scheduler.step()

        # Validation
        val_dsc = None
        if val_loader is not None:
            val_dsc = validate(model, val_loader, args.num_classes, device)
            print(f"\n  Epoch {epoch:3d}  loss {loss:.4f}"
                  f"  val-DSC {val_dsc:.2f}%\n")

        is_best = val_dsc is not None and val_dsc > best_dsc
        if is_best:
            best_dsc = val_dsc
            save_ckpt(os.path.join(args.output_dir, "ckpt_best.pth"),
                      epoch, model, optimizer, scheduler, best_dsc)

        if epoch % 50 == 0 or epoch == args.epochs:
            save_ckpt(
                os.path.join(args.output_dir, f"ckpt_e{epoch:03d}.pth"),
                epoch, model, optimizer, scheduler, best_dsc,
            )

    print(f"\n  Done.  Best val-DSC: {best_dsc:.2f}%")


if __name__ == "__main__":
    main()
