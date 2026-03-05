# -*- coding: utf-8 -*-
"""
Created on Feb 21 11:04:38 2026

@author: drsaq

utils.py - Losses, metrics, and training utilities.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Loss Functions

class DiceLoss(nn.Module):
    """
    Soft multi-class Dice loss.

    Args
    num_classes   : Total number of classes.
    smooth        : Laplace smoothing term.
    ignore_bg     : Whether to skip class 0 (background) from averaging.
    """
    def __init__(self, num_classes: int, smooth: float = 1e-5,
                 ignore_bg: bool = True):
        super().__init__()
        self.C        = num_classes
        self.smooth   = smooth
        self.start    = 1 if ignore_bg else 0

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : B Ã C Ã H Ã W  (raw, NOT softmaxed)
        targets : B Ã H Ã W      (integer labels 0..C-1)
        """
        probs = F.softmax(logits, dim=1)
        B     = probs.shape[0]
        oh    = F.one_hot(targets, self.C).permute(0, 3, 1, 2).float()

        dice, n = 0.0, 0
        for c in range(self.start, self.C):
            p     = probs[:, c].reshape(B, -1)
            t     = oh[:, c].reshape(B, -1)
            inter = (p * t).sum(1)
            union = p.sum(1) + t.sum(1)
            dice += ((2.0 * inter + self.smooth) /
                     (union + self.smooth)).mean()
            n += 1
        return 1.0 - dice / max(n, 1)


class CombinedLoss(nn.Module):
    """
    L_total = Î»_dice Â· L_Dice  +  Î»_ce Â· L_CE
    Paper: Î»_dice=Î»_ce=1.0 for Synapse;  Î»_dice=0.6, Î»_ce=0.4 for ACDC.
    """
    def __init__(self, num_classes: int,
                 lambda_dice: float = 1.0, lambda_ce: float = 1.0,
                 ignore_bg: bool = True):
        super().__init__()
        self.dice = DiceLoss(num_classes, ignore_bg=ignore_bg)
        self.ce   = nn.CrossEntropyLoss()
        self.ld   = lambda_dice
        self.lc   = lambda_ce

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        return self.ld * self.dice(logits, targets) + \
               self.lc * self.ce(logits, targets)


# Metrics

def dice_coef(pred: np.ndarray, gt: np.ndarray) -> float:
    """Volumetric DSC for one binary class pair."""
    ps, gs = pred.sum(), gt.sum()
    if ps == 0 and gs == 0:
        return 1.0
    if ps == 0 or gs == 0:
        return 0.0
    return 2.0 * float((pred & gt).sum()) / float(ps + gs)


def hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    95th-percentile Hausdorff distance (mm / voxels).
    Falls back to a simple max-dist approximation when medpy is absent.
    """
    try:
        from medpy.metric.binary import hd95 as _hd95
        if pred.sum() == 0 or gt.sum() == 0:
            return float("nan")
        return float(_hd95(pred, gt))
    except ImportError:
        # Lightweight fallback: bounding-box diagonal distance
        if pred.sum() == 0 or gt.sum() == 0:
            return float("nan")
        def _bbox(a):
            idx = np.where(a)
            return [(i.min(), i.max()) for i in idx]
        pb, gb = _bbox(pred), _bbox(gt)
        d = sum((pb[i][0]-gb[i][0])**2 + (pb[i][1]-gb[i][1])**2
                for i in range(len(pb)))
        return float(np.sqrt(d))


def eval_volume(pred_vol: np.ndarray, gt_vol: np.ndarray,
                num_classes: int) -> dict:
    """
    Compute per-class DSC and HD95 for a single 3-D volume.
    Classes 1..num_classes-1 are evaluated (background skipped).
    Returns {"dsc": [...], "hd95": [...], "mean_dsc": float, "mean_hd95": float}
    """
    dsc_list, hd_list = [], []
    for c in range(1, num_classes):
        p  = (pred_vol == c)
        g  = (gt_vol   == c)
        dsc_list.append(dice_coef(p, g))
        h = hd95(p, g)
        hd_list.append(h if not np.isnan(h) else 0.0)

    valid_hd = [h for h in hd_list if h > 0]
    return {
        "dsc":      dsc_list,
        "hd95":     hd_list,
        "mean_dsc":  float(np.mean(dsc_list)),
        "mean_hd95": float(np.mean(valid_hd)) if valid_hd else 0.0,
    }


# Sliding-window inference on 3-D volumes (axial slice-by-slice)

@torch.no_grad()
def predict_volume(volume: np.ndarray,
                   model: nn.Module,
                   patch_size: int = 224,
                   num_classes: int = 9,
                   device: torch.device = torch.device("cpu")) -> np.ndarray:
    """
    Predict a full 3-D volume slice by slice.

    volume : D Ã H Ã W  float32 array, values in [0,1].
    Returns D Ã H Ã W  uint8 prediction.
    """
    from scipy.ndimage import zoom as _zoom

    model.eval()
    D, H, W = volume.shape
    pred    = np.zeros((D, H, W), dtype=np.uint8)

    for d in range(D):
        slc = volume[d]                                    # H W

        # Resize to patch_size if needed
        if H != patch_size or W != patch_size:
            s = _zoom(slc, (patch_size/H, patch_size/W), order=3)
        else:
            s = slc

        x = torch.from_numpy(s).float().unsqueeze(0).unsqueeze(0)  # 1 1 P P
        x = x.repeat(1, 3, 1, 1).to(device)

        logits  = model(x)
        pred_r  = logits.argmax(1).squeeze().cpu().numpy().astype(np.uint8)

        if H != patch_size or W != patch_size:
            pred_r = _zoom(pred_r, (H/patch_size, W/patch_size), order=0)

        pred[d] = pred_r

    return pred


# Misc helpers

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val: float, n: int = 1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count


def param_count(model: nn.Module) -> str:
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return f"{tr/1e6:.2f} M trainable  /  {total/1e6:.2f} M total"


def save_ckpt(path: str, epoch: int, model: nn.Module,
              optimizer=None, scheduler=None, best_dsc: float = 0.0):
    state = {"epoch": epoch, "model": model.state_dict(), "best_dsc": best_dsc}
    if optimizer  is not None: state["optimizer"]  = optimizer.state_dict()
    if scheduler  is not None: state["scheduler"]  = scheduler.state_dict()
    torch.save(state, path)
    print(f"  â  Saved  â  {path}")


def load_ckpt(path: str, model: nn.Module,
              optimizer=None, scheduler=None) -> int:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    epoch = ckpt.get("epoch", 0)
    print(f"  a:   Loaded epoch {epoch}  (best DSC {ckpt.get('best_dsc',0):.2f}%)")
    return epoch
