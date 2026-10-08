import json
import hashlib
import os
import platform
import random
import subprocess

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_determinism(seed, deterministic=True):
    """Seed Python, NumPy and PyTorch, and enable deterministic cuDNN settings.

    Note: the fused selective-scan kernel of mamba_ssm uses atomic additions in
    its backward pass, so bitwise-identical repeated runs are not guaranteed on
    GPU even with these settings; results agree to within floating-point noise.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


class DiceLoss(nn.Module):
    """Soft multi-class Dice loss over the foreground classes, averaged over classes."""

    def __init__(self, num_classes, smooth=1e-5):
        super().__init__()
        self.num_classes, self.smooth = num_classes, smooth

    def forward(self, logits, target):
        probs = torch.softmax(logits, dim=1)
        onehot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        inter = (probs * onehot).sum(dims)
        union = probs.sum(dims) + onehot.sum(dims)
        dsc = (2 * inter + self.smooth) / (union + self.smooth)
        return 1.0 - dsc[1:].mean()


class CompositeLoss(nn.Module):
    """L_total = lambda_1 * L_Dice + lambda_2 * L_CE (Eq. 9)."""

    def __init__(self, num_classes, lambda_dice=1.0, lambda_ce=1.0):
        super().__init__()
        self.dice, self.ce = DiceLoss(num_classes), nn.CrossEntropyLoss()
        self.l1, self.l2 = lambda_dice, lambda_ce

    def forward(self, logits, target):
        return self.l1 * self.dice(logits, target) + self.l2 * self.ce(logits, target)


def save_checkpoint(path, model, epoch, metric, optimizer=None, scheduler=None, extra=None):
    state = {"model": model.state_dict(), "config": getattr(model, "config", None),
             "epoch": epoch, "val_dsc": metric}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if extra:
        state.update(extra)
    torch.save(state, path)


def build_from_checkpoint(path, device="cpu", scan_backend="auto"):
    from .model import AdapMambaUNet
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = dict(ck["config"])
    num_classes = cfg.pop("num_classes")
    cfg["depths"] = tuple(cfg["depths"])
    cfg["apfa_dilations"] = tuple(cfg["apfa_dilations"])
    model = AdapMambaUNet(num_classes, scan_backend=scan_backend, **cfg)
    model.load_state_dict(ck["model"], strict=True)
    return model.to(device).eval(), ck



def file_sha256(path, chunk_size=1024 * 1024):
    """Return SHA256 of a file, or None if the file is absent."""
    if not path or not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def environment_info():
    info = {"python": platform.python_version(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    try:
        from .ss2d import cuda_scan_available
        info["mamba_ssm_kernel"] = cuda_scan_available()
    except Exception:
        info["mamba_ssm_kernel"] = False
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        info["git_commit"] = subprocess.check_output(
            ["git", "-C", root, "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        info["git_commit"] = os.environ.get("ADAPMAMBA_RELEASE_COMMIT")
    info["release_version"] = "1.0.0"
    return info


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
