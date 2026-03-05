# -*- coding: utf-8 -*-
"""
Created on Feb 12 11:04:38 2026

@author: drsaq


dataset.py - Synapse & ACDC dataset loaders for AdapMamba-UNet.

Preprocessing convention matches TransUNet:
  https://github.com/Beckschen/TransUNet

Synapse folder layout

  data/Synapse/
    train_npz/   case0001_slice000.npz  ...   keys: "image" (H×W), "label" (H×W)
    test_vol_h5/ case0008.npy.h5        ...   keys: "image" (D×H×W), "label" (D×H×W)
    train.txt    (slice names, one per line)
    test_vol.txt (volume names, one per line)

ACDC folder layout
  data/ACDC/
    train/  *.npz     keys: "image", "label"
    val/    *.npz
    test/   *.npz
"""

import os
import random
from pathlib import Path

import h5py
import numpy as np
from scipy import ndimage

import torch
from torch.utils.data import Dataset, DataLoader


# Augmentation helpers

def rand_rot_flip(img: np.ndarray, lbl: np.ndarray):
    k = random.randint(0, 3)
    img = np.rot90(img, k)
    lbl = np.rot90(lbl, k)
    if random.random() > 0.5:
        img = np.fliplr(img).copy()
        lbl = np.fliplr(lbl).copy()
    return img, lbl


def rand_rotate(img: np.ndarray, lbl: np.ndarray, max_deg: float = 30.0):
    angle = random.uniform(-max_deg, max_deg)
    img   = ndimage.rotate(img, angle, order=3, reshape=False, cval=img.min())
    lbl   = ndimage.rotate(lbl, angle, order=0, reshape=False, cval=0)
    return img, lbl


def rand_zoom(img: np.ndarray, lbl: np.ndarray,
              lo: float = 0.75, hi: float = 1.25):
    fac   = random.uniform(lo, hi)
    H, W  = img.shape
    img_z = ndimage.zoom(img, (fac, fac), order=3)
    lbl_z = ndimage.zoom(lbl, (fac, fac), order=0)

    def _crop_pad(a, th, tw):
        ah, aw = a.shape[:2]
        ph = max(0, th - ah); pw = max(0, tw - aw)
        a  = np.pad(a, [(ph//2, ph-ph//2), (pw//2, pw-pw//2)], mode='constant')
        sh = (a.shape[0]-th)//2; sw = (a.shape[1]-tw)//2
        return a[sh:sh+th, sw:sw+tw]

    return _crop_pad(img_z, H, W), _crop_pad(lbl_z, H, W)


def normalise(img: np.ndarray) -> np.ndarray:
    lo, hi = img.min(), img.max()
    return (img - lo) / (hi - lo + 1e-8)


def resize_slice(arr: np.ndarray, size: int, order: int) -> np.ndarray:
    H, W = arr.shape
    if H == size and W == size:
        return arr
    return ndimage.zoom(arr, (size / H, size / W), order=order)


# Synapse

SYNAPSE_CLASSES = [
    "background", "aorta", "gallbladder",
    "spleen", "left_kidney", "right_kidney",
    "liver", "stomach", "pancreas",
]


class SynapseDataset(Dataset):
    """2-D slice training set.  Returns (image: 3×H×W, label: H×W)."""

    def __init__(self, root: str, list_file: str,
                 img_size: int = 224, augment: bool = True):
        self.root     = Path(root)
        self.size     = img_size
        self.augment  = augment
        with open(list_file) as f:
            self.names = [l.strip() for l in f if l.strip()]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx: int):
        d   = np.load(self.root / f"{self.names[idx]}.npz")
        img = d["image"].astype(np.float32)
        lbl = d["label"].astype(np.uint8)

        if self.augment:
            img, lbl = rand_rot_flip(img, lbl)
            img, lbl = rand_rotate(img, lbl)
            img, lbl = rand_zoom(img, lbl)

        img = resize_slice(normalise(img), self.size, 3)
        lbl = resize_slice(lbl, self.size, 0)

        img_t = torch.from_numpy(img).unsqueeze(0).repeat(3, 1, 1)  # 3 H W
        lbl_t = torch.from_numpy(lbl.copy()).long()                  # H W
        return img_t, lbl_t


class SynapseTestDataset(Dataset):
    """3-D HDF5 volumes for full-volume evaluation."""

    def __init__(self, root: str, list_file: str, img_size: int = 224):
        self.root  = Path(root)
        self.size  = img_size
        with open(list_file) as f:
            self.names = [l.strip() for l in f if l.strip()]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx: int):
        name = self.names[idx]
        with h5py.File(self.root / f"{name}.npy.h5", "r") as hf:
            img = hf["image"][:]    # D H W  float32
            lbl = hf["label"][:]    # D H W  uint8
        return img.astype(np.float32), lbl.astype(np.uint8), name


# ACDC

ACDC_CLASSES = ["background", "RV", "Myocardium", "LV"]


class ACDCDataset(Dataset):
    """2-D slice dataset for ACDC.  Returns (image: 3×H×W, label: H×W)."""

    def __init__(self, root: str, img_size: int = 256, augment: bool = True):
        self.root    = Path(root)
        self.size    = img_size
        self.augment = augment
        self.files   = sorted(self.root.glob("*.npz"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        d   = np.load(self.files[idx])
        img = d["image"].astype(np.float32)
        lbl = d["label"].astype(np.uint8)

        if self.augment:
            img, lbl = rand_rot_flip(img, lbl)
            img, lbl = rand_rotate(img, lbl, max_deg=20.0)

        img = resize_slice(normalise(img), self.size, 3)
        lbl = resize_slice(lbl, self.size, 0)

        img_t = torch.from_numpy(img).unsqueeze(0).repeat(3, 1, 1)
        lbl_t = torch.from_numpy(lbl.copy()).long()
        return img_t, lbl_t


# DataLoader factories

def get_synapse_loaders(data_root: str, img_size: int = 224,
                        batch_size: int = 24, num_workers: int = 4):
    """
    Returns (train_loader, test_dataset).
    Test dataset is returned raw because eval requires per-volume iteration.
    """
    train_ds = SynapseDataset(
        root      = os.path.join(data_root, "train_npz"),
        list_file = os.path.join(data_root, "train.txt"),
        img_size  = img_size, augment=True,
    )
    test_ds = SynapseTestDataset(
        root      = os.path.join(data_root, "test_vol_h5"),
        list_file = os.path.join(data_root, "test_vol.txt"),
        img_size  = img_size,
    )
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=True, drop_last=True)
    return loader, test_ds


def get_acdc_loaders(data_root: str, img_size: int = 256,
                     batch_size: int = 32, num_workers: int = 4):
    """Returns (train_loader, val_loader, test_loader)."""
    def _mk(split, aug):
        ds = ACDCDataset(os.path.join(data_root, split),
                         img_size=img_size, augment=aug)
        return DataLoader(ds, batch_size=batch_size, shuffle=aug,
                          num_workers=num_workers, pin_memory=True)
    return _mk("train", True), _mk("val", False), _mk("test", False)
