

import glob
import os
import random

import h5py
import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import Dataset

SYNAPSE_CLASSES = ["aorta", "gallbladder", "spleen", "kidney_l", "kidney_r", "liver", "stomach", "pancreas"]
ACDC_CLASSES = ["RV", "Myo", "LV"]
SYNAPSE_KIDNEY_L, SYNAPSE_KIDNEY_R = 4, 5


def read_split(path):
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]


def volume_ids(dataset, root, ids):
    """Split files hold case ids (Synapse) or patient ids (ACDC); return volume ids."""
    if dataset == "synapse":
        return list(ids)
    out = []
    for pid in ids:
        vols = sorted(os.path.basename(p)[:-3] for p in glob.glob(os.path.join(root, "volumes", f"{pid}_frame*.h5")))
        if not vols:
            raise FileNotFoundError(f"no volumes for {pid} in {root}/volumes")
        out.extend(vols)
    return out


def resize2d(a, size, order):
    h, w = a.shape
    if (h, w) == (size, size):
        return a
    return ndimage.zoom(a, (size / h, size / w), order=order)


def _center_fit(a, h, w):
    ah, aw = a.shape
    ph, pw = max(0, h - ah), max(0, w - aw)
    a = np.pad(a, [(ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2)], mode="constant")
    sh, sw = (a.shape[0] - h) // 2, (a.shape[1] - w) // 2
    return a[sh:sh + h, sw:sw + w]


def augment(img, lbl, swap_kidneys, rng):
    if rng.random() < 0.5:
        img, lbl = np.fliplr(img).copy(), np.fliplr(lbl).copy()
        if swap_kidneys:
            left, right = lbl == SYNAPSE_KIDNEY_L, lbl == SYNAPSE_KIDNEY_R
            lbl[left], lbl[right] = SYNAPSE_KIDNEY_R, SYNAPSE_KIDNEY_L
    if rng.random() < 0.5:
        ang = rng.uniform(-30.0, 30.0)
        img = ndimage.rotate(img, ang, order=1, reshape=False, mode="constant", cval=float(img.min()))
        lbl = ndimage.rotate(lbl, ang, order=0, reshape=False, mode="constant", cval=0)
    if rng.random() < 0.5:
        s = rng.uniform(0.75, 1.25)
        h, w = img.shape
        img = _center_fit(ndimage.zoom(img, s, order=1), h, w)
        lbl = _center_fit(ndimage.zoom(lbl, s, order=0), h, w)
    return img, lbl


class SliceDataset(Dataset):
    """2D slices of the given volumes. Returns (image 3xSxS float32, label SxS int64)."""

    def __init__(self, root, vol_ids, img_size, augment_data=True, swap_kidneys=False, seed=0):
        self.files = []
        for vid in vol_ids:
            fs = sorted(glob.glob(os.path.join(root, "slices", f"{vid}_slice*.npz")))
            if not fs:
                raise FileNotFoundError(f"no slices for {vid} in {root}/slices")
            self.files.extend(fs)
        self.size, self.augment, self.swap = img_size, augment_data, swap_kidneys
        self.seed = seed

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = np.load(self.files[i])
        img, lbl = d["image"].astype(np.float32), d["label"].astype(np.uint8)
        if self.augment:
            img, lbl = augment(img, lbl, self.swap, random)
        img = resize2d(img, self.size, order=1)
        lbl = resize2d(lbl, self.size, order=0)
        x = torch.from_numpy(np.ascontiguousarray(img)).unsqueeze(0).repeat(3, 1, 1)
        return x, torch.from_numpy(np.ascontiguousarray(lbl)).long()


def load_volume(root, vid):
    with h5py.File(os.path.join(root, "volumes", f"{vid}.h5"), "r") as f:
        spacing = tuple(float(s) for s in f.attrs["spacing"]) if "spacing" in f.attrs else None
        return f["image"][:].astype(np.float32), f["label"][:].astype(np.uint8), spacing


def seed_worker(worker_id):
    """DataLoader worker seeding so augmentation is reproducible for a given seed."""
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)
