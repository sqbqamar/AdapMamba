"""


Synapse (raw BTCV NIfTI: RawData/Training/img/img0001.nii.gz, label/label0001.nii.gz)
    * CT intensities clipped to the soft-tissue window [-125, 275] HU and rescaled to [0, 1].
    * Labels mapped from the 13 BTCV classes to the 8 organs of the standard protocol:
      1 aorta, 2 gallbladder, 3 spleen, 4 left kidney, 5 right kidney, 6 liver,
      7 stomach, 8 pancreas; all other structures become background.
    * Case ids: case0001 ... case0040.

ACDC (raw: training/patientXXX/patientXXX_frameYY.nii.gz and *_gt.nii.gz, Info.cfg)
    * Both annotated frames (end-diastole and end-systole) of every patient.
    * Each volume normalised to zero mean and unit variance.
    * Labels: 1 RV, 2 myocardium, 3 LV. All slices are kept, including slices
      without labelled foreground.
    * Volume ids: patientXXX_frameYY; the split files list patient ids.

Usage
    python preprocess.py synapse --raw RawData/Training --out data/synapse
    python preprocess.py acdc    --raw ACDC/database    --out data/acdc
"""

import argparse
import glob
import os
import re

import h5py
import numpy as np

# BTCV label -> protocol label (TransUNet ordering)
BTCV_TO_PROTOCOL = {8: 1, 4: 2, 1: 3, 3: 4, 2: 5, 6: 6, 7: 7, 11: 8}


def _load_nifti(path):
    import nibabel as nib
    img = nib.load(path)
    arr = np.asarray(img.dataobj)                     # (X, Y, Z)
    zooms = img.header.get_zooms()[:3]               # (sx, sy, sz)
    return np.transpose(arr, (2, 1, 0)), (float(zooms[2]), float(zooms[1]), float(zooms[0]))


def _write(out, vid, image, label, spacing):
    os.makedirs(os.path.join(out, "volumes"), exist_ok=True)
    os.makedirs(os.path.join(out, "slices"), exist_ok=True)
    with h5py.File(os.path.join(out, "volumes", f"{vid}.h5"), "w") as f:
        f.create_dataset("image", data=image.astype(np.float32), compression="gzip")
        f.create_dataset("label", data=label.astype(np.uint8), compression="gzip")
        f.attrs["spacing"] = np.asarray(spacing, dtype=np.float64)
    for k in range(image.shape[0]):
        np.savez_compressed(os.path.join(out, "slices", f"{vid}_slice{k:03d}.npz"),
                            image=image[k].astype(np.float32), label=label[k].astype(np.uint8))


def preprocess_synapse(raw, out):
    imgs = sorted(glob.glob(os.path.join(raw, "img", "img*.nii*")))
    if not imgs:
        raise FileNotFoundError(f"no img*.nii(.gz) under {raw}/img")
    for p in imgs:
        num = re.search(r"img(\d+)", os.path.basename(p)).group(1)
        lab_path = os.path.join(raw, "label", os.path.basename(p).replace("img", "label"))
        image, spacing = _load_nifti(p)
        label, _ = _load_nifti(lab_path)
        image = (np.clip(image.astype(np.float32), -125.0, 275.0) + 125.0) / 400.0
        mapped = np.zeros(label.shape, dtype=np.uint8)
        for src, dst in BTCV_TO_PROTOCOL.items():
            mapped[label == src] = dst
        vid = f"case{int(num):04d}"
        _write(out, vid, image, mapped, spacing)
        print(f"{vid}: {image.shape} spacing={spacing}")


def _acdc_frames(patient_dir):
    """Return the two annotated frame numbers (ED, ES) from Info.cfg."""
    info = {}
    with open(os.path.join(patient_dir, "Info.cfg")) as f:
        for line in f:
            if ":" in line:
                k, v = line.split(":", 1)
                info[k.strip()] = v.strip()
    return int(info["ED"]), int(info["ES"])


def preprocess_acdc(raw, out):
    patients = sorted(d for d in glob.glob(os.path.join(raw, "*", "patient*")) if os.path.isdir(d))
    if not patients:
        patients = sorted(d for d in glob.glob(os.path.join(raw, "patient*")) if os.path.isdir(d))
    if not patients:
        raise FileNotFoundError(f"no patientXXX folders under {raw}")
    for pdir in patients:
        pid = os.path.basename(pdir)
        for frame in _acdc_frames(pdir):
            stem = os.path.join(pdir, f"{pid}_frame{frame:02d}")
            image, spacing = _load_nifti(glob.glob(stem + ".nii*")[0])
            label, _ = _load_nifti(glob.glob(stem + "_gt.nii*")[0])
            image = image.astype(np.float32)
            image = (image - image.mean()) / (image.std() + 1e-8)
            vid = f"{pid}_frame{frame:02d}"
            _write(out, vid, image, label.astype(np.uint8), spacing)
            print(f"{vid}: {image.shape} spacing={spacing}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["synapse", "acdc"])
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    (preprocess_synapse if a.dataset == "synapse" else preprocess_acdc)(a.raw, a.out)
