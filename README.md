# AdapMamba-UNet

**Adaptive Multi-Scale Feature Aggregation and Hierarchical Gated Fusion for Mamba-Based Medical Image Segmentation**

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-22C55E)](LICENSE)

---

## Highlights

AdapMamba-UNet addresses three persistent weaknesses of current Mamba-based U-Nets:

| Problem in existing methods | Our solution |
|---|---|
| Static skip connections pass noisy encoder features unchanged | **HCAG** -> dual-branch channel + spatial gating |
| Single-scale terminal decoder prediction misses small organs | **APFA** -> multi-dilation pyramid aggregation with learnable α weights |
| Patch-expanding upsampling blurs fine boundaries | **PAU** -> CARAFE content-aware reassembly + position-sensitive attention |




## Installation

```bash
git clone https://github.com/sqbqamar/AdapMamba.git
cd AdapMamba
pip install -r requirements.txt
```

**Optional** — faster SS2D via CUDA kernel (requires CUDA 11.6+):
```bash
pip install mamba-ssm causal-conv1d
```

---

## Dataset Preparation

### Synapse Multi-Organ CT

Download the TransUNet-formatted preprocessed data from the
[official TransUNet repository](https://github.com/Beckschen/TransUNet#dataset).

Expected layout:
```
data/Synapse/
├── train_npz/
│   ├── case0001_slice000.npz      # keys: "image" (H×W), "label" (H×W)
│   └── ...
├── test_vol_h5/
│   ├── case0008.npy.h5            # keys: "image" (D×H×W), "label" (D×H×W)
│   └── ...
├── train.txt                      # one slice name per line
└── test_vol.txt                   # one case name per line
```

### ACDC Cardiac MRI

Download from [ACDC challenge](https://acdc.creatis.insa-lyon.fr/) and preprocess
following the same `.npz` slice convention as TransUNet.

```
data/ACDC/
├── train/   *.npz    # keys: "image" (H×W), "label" (H×W, values 0–3)
├── val/     *.npz
└── test/    *.npz
```

---

## Training

### Synapse

```bash
python train.py \
    --dataset      synapse \
    --data_root    data/Synapse \
    --output_dir   runs/synapse \
    --num_classes  9 \
    --img_size     224 \
    --epochs       300 \
    --batch_size   24 \
    --lr           3e-4 \
    --lambda_dice  1.0 \
    --lambda_ce    1.0 \
    --drop_path    0.1
```

### ACDC

```bash
python train.py \
    --dataset      acdc \
    --data_root    data/ACDC \
    --output_dir   runs/acdc \
    --num_classes  4 \
    --img_size     256 \
    --epochs       200 \
    --batch_size   32 \
    --lr           1e-4 \
    --lambda_dice  0.6 \
    --lambda_ce    0.4
```

---

## Evaluation

```bash
# Synapse — full volume DSC + HD95
python test.py \
    --dataset    synapse \
    --data_root  data/Synapse \
    --checkpoint runs/synapse/ckpt_best.pth \
    --num_classes 9 --img_size 224

# ACDC — per-structure DSC
python test.py \
    --dataset    acdc \
    --data_root  data/ACDC \
    --checkpoint runs/acdc/ckpt_best.pth \
    --num_classes 4 --img_size 256

# Save predicted masks
python test.py ... --save_preds --save_dir predictions/
```

---

## Project Structure

```
AdapMamba-UNet/
├── model.py          # VSSBlock · PAU · HCAG · APFA · AdapMambaUNet
├── dataset.py        # Synapse & ACDC loaders with on-the-fly augmentation
├── train.py          # AMP training with cosine / OneCycleLR schedulers
├── test.py           # Volume-level DSC + HD95 evaluation
├── utils.py          # DiceLoss · CombinedLoss · metrics · checkpoint I/O
├── requirements.txt
└── README.md
```

---

## Module Details

### VSS Block

The Vision State Space Block (from VMamba) is the core feature extractor.
`SS2D` scans the feature map as four 1-D sequences (↗ ↘ ↙ ↖) and merges
the outputs, giving every pixel access to global context at **O(N)** cost.



### PAU — Progressive Adaptive Upsampling

Replaces patch-expanding layers.

| Step | Operation | Effect |
|---|---|---|
| 1 | CARAFE kernel head (5×5) | Predicts per-location reassembly kernel conditioned on local content |
| 2 | Content-aware spatial reassembly | Upsampled pixel = weighted neighbourhood sum using predicted kernel |
| 3 | Channel attention (GAP→FC→Sig) | Identifies task-relevant channels |
| 4 | Spatial attention (DWConv→Sig) | Highlights boundary-rich regions |
| 5 | `X_out = X_up ⊗ σ(A_c) ⊗ σ(A_s)` | Multiplicative gating |

### HCAG — Hierarchical Cross-Attention Gate

Filters noisy encoder skip features before they enter the decoder.


### APFA — Adaptive Pyramid Feature Aggregation

Collects all four decoder outputs, aligns to H/4, and fuses them.

```
F_d1 ─┐
F_d2↑ ─┤ concat → DW-Sep Conv d=1 ─→ × α₁ ─┐
F_d3↑ ─┤         DW-Sep Conv d=2 ─→ × α₂ ─┤─ sum → Conv1×1+BN+SiLU
F_d4↑ ─┘         DW-Sep Conv d=4 ─→ × α₃ ─┘
                  (α softmax-normalised, learnable)
```



## Citation

If you find this work useful, please cite:

```bibtex
@article{adapmamba2025,
  title   = {Adaptive Multi-Scale Feature Aggregation and Hierarchical Gated Fusion for Mamba-Based Medical Image Segmentation},
  author  = {Saqib Qamar},
  journal = {},
  year    = {2026}
}
```

---

## License

Released under the [MIT License](LICENSE).
