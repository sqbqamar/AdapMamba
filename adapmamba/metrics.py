
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage


def dice(pred, gt):
    ps, gs = pred.sum(), gt.sum()
    if ps == 0 and gs == 0:
        return 1.0
    if ps == 0 or gs == 0:
        return 0.0
    return float(2.0 * np.logical_and(pred, gt).sum() / (ps + gs))


def _surface_distances(a, b, spacing):
    """Distances from the surface voxels of a to the surface of b (medpy __surface_distances)."""
    fp = ndimage.generate_binary_structure(a.ndim, 1)
    a_border = a ^ ndimage.binary_erosion(a, structure=fp, iterations=1)
    b_border = b ^ ndimage.binary_erosion(b, structure=fp, iterations=1)
    dt = ndimage.distance_transform_edt(~b_border, sampling=spacing)
    return dt[a_border]


def hd95(pred, gt, spacing=None):
    d1 = _surface_distances(pred, gt, spacing)
    d2 = _surface_distances(gt, pred, spacing)
    return float(np.percentile(np.hstack([d1, d2]), 95))


def assd(pred, gt, spacing=None):
    """Average symmetric surface distance: mean of the pooled surface distances in
    both directions (current medpy.metric.binary.assd definition)."""
    return float(np.concatenate([_surface_distances(pred, gt, spacing),
                                 _surface_distances(gt, pred, spacing)]).mean())


def evaluate_volume(pred, gt, num_classes, spacing=None, protocol="mm"):
    """Per-class metrics for one volume. Classes 1..num_classes-1."""
    out = {"dice": [], "hd95": [], "assd": []}
    for c in range(1, num_classes):
        p, g = pred == c, gt == c
        if protocol == "transunet":
            if p.sum() > 0 and g.sum() > 0:
                out["dice"].append(dice(p, g))
                out["hd95"].append(hd95(p, g, None))
                out["assd"].append(assd(p, g, None))
            elif p.sum() > 0:
                out["dice"].append(1.0); out["hd95"].append(0.0); out["assd"].append(0.0)
            else:
                out["dice"].append(0.0); out["hd95"].append(0.0); out["assd"].append(0.0)
        else:
            out["dice"].append(dice(p, g))
            if p.sum() > 0 and g.sum() > 0:
                out["hd95"].append(hd95(p, g, spacing))
                out["assd"].append(assd(p, g, spacing))
            else:
                out["hd95"].append(float("nan")); out["assd"].append(float("nan"))
    return out


@torch.no_grad()
def predict_volume_probs(models, volume, img_size, device, batch_size=16):
    """Slice-by-slice inference. Returns softmax probabilities averaged over the
    given models (one model per seed), at network resolution: (D, K, S, S)."""
    D, H, W = volume.shape
    x = torch.from_numpy(volume).float().unsqueeze(1)                 # (D, 1, H, W)
    x = F.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)
    x = x.repeat(1, 3, 1, 1)
    probs = None
    for m in models:
        m.eval()
        chunks = []
        for i in range(0, D, batch_size):
            chunks.append(torch.softmax(m(x[i:i + batch_size].to(device)), dim=1).float().cpu())
        p = torch.cat(chunks)
        probs = p if probs is None else probs + p
    return probs / len(models)


def probs_to_label(probs, out_hw):
    """Argmax at network resolution, then nearest-neighbour resize to the original grid."""
    lab = probs.argmax(1, keepdim=True).float()                        # (D, 1, S, S)
    lab = F.interpolate(lab, size=out_hw, mode="nearest")
    return lab[:, 0].numpy().astype(np.uint8)
