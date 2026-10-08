

import argparse
import csv
import time

import torch
from torch.utils.flop_counter import FlopCounterMode

from adapmamba.configs import get_config
from adapmamba.model import AdapMambaUNet
from adapmamba.ss2d import SS2D


def scan_flops(model, x):
    shapes = []
    hooks = [m.register_forward_hook(lambda mod, inp, out: shapes.append((inp[0].shape[1], inp[0].shape[2], mod.d_inner, mod.d_state)))
             for m in model.modules() if isinstance(m, SS2D)]
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    total = 0
    for H, W, D, N in shapes:
        L, KD = H * W, 4 * D
        total += 9 * L * KD * N + L * KD
    return total


def dense_flops(model, x):
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        model(x)
    return fc.get_total_flops()


def latency_memory(model, x, warmup, iters):
    dev = x.device
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        if dev.type == "cuda":
            torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000 / iters
    mem = torch.cuda.max_memory_allocated() / 2 ** 30 if dev.type == "cuda" else float("nan")
    return ms, mem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["full"])
    ap.add_argument("--resolutions", nargs="+", type=int, default=[224])
    ap.add_argument("--num_classes", type=int, default=9)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--no_latency", action="store_true")
    ap.add_argument("--out", default="cost.csv")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backend = "auto" if device.type == "cuda" else "torch"

    rows = []
    for name in a.configs:
        model = AdapMambaUNet(a.num_classes, scan_backend=backend, **get_config(name)).eval()
        params = sum(p.numel() for p in model.parameters())
        for res in a.resolutions:
            x = torch.randn(1, 3, res, res)
            d = dense_flops(model, x)            # counted on CPU: device-independent
            s = scan_flops(model, x)
            row = {"config": name, "resolution": res, "params_M": params / 1e6,
                   "dense_GFLOPs": d / 1e9, "scan_GFLOPs": s / 1e9, "total_GFLOPs": (d + s) / 1e9,
                   "total_GMACs": (d + s) / 2e9, "latency_ms": float("nan"), "peak_mem_GB": float("nan"),
                   "device": str(device), "scan_backend": backend}
            if not a.no_latency:
                m = model.to(device)
                row["latency_ms"], row["peak_mem_GB"] = latency_memory(m, x.to(device), a.warmup, a.iters)
                model.cpu()
            rows.append(row)
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})

    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
