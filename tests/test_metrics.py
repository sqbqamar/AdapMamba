"""Metric tests: agreement with medpy (if installed) and analytic sanity checks."""
import sys, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from adapmamba.metrics import dice, hd95, assd, evaluate_volume


def sphere(shape, c, r):
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    return (z - c[0]) ** 2 + (y - c[1]) ** 2 + (x - c[2]) ** 2 <= r * r


def main():
    rng = np.random.default_rng(0)
    a = sphere((30, 40, 40), (15, 20, 20), 8)
    b = sphere((30, 40, 40), (15, 22, 19), 7)
    sp = (2.5, 0.8, 0.8)
    try:
        from medpy.metric import binary as mb
        for spacing in (None, sp):
            e1 = abs(hd95(a, b, spacing) - mb.hd95(a, b, voxelspacing=spacing))
            e2 = abs(assd(a, b, spacing) - mb.assd(a, b, voxelspacing=spacing))
            print(f"spacing={spacing}: |hd95-medpy|={e1:.2e}  |assd-medpy|={e2:.2e}")
            assert e1 < 1e-9 and e2 < 1e-9
        for _ in range(5):
            p = rng.random((12, 20, 20)) > 0.7
            g = rng.random((12, 20, 20)) > 0.6
            assert abs(hd95(p, g, sp) - mb.hd95(p, g, voxelspacing=sp)) < 1e-9
            assert abs(dice(p, g) - mb.dc(p, g)) < 1e-12
        print("agreement with medpy on spheres and random masks: OK")
    except ImportError:
        print("medpy not installed: skipping cross-check")
    # identical masks -> dice 1, hd95 0
    assert dice(a, a) == 1.0 and hd95(a, a) == 0.0
    # empty-prediction handling under both protocols
    gt = np.zeros((4, 8, 8), np.uint8); gt[1:3, 2:5, 2:5] = 1
    pred = np.zeros_like(gt)
    mm = evaluate_volume(pred, gt, 2, sp, "mm")
    tu = evaluate_volume(pred, gt, 2, sp, "transunet")
    assert mm["dice"] == [0.0] and np.isnan(mm["hd95"][0])
    assert tu["dice"] == [0.0] and tu["hd95"] == [0.0]
    print("empty-prediction rules for 'mm' and 'transunet' protocols: OK")
    print("PASS")


if __name__ == "__main__":
    main()
