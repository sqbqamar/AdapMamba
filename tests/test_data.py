"""Data tests: BTCV label remapping and kidney-label swap under horizontal flip."""
import sys, pathlib, random
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from adapmamba.data import augment, SYNAPSE_KIDNEY_L, SYNAPSE_KIDNEY_R
from preprocess import BTCV_TO_PROTOCOL


class ForceFlipOnly:
    """rng stub: first draw < 0.5 (flip), later draws >= 0.5 (no rotation, no scaling)."""
    def __init__(self): self.n = 0
    def random(self):
        self.n += 1
        return 0.0 if self.n == 1 else 0.9
    def uniform(self, a, b): return a


def main():
    # 1. BTCV -> protocol mapping matches the standard 8-organ ordering
    names = {8: "aorta", 4: "gallbladder", 1: "spleen", 3: "left kidney", 2: "right kidney",
             6: "liver", 7: "stomach", 11: "pancreas"}
    expected = {"aorta": 1, "gallbladder": 2, "spleen": 3, "left kidney": 4, "right kidney": 5,
                "liver": 6, "stomach": 7, "pancreas": 8}
    for btcv, organ in names.items():
        assert BTCV_TO_PROTOCOL[btcv] == expected[organ], organ
    print("BTCV label mapping: OK")

    # 2. Flip swaps kidney labels (Synapse) and leaves them alone (ACDC)
    lbl = np.zeros((8, 8), np.uint8)
    lbl[3:5, 1:3] = SYNAPSE_KIDNEY_L          # left kidney on image-left
    lbl[3:5, 5:7] = SYNAPSE_KIDNEY_R          # right kidney on image-right
    img = np.zeros((8, 8), np.float32)
    _, out = augment(img, lbl.copy(), True, ForceFlipOnly())
    assert (out[3:5, 1:3] == SYNAPSE_KIDNEY_L).all() and (out[3:5, 5:7] == SYNAPSE_KIDNEY_R).all()
    _, out_noswap = augment(img, lbl.copy(), False, ForceFlipOnly())
    assert (out_noswap[3:5, 1:3] == SYNAPSE_KIDNEY_R).all()
    print("horizontal flip with kidney swap keeps each side's label; without swap it mirrors: OK")

    # 3. Other organ labels are preserved by the swap
    lbl2 = np.full((4, 4), 6, np.uint8)
    _, out2 = augment(np.zeros((4, 4), np.float32), lbl2, True, ForceFlipOnly())
    assert (out2 == 6).all()
    print("PASS")


if __name__ == "__main__":
    main()
