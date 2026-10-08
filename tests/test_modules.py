"""Unit tests for CARAFE indexing, PAU/HCAG/APFA shapes and the APFA weight initialisation."""
import sys, pathlib
import torch, torch.nn.functional as F
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from adapmamba.modules import carafe_reassemble, PAU, HCAG, APFA, make_upsampler


def naive_carafe(x, kern, k, s):
    B, C, H, W = x.shape
    p = k // 2
    xp = F.pad(x, [p, p, p, p])
    out = torch.zeros(B, C, s * H, s * W)
    for oh in range(s * H):
        for ow in range(s * W):
            h, w = oh // s, ow // s
            for di in range(k):
                for dj in range(k):
                    out[:, :, oh, ow] += xp[:, :, h + di, w + dj] * kern[:, di * k + dj, oh, ow][:, None]
    return out


def test_carafe():
    torch.manual_seed(0)
    k, s = 5, 2
    x = torch.randn(2, 3, 4, 5)
    kern = torch.softmax(torch.randn(2, k * k, s * 4, s * 5), dim=1)
    err = (carafe_reassemble(x, kern, k, s) - naive_carafe(x, kern, k, s)).abs().max().item()
    print(f"CARAFE vs brute force: max err {err:.2e}")
    assert err < 1e-5


def test_shapes():
    x = torch.randn(2, 64, 8, 8)
    for kind in ["pau", "carafe", "bilinear", "transposed", "patch_expand"]:
        y = make_upsampler(kind, 64, 32)(x)
        assert y.shape == (2, 32, 16, 16), (kind, y.shape)
    for mode in ["both", "channel", "spatial", "none"]:
        y = HCAG(32, mode=mode)(torch.randn(2, 32, 16, 16), torch.randn(2, 32, 16, 16))
        assert y.shape == (2, 32, 16, 16), mode
    feats = [torch.randn(2, 16, 16, 16), torch.randn(2, 32, 8, 8),
             torch.randn(2, 64, 4, 4), torch.randn(2, 128, 2, 2)]
    for dil in [(1,), (1, 2), (1, 2, 4), (1, 3, 6), (1, 2, 4, 8)]:
        a = APFA([16, 32, 64, 128], 16, dilations=dil)
        assert a(feats).shape == (2, 16, 16, 16)
        assert torch.allclose(a.alphas(), torch.full((len(dil),), 1.0 / len(dil)))
    fixed = APFA([16, 32, 64, 128], 16, learnable=False)
    assert "logits" not in dict(fixed.named_parameters())
    print("module shapes, HCAG modes, APFA configurations and alpha=1/n initialisation: OK")


if __name__ == "__main__":
    test_carafe()
    test_shapes()
    print("PASS")
