
import sys, pathlib, time
import torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from adapmamba.model import AdapMambaUNet
from adapmamba.configs import CONFIGS


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def main(ckpt):
    torch.manual_seed(0)
    # 1. Pretrained encoder: every encoder tensor must be covered.
    m = AdapMambaUNet(num_classes=9, scan_backend="torch", **CONFIGS["full"])
    rep = m.load_pretrained_encoder(ckpt)
    print(f"pretrained: {rep['loaded']}/{rep['encoder_tensors']} encoder tensors loaded, "
          f"missing={rep['missing']}; unused checkpoint groups: {len(rep['unused_checkpoint_groups'])}")
    assert rep["loaded"] == rep["encoder_tensors"]
    # The loaded weights really are the checkpoint's (spot check stage-3 block 2).
    src = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
    key = "layers.2.blocks.2.self_attention.x_proj_weight"
    assert torch.equal(m.encoder.state_dict()[key], src[key])

    # 2. Forward + backward for every configuration (64x64 keeps the CPU scan fast).
    x = torch.randn(2, 3, 64, 64)
    t0 = time.time()
    for name, cfg in CONFIGS.items():
        net = AdapMambaUNet(num_classes=9, scan_backend="torch", **cfg)
        out = net(x)
        assert out.shape == (2, 9, 64, 64), (name, out.shape)
        out.mean().backward()
        assert all(p.grad is not None for p in net.parameters() if p.requires_grad), name
    print(f"forward/backward OK for all {len(CONFIGS)} configurations ({time.time()-t0:.0f}s)")

    # 3. Parameter counts (independent of input size).
    print(f"\n{'configuration':28}{'params (M)':>12}")
    for name in ["baseline", "step_pau", "step_pau_hcag", "step_pau_hcag_apfa_fixed", "full"]:
        net = AdapMambaUNet(num_classes=9, scan_backend="torch", **CONFIGS[name])
        print(f"{name:28}{n_params(net)/1e6:>12.3f}")
    full = AdapMambaUNet(num_classes=9, scan_backend="torch", **CONFIGS["full"])
    parts = {"encoder": full.encoder, "decoder VSS": [full.dec4, full.dec3, full.dec2, full.dec1],
             "PAU (x5)": [full.up3, full.up2, full.up1, full.final_up1, full.final_up2],
             "HCAG (x3)": [full.fuse3, full.fuse2, full.fuse1], "APFA": full.apfa, "head": full.head}
    print()
    for k, v in parts.items():
        mods = v if isinstance(v, list) else [v]
        print(f"  {k:14}{sum(n_params(mm) for mm in mods)/1e6:>10.3f} M")
    print("PASS")


if __name__ == "__main__":
    main(sys.argv[1])
