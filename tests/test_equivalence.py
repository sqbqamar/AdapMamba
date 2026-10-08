"""
Numerical equivalence of adapmamba.ss2d with the VMamba v0 reference used by VM-UNet.

"""
import sys, types, importlib.util, pathlib
import torch, torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapmamba.ss2d import SS2D, VSSBlock  # noqa: E402


def independent_scan(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                     delta_softplus=False, return_last_state=False):
    u, delta = u.float(), delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    b, d, l = u.shape
    g = B.shape[1]
    Bx = B.float().unsqueeze(2).expand(b, g, d // g, *B.shape[2:]).reshape(b, d, *B.shape[2:])
    Cx = C.float().unsqueeze(2).expand(b, g, d // g, *C.shape[2:]).reshape(b, d, *C.shape[2:])
    deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A.float()))
    deltaBu = torch.einsum("bdl,bdnl,bdl->bdln", delta, Bx, u)
    x = u.new_zeros(b, d, A.shape[1])
    ys = []
    for i in range(l):
        x = deltaA[:, :, i] * x + deltaBu[:, :, i]
        ys.append(torch.einsum("bdn,bdn->bd", x, Cx[:, :, :, i]))
    y = torch.stack(ys, dim=2)
    if D is not None:
        y = y + u * D.float()[:, None]
    return y


def load_reference(path):
    stub = types.ModuleType("mamba_ssm")
    ops = types.ModuleType("mamba_ssm.ops")
    ssi = types.ModuleType("mamba_ssm.ops.selective_scan_interface")
    ssi.selective_scan_fn = independent_scan
    ssi.selective_scan_ref = independent_scan
    sys.modules.update({"mamba_ssm": stub, "mamba_ssm.ops": ops,
                        "mamba_ssm.ops.selective_scan_interface": ssi})
    spec = importlib.util.spec_from_file_location("vmunet_vmamba", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(ref_path):
    torch.manual_seed(0)
    ref = load_reference(ref_path)
    worst = 0.0
    # Non-square inputs catch any H/W transposition error in the scan directions.
    for dim, (H, W) in [(32, (5, 7)), (48, (6, 4)), (96, (8, 8))]:
        r = ref.VSSBlock(hidden_dim=dim, d_state=16).eval()
        m = VSSBlock(dim, d_state=16, scan_backend="torch").eval()
        missing, unexpected = m.load_state_dict(r.state_dict(), strict=True), None
        x = torch.randn(2, H, W, dim)
        with torch.no_grad():
            a, b = r(x), m(x)
        err = (a - b).abs().max().item()
        worst = max(worst, err)
        print(f"VSSBlock dim={dim:3d} HxW={H}x{W}: max |ref - ours| = {err:.2e}")
    assert worst < 1e-4, f"SS2D does not match the reference (max err {worst:.2e})"
    print("PASS: adapmamba VSSBlock is numerically equivalent to the VMamba v0 reference")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "vmunet_vmamba.py")
