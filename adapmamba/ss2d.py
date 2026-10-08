"""
2D Selective Scan (SS2D) and Vision State Space (VSS) block.

For each VSS block, the input is projected into two branches. One branch passes
through a depth-wise 3x3 convolution and SiLU, and is then scanned in four
directions: row-major, column-major, and the reverse of each. Every direction is
processed by the selective S6 operator, whose step size, input matrix B and
output matrix C are computed from the input (Eqs. 1-3 of the manuscript). The
four outputs are mapped back to 2D and summed. The second branch acts as a SiLU
gate; the gated result is projected and added to the residual stream.

Selective-scan backends
  "cuda"  : mamba_ssm.ops.selective_scan_interface.selective_scan_fn, the fused
            CUDA kernel also used by VM-UNet. Required for practical training.
  "torch" : an exact sequential reference in pure PyTorch. Slow; used for CPU
            tests and as the numerical reference.
  "auto"  : "cuda" when mamba_ssm is installed and the input is on a GPU,
            otherwise "torch".
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # optional fused kernel
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _cuda_selective_scan
except Exception:  # pragma: no cover - depends on the environment
    _cuda_selective_scan = None


def cuda_scan_available() -> bool:
    return _cuda_selective_scan is not None


# ---------------------------------------------------------------------------
# Selective scan
# ---------------------------------------------------------------------------

def selective_scan_ref(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
    """Exact sequential selective scan.

    u, delta : (b, d, l)     input sequence and step size
    A        : (d, n)        state matrix (negative real)
    B, C     : (b, g, n, l)  input-dependent input/output matrices, g groups
    D        : (d,)          skip connection
    Returns  : (b, d, l)

    Implements h_t = exp(delta_t * A) h_{t-1} + delta_t * B_t * u_t and
    y_t = C_t h_t + D u_t, which is the zero-order-hold discretisation used by
    Mamba. Memory is O(b*d*n) because the recurrence is evaluated step by step.
    """
    dtype = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias.float()[None, :, None]
    if delta_softplus:
        delta = F.softplus(delta)
    b, d, l = u.shape
    n = A.shape[1]
    g = B.shape[1]
    B = B.float().repeat_interleave(d // g, dim=1)  # (b, d, n, l)
    C = C.float().repeat_interleave(d // g, dim=1)  # (b, d, n, l)
    A = A.float()

    h = u.new_zeros(b, d, n)
    ys = []
    for t in range(l):
        dt = delta[:, :, t, None]                                  # (b, d, 1)
        h = torch.exp(dt * A) * h + dt * B[:, :, :, t] * u[:, :, t, None]
        ys.append((h * C[:, :, :, t]).sum(-1))                     # (b, d)
    y = torch.stack(ys, dim=-1)                                    # (b, d, l)
    if D is not None:
        y = y + u * D.float()[None, :, None]
    return y.to(dtype)


def selective_scan(u, delta, A, B, C, D, delta_bias, backend="auto"):
    use_cuda = backend == "cuda" or (backend == "auto" and _cuda_selective_scan is not None and u.is_cuda)
    if use_cuda:
        if _cuda_selective_scan is None:
            raise RuntimeError("scan_backend='cuda' requested but mamba_ssm is not installed")
        return _cuda_selective_scan(u, delta, A, B, C, D, None, delta_bias, True)
    return selective_scan_ref(u, delta, A, B, C, D, delta_bias, True)


# ---------------------------------------------------------------------------
# SS2D
# ---------------------------------------------------------------------------

class SS2D(nn.Module):
    """VMamba v0 2D Selective Scan. Input and output: (B, H, W, C)."""

    K = 4  # scan directions

    def __init__(self, d_model, d_state=16, d_conv=3, expand=2, dt_rank="auto",
                 dt_min=0.001, dt_max=0.1, dt_scale=1.0, dt_init_floor=1e-4,
                 scan_backend="auto"):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else int(dt_rank)
        self.scan_backend = scan_backend
        K, D, N, R = self.K, self.d_inner, d_state, self.dt_rank

        self.in_proj = nn.Linear(d_model, 2 * D, bias=False)
        self.conv2d = nn.Conv2d(D, D, d_conv, padding=(d_conv - 1) // 2, groups=D, bias=True)
        self.act = nn.SiLU()

        # Input-dependent projections to (delta, B, C), one per direction.
        x_proj = [nn.Linear(D, R + 2 * N, bias=False) for _ in range(K)]
        self.x_proj_weight = nn.Parameter(torch.stack([m.weight for m in x_proj], dim=0))  # (K, R+2N, D)

        # Step-size projection, initialised as in Mamba.
        dt_w, dt_b = [], []
        for _ in range(K):
            w = torch.empty(D, R)
            std = R ** -0.5 * dt_scale
            nn.init.uniform_(w, -std, std)
            dt = torch.exp(torch.rand(D) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
            dt = dt.clamp(min=dt_init_floor)
            dt_w.append(w)
            dt_b.append(dt + torch.log(-torch.expm1(-dt)))  # inverse softplus
        self.dt_projs_weight = nn.Parameter(torch.stack(dt_w, dim=0))  # (K, D, R)
        self.dt_projs_bias = nn.Parameter(torch.stack(dt_b, dim=0))    # (K, D)

        A = torch.arange(1, N + 1, dtype=torch.float32).repeat(K * D, 1)
        self.A_logs = nn.Parameter(torch.log(A))                        # (K*D, N)
        self.Ds = nn.Parameter(torch.ones(K * D))                        # (K*D,)
        # VMamba excludes these two from weight decay.
        self.A_logs._no_weight_decay = True
        self.Ds._no_weight_decay = True

        self.out_norm = nn.LayerNorm(D)
        self.out_proj = nn.Linear(D, d_model, bias=False)

    def _scan_2d(self, x):
        """x: (B, D, H, W) -> (B, D, H*W), summed over the four directions."""
        Bsz, D, H, W = x.shape
        L, K = H * W, self.K
        x_hw = x.reshape(Bsz, D, L)                                   # row-major
        x_wh = x.transpose(2, 3).contiguous().reshape(Bsz, D, L)      # column-major
        xs = torch.stack([x_hw, x_wh], dim=1)
        xs = torch.cat([xs, xs.flip(-1)], dim=1)                      # (B, K, D, L)

        x_dbl = torch.einsum("bkdl,kcd->bkcl", xs, self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("bkrl,kdr->bkdl", dts, self.dt_projs_weight)

        out = selective_scan(
            xs.float().reshape(Bsz, K * D, L),
            dts.contiguous().float().reshape(Bsz, K * D, L),
            -torch.exp(self.A_logs.float()),
            Bs.float().contiguous(),
            Cs.float().contiguous(),
            self.Ds.float(),
            self.dt_projs_bias.float().reshape(-1),
            backend=self.scan_backend,
        ).reshape(Bsz, K, D, L)

        inv = out[:, 2:4].flip(-1)
        y_wh = out[:, 1].reshape(Bsz, D, W, H).transpose(2, 3).contiguous().reshape(Bsz, D, L)
        y_invwh = inv[:, 1].reshape(Bsz, D, W, H).transpose(2, 3).contiguous().reshape(Bsz, D, L)
        return out[:, 0] + inv[:, 0] + y_wh + y_invwh

    def forward(self, x):
        Bsz, H, W, _ = x.shape
        x, z = self.in_proj(x).chunk(2, dim=-1)
        x = self.act(self.conv2d(x.permute(0, 3, 1, 2).contiguous()))
        y = self._scan_2d(x)                                          # (B, D, L)
        y = y.transpose(1, 2).contiguous().reshape(Bsz, H, W, -1)
        y = self.out_norm(y) * F.silu(z)
        return self.out_proj(y)


class DropPath(nn.Module):
    """Per-sample stochastic depth."""

    def __init__(self, p=0.0):
        super().__init__()
        self.p = float(p)

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep) / keep
        return x * mask


class VSSBlock(nn.Module):
    """VMamba v0 VSS block: x + DropPath(SS2D(LN(x))). Input/output (B, H, W, C)."""

    def __init__(self, hidden_dim, drop_path=0.0, d_state=16, scan_backend="auto"):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.self_attention = SS2D(hidden_dim, d_state=d_state, scan_backend=scan_backend)
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        return x + self.drop_path(self.self_attention(self.ln_1(x)))
