"""
Decoder modules of AdapMamba-UNet: PAU, HCAG and APFA, plus the alternatives
used in the ablation studies. All tensors are (B, C, H, W) unless stated.

Equation numbers refer to the manuscript.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # optional fused CARAFE op
    from mmcv.ops import carafe as _mmcv_carafe
except Exception:  # pragma: no cover
    _mmcv_carafe = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class ChannelLayerNorm(nn.Module):
    """LayerNorm over the channel dimension of a (B, C, H, W) tensor."""

    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(c, eps=eps)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


def carafe_reassemble(x, kernels, k, scale=2):
    """Content-aware reassembly (CARAFE, Wang et al., ICCV 2019).

    x       : (B, C, H, W) features to reassemble
    kernels : (B, k*k, sH, sW) softmax-normalised kernels, one per output location
    Output  : (B, C, sH, sW). Output location (s*h + i, s*w + j) is the kernel-
              weighted sum over the k x k neighbourhood of input location (h, w).
    """
    if _mmcv_carafe is not None and x.is_cuda:
        return _mmcv_carafe(x.contiguous(), kernels.contiguous(), k, 1, scale)
    B, C, H, W = x.shape
    nb = F.unfold(x, kernel_size=k, padding=k // 2).view(B, C, k * k, H, W)
    kern = F.pixel_unshuffle(kernels, scale).view(B, k * k, scale * scale, H, W)
    out = torch.einsum("bcnhw,bnshw->bcshw", nb, kern).reshape(B, C * scale * scale, H, W)
    return F.pixel_shuffle(out, scale)


# ---------------------------------------------------------------------------
# PAU: Progressive Adaptive Upsampling (Section 3.3, Eq. 4)
# ---------------------------------------------------------------------------

class PAU(nn.Module):
    """2x upsampling: CARAFE reassembly followed by position-sensitive recalibration.

    Stage 1 (content-aware reassembly). A 1x1 convolution compresses X_in from
    C_in to C_mid = C_in / 4; a 3x3 convolution then predicts the k^2-dimensional
    kernel for every output location (s^2 * k^2 channels at input resolution,
    rearranged by pixel shuffle and softmax-normalised). The kernels reassemble the
    content features into X_up (2H x 2W x C_out). The content features are mapped
    from C_in to C_out by a 1x1 convolution; because the reassembly kernels sum to
    one, this projection commutes with reassembly and is applied before it to
    reduce cost.

    Stage 2 (position-sensitive recalibration). Layer normalisation is applied to
    X_up. A channel pathway (global average pooling -> FC -> ReLU -> FC) gives A_c,
    and a spatial pathway (3x3 depth-wise convolution -> 1x1 projection to one
    channel) gives A_s. Output (Eq. 4): X_out = X_up * sigmoid(A_c) * sigmoid(A_s).

    attention=False gives the "CARAFE only" ablation (stage 1 only).
    """

    def __init__(self, c_in, c_out, k_up=5, scale=2, reduction=8, attention=True):
        super().__init__()
        self.k, self.scale, self.attention = k_up, scale, attention
        c_mid = max(c_in // 4, 1)
        self.compressor = nn.Conv2d(c_in, c_mid, 1)
        self.kernel_encoder = nn.Conv2d(c_mid, scale * scale * k_up * k_up, 3, padding=1)
        self.content_proj = nn.Conv2d(c_in, c_out, 1)
        if attention:
            r = max(c_out // reduction, 4)
            self.norm = ChannelLayerNorm(c_out)
            self.channel_att = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Linear(c_out, r), nn.ReLU(inplace=True), nn.Linear(r, c_out))
            self.spatial_att = nn.Sequential(
                nn.Conv2d(c_out, c_out, 3, padding=1, groups=c_out),
                nn.Conv2d(c_out, 1, 1))

    def forward(self, x):
        kern = self.kernel_encoder(self.compressor(x))            # (B, s^2 k^2, H, W)
        kern = F.softmax(F.pixel_shuffle(kern, self.scale), dim=1)  # (B, k^2, sH, sW)
        x_up = carafe_reassemble(self.content_proj(x), kern, self.k, self.scale)
        if not self.attention:
            return x_up
        x_up = self.norm(x_up)
        a_c = self.channel_att(x_up)[:, :, None, None]            # (B, C, 1, 1)
        a_s = self.spatial_att(x_up)                              # (B, 1, sH, sW)
        return x_up * torch.sigmoid(a_c) * torch.sigmoid(a_s)


# ---------------------------------------------------------------------------
# Upsampling alternatives (Table: upsampling strategy comparison)
# ---------------------------------------------------------------------------

class BilinearUp(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.proj = nn.Conv2d(c_in, c_out, 1)

    def forward(self, x):
        return self.proj(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))


class TransposedUp(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.up = nn.ConvTranspose2d(c_in, c_out, 2, stride=2)

    def forward(self, x):
        return self.up(x)


class PatchExpand(nn.Module):
    """Patch expanding (Swin-UNet / VM-UNet): Linear to 4*C_out, rearrange, LayerNorm."""

    def __init__(self, c_in, c_out):
        super().__init__()
        self.expand = nn.Linear(c_in, 4 * c_out, bias=False)
        self.norm = nn.LayerNorm(c_out)

    def forward(self, x):
        x = self.expand(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)   # (B, 4C, H, W)
        x = F.pixel_shuffle(x, 2)                                     # (B, C, 2H, 2W)
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


def make_upsampler(kind, c_in, c_out, k_up=5):
    if kind == "pau":
        return PAU(c_in, c_out, k_up=k_up, attention=True)
    if kind == "carafe":
        return PAU(c_in, c_out, k_up=k_up, attention=False)
    if kind == "bilinear":
        return BilinearUp(c_in, c_out)
    if kind == "transposed":
        return TransposedUp(c_in, c_out)
    if kind == "patch_expand":
        return PatchExpand(c_in, c_out)
    raise ValueError(f"unknown upsampler: {kind}")


# ---------------------------------------------------------------------------
# HCAG: Hierarchical Channel-spatial Attention Gate (Section 3.4, Eqs. 5-6)
# ---------------------------------------------------------------------------

class HCAG(nn.Module):
    """Conditional channel-spatial gating of the encoder skip feature.

    X_cat = [X_up; X_skip] (2C channels).
    Channel branch : e_c  = sigmoid(FC2(GELU(FC1(GAP(X_cat))))), 2C -> 2C/r -> C.
    Spatial branch : A_sp = sigmoid(Conv1x1(GELU(DilConv3x3(X_cat)))), dilation d.
    Gate           : G = e_c * A_sp;  X'_skip = X_skip * G            (Eq. 5)
    Fusion         : X_fused = LN(Linear([X_up; X'_skip]))            (Eq. 6)

    mode selects the gate used in the gate-design ablation:
      "both"    : channel and spatial branches (HCAG)
      "channel" : channel branch only
      "spatial" : spatial branch only
      "none"    : no gate (concatenation + linear projection + LN)
    The dilated 3x3 convolution is depth-wise.
    """

    def __init__(self, c, r=8, dilation=2, mode="both"):
        super().__init__()
        if mode not in ("both", "channel", "spatial", "none"):
            raise ValueError(f"unknown HCAG mode: {mode}")
        self.mode = mode
        c2 = 2 * c
        if mode in ("both", "channel"):
            self.fc = nn.Sequential(nn.Linear(c2, max(c2 // r, 1)), nn.GELU(),
                                    nn.Linear(max(c2 // r, 1), c))
        if mode in ("both", "spatial"):
            self.spatial = nn.Sequential(
                nn.Conv2d(c2, c2, 3, padding=dilation, dilation=dilation, groups=c2),
                nn.GELU(),
                nn.Conv2d(c2, 1, 1))
        self.proj = nn.Linear(c2, c)
        self.norm = nn.LayerNorm(c)

    def forward(self, x_up, x_skip):
        x_cat = torch.cat([x_up, x_skip], dim=1)
        gate = None
        if self.mode in ("both", "channel"):
            gate = torch.sigmoid(self.fc(x_cat.mean(dim=(2, 3))))[:, :, None, None]
        if self.mode in ("both", "spatial"):
            a_sp = torch.sigmoid(self.spatial(x_cat))
            gate = a_sp if gate is None else gate * a_sp
        x_skip = x_skip if gate is None else x_skip * gate
        fused = torch.cat([x_up, x_skip], dim=1).permute(0, 2, 3, 1)
        return self.norm(self.proj(fused)).permute(0, 3, 1, 2)


# ---------------------------------------------------------------------------
# APFA: Adaptive Pyramid Feature Aggregation (Section 3.5, Eqs. 7-8)
# ---------------------------------------------------------------------------

class DWSepConv(nn.Module):
    """Dilated depth-wise separable convolution: DW 3x3 (dilation d) -> PW 1x1 -> BN -> SiLU."""

    def __init__(self, c_in, c_out, dilation):
        super().__init__()
        self.dw = nn.Conv2d(c_in, c_in, 3, padding=dilation, dilation=dilation, groups=c_in, bias=False)
        self.pw = nn.Conv2d(c_in, c_out, 1, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


class APFA(nn.Module):
    """Multi-scale aggregation of the four decoder outputs.

    Spatial alignment: F_d,2..F_d,4 are upsampled to H/4 x W/4 by bilinear
    interpolation and projected to C_final by 1x1 convolutions; F_d,1 is already at
    H/4 with C_final channels. The four maps are concatenated into X_APFA
    (4 * C_final channels).
    Multi-scale dilated fusion (Eq. 7): parallel dilated depth-wise separable
    branches, one per dilation rate, map X_APFA to X_d (4 * C_final channels each);
    X_fused = sum_d alpha_d * X_d with alpha = softmax(logits). The logits are
    global learnable scalars initialised to zero, so alpha starts at 1/n; they are
    shared across all images.
    Channel reduction (Eq. 8): X_out = SiLU(BN(Conv1x1(X_fused))), 4*C_final -> C_final.

    learnable=False fixes alpha = 1/n (the "Scale W." ablation column).
    """

    def __init__(self, in_channels, c_final, dilations=(1, 2, 4), learnable=True):
        super().__init__()
        self.align = nn.ModuleList([
            nn.Identity() if (i == 0 and c == c_final) else nn.Conv2d(c, c_final, 1)
            for i, c in enumerate(in_channels)])
        c_cat = c_final * len(in_channels)
        self.branches = nn.ModuleList([DWSepConv(c_cat, c_cat, d) for d in dilations])
        logits = torch.zeros(len(dilations))
        if learnable:
            self.logits = nn.Parameter(logits)
        else:
            self.register_buffer("logits", logits)
        self.reduce = nn.Sequential(nn.Conv2d(c_cat, c_final, 1, bias=False),
                                    nn.BatchNorm2d(c_final), nn.SiLU(inplace=True))

    def alphas(self):
        return F.softmax(self.logits, dim=0)

    def forward(self, feats):
        H, W = feats[0].shape[-2:]
        aligned = []
        for f, align in zip(feats, self.align):
            if f.shape[-2:] != (H, W):
                f = F.interpolate(f, size=(H, W), mode="bilinear", align_corners=False)
            aligned.append(align(f))
        x = torch.cat(aligned, dim=1)
        alpha = self.alphas()
        fused = sum(alpha[i] * branch(x) for i, branch in enumerate(self.branches))
        return self.reduce(fused)
