# -*- coding: utf-8 -*-
"""
Created on Feb  15 11:04:38 2026

@author: drsaq


model.py - AdapMamba-UNet

Full architecture in pure PyTorch (no einops, no mamba_ssm required).
Drop in `mamba_ssm.Mamba` for the production CUDA kernel if available.

Components

  PatchEmbed      4Ã—4 patch tokenisation
  PatchMerging    spatial â†“2Ã—, channels Ã—2
  VSSBlock        Vision State Space Block (SS2D approximated with
                  4-direction causal depth-wise convolutions)
  PAU             Progressive Adaptive Upsampling (CARAFE + dual attention)
  HCAG            Hierarchical Cross-Attention Gate (skip fusion)
  APFA            Adaptive Pyramid Feature Aggregation
  AdapMambaUNet   Full segmentation network
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# Utilities

class LayerNorm2d(nn.Module):
    """Channel-last LayerNorm for BCHW tensors."""
    def __init__(self, c: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(c, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class DropPath(nn.Module):
    """Per-sample stochastic depth."""
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep, device=x.device)) / keep
        return x * mask


class DWSepConv(nn.Module):
    """Depthwise-separable convolution with optional dilation."""
    def __init__(self, in_c: int, out_c: int, k: int = 3, d: int = 1):
        super().__init__()
        p = d * (k // 2)
        self.dw  = nn.Conv2d(in_c, in_c, k, padding=p, dilation=d,
                              groups=in_c, bias=False)
        self.pw  = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.bn  = nn.BatchNorm2d(out_c)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pw(self.dw(x))))


# SS2D â€” 2-D Selective Scan (pure-PyTorch approximation)
#
# Each of the 4 scan directions (row-fwd, row-bwd, col-fwd, col-bwd) is
# approximated by a causal depth-wise 1-D convolution.  This captures the
# directional inductive bias of the original SS2D at full PyTorch portability.
#
# To use the real CUDA kernel:
#   pip install mamba-ssm causal-conv1d
#   Replace SS2D with mamba_ssm.Mamba (same forward signature).

class SS2D(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2):
        super().__init__()
        self.d_inner = d_model * expand

        self.in_proj  = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv2d   = nn.Conv2d(self.d_inner, self.d_inner, 3,
                                   padding=1, groups=self.d_inner, bias=True)
        self.act      = nn.SiLU(inplace=True)

        # 4 directional causal depth-wise convolutions
        self.dir_convs = nn.ModuleList([
            nn.Conv1d(self.d_inner, self.d_inner, d_state,
                      padding=d_state - 1, groups=self.d_inner, bias=False)
            for _ in range(4)
        ])
        self.merge = nn.Conv2d(self.d_inner * 4, self.d_inner, 1, bias=False)
        self.norm  = nn.LayerNorm(self.d_inner)
        self.out   = nn.Linear(self.d_inner, d_model, bias=False)

    @staticmethod
    def _causal_conv(conv: nn.Conv1d, x: torch.Tensor,
                     flip: bool) -> torch.Tensor:
        if flip:
            x = x.flip(-1)
        y = conv(x)[..., :x.size(-1)]
        if flip:
            y = y.flip(-1)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : B H W C  â†’  B H W C"""
        B, H, W, C = x.shape

        xz       = self.in_proj(x)                        # B H W 2D
        x_b, z   = xz.chunk(2, dim=-1)                   # B H W D each

        # Local depth-wise context
        x_b = x_b.permute(0, 3, 1, 2)                    # B D H W
        x_b = self.act(self.conv2d(x_b))

        # 4 directional scans
        row = x_b.view(B, self.d_inner, H * W)
        col = x_b.permute(0, 1, 3, 2).contiguous().view(B, self.d_inner, W * H)

        d = [
            self._causal_conv(self.dir_convs[0], row,  False).view(B, self.d_inner, H, W),
            self._causal_conv(self.dir_convs[1], row,  True ).view(B, self.d_inner, H, W),
            self._causal_conv(self.dir_convs[2], col,  False).view(B, self.d_inner, W, H).permute(0,1,3,2),
            self._causal_conv(self.dir_convs[3], col,  True ).view(B, self.d_inner, W, H).permute(0,1,3,2),
        ]

        merged = self.merge(torch.cat(d, dim=1))          # B D H W
        merged = merged.permute(0, 2, 3, 1)               # B H W D

        y = self.norm(merged) * F.silu(z)
        return self.out(y)                                 # B H W C


# VSSBlock

class VSSBlock(nn.Module):
    """
    Vision State Space Block.
    LayerNorm â†’ [SS2D branch âŠ— Gate branch] â†’ residual
    """
    def __init__(self, dim: int, d_state: int = 16,
                 expand: int = 2, drop_path: float = 0.0):
        super().__init__()
        self.norm1    = nn.LayerNorm(dim)
        self.ss2d     = SS2D(dim, d_state=d_state, expand=expand)
        self.norm2    = nn.LayerNorm(dim)
        self.gate     = nn.Linear(dim, dim, bias=False)
        self.drop     = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : B C H W  â†’  B C H W"""
        B, C, H, W = x.shape
        cl = x.permute(0, 2, 3, 1)                        # B H W C
        y  = self.ss2d(self.norm1(cl)) * F.silu(self.gate(self.norm2(cl)))
        return x + self.drop(y.permute(0, 3, 1, 2))


# Patch Embedding & Patch Merging

class PatchEmbed(nn.Module):
    """4Ã—4 non-overlapping patches â†’ C-dim tokens.  Out: B C H/4 W/4."""
    def __init__(self, in_c: int = 3, embed: int = 96, patch: int = 4):
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed, patch, stride=patch)
        self.norm = nn.LayerNorm(embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                                   # B C H/p W/p
        B, C, H, W = x.shape
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x


class PatchMerging(nn.Module):
    """2Ã—2 merge â†’ halves spatial, doubles channels."""
    def __init__(self, in_c: int):
        super().__init__()
        self.norm   = nn.LayerNorm(4 * in_c)
        self.linear = nn.Linear(4 * in_c, 2 * in_c, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)                         # B H W C
        x = torch.cat([x[:, 0::2, 0::2], x[:, 1::2, 0::2],
                        x[:, 0::2, 1::2], x[:, 1::2, 1::2]], dim=-1)
        x = self.linear(self.norm(x))                      # B H/2 W/2 2C
        return x.permute(0, 3, 1, 2)


# PAU - Progressive Adaptive Upsampling

class PAU(nn.Module):
    """
    Stage 1 â€” CARAFE-style content-aware 2Ã— upsampling.
    Stage 2 â€” Channel attention (GAPâ†’FC) Ã— Spatial attention (DWConvâ†’Sig).
    Output  : X_out = X_up âŠ— Ïƒ(A_c) âŠ— Ïƒ(A_s)
    """
    def __init__(self, in_c: int, out_c: int, k_up: int = 5):
        super().__init__()
        self.out_c = out_c
        self.k_up  = k_up

        # Channel reduction before upsampling
        self.reduce = nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, bias=False),
            LayerNorm2d(out_c),
        )

        # Kernel prediction head (CARAFE-style)
        hidden = max(out_c // 4, 32)
        self.khead = nn.Sequential(
            nn.Conv2d(out_c, hidden, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, k_up * k_up * 4, 1),       # 4 sub-pixel positions
        )

        self.norm = LayerNorm2d(out_c)

        # Channel attention
        r = max(out_c // 8, 4)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(out_c, r),
            nn.ReLU(inplace=True),
            nn.Linear(r, out_c),
            nn.Sigmoid(),
        )

        # Spatial attention
        self.sa = nn.Sequential(
            nn.Conv2d(out_c, 1, 3, padding=1, bias=False),
            nn.Sigmoid(),
        )

    def _upsample(self, x: torch.Tensor) -> torch.Tensor:
        """Content-aware 2Ã— upsampling via CARAFE reassembly."""
        B, C, H, W = x.shape
        k, pad = self.k_up, self.k_up // 2

        # Predict per-location kernels and normalise
        kern = self.khead(x)                               # B kÂ²Â·4 H W
        kern = kern.view(B, k * k, 4, H, W)
        kern = F.softmax(kern, dim=1)                      # B kÂ² 4 H W

        # Unfold input neighbourhood
        x_pad = F.pad(x, [pad, pad, pad, pad])
        x_unf = x_pad.unfold(2, k, 1).unfold(3, k, 1)    # B C H W k k
        x_unf = x_unf.contiguous().view(B, C, H, W, k * k)  # B C H W kÂ²

        # Weighted sum for each of the 4 sub-pixel positions
        # kern: B kÂ² 4 H W â†’ B H W kÂ² 4
        kern  = kern.permute(0, 3, 4, 1, 2)               # B H W kÂ² 4
        x_unf = x_unf.permute(0, 2, 3, 4, 1)             # B H W kÂ² C
        out   = torch.einsum('bhwkc,bhwks->bhwsc', x_unf, kern)  # B H W 4 C

        # Rearrange 4 sub-pixels into 2Ã—2 spatial grid
        out = out.permute(0, 4, 1, 2, 3).view(B, C, H, W, 2, 2)
        out = out.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C, 2*H, 2*W)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x    = self.reduce(x)                              # B out_c H W
        x_up = self._upsample(x)                          # B out_c 2H 2W
        x_up = self.norm(x_up)

        A_c  = self.ca(x_up).view(-1, self.out_c, 1, 1)
        A_s  = self.sa(x_up)
        return x_up * A_c * A_s


# HCAG - Hierarchical Cross-Attention Gate

class HCAG(nn.Module):
    """
    Dual-branch gating for skip connections.

    Channel branch : GAP(X_cat) â†’ FCâ€“GELUâ€“FC â†’ Sigmoid â†’ e_c  (reweights channels)
    Spatial branch : DilConv(d=2) â†’ Conv1Ã—1 â†’ Sigmoid â†’ A_sp  (reweights locations)
    Gate           : G = e_c âŠ— A_sp  applied to X_skip
    Fusion         : LN(Linear([X_up ; X_skipâŠ—G]))
    """
    def __init__(self, c: int, r: int = 8):
        super().__init__()
        cat = 2 * c
        red = max(cat // r, 4)

        # Channel branch
        self.ch_gap = nn.AdaptiveAvgPool2d(1)
        self.ch_fc  = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cat, red), nn.GELU(),
            nn.Linear(red, c),   nn.Sigmoid(),
        )

        # Spatial branch (dilated d=2 for wider context)
        self.sp_dil  = nn.Conv2d(cat, cat, 3, padding=2,
                                  dilation=2, groups=cat, bias=False)
        self.sp_conv = nn.Conv2d(cat, 1, 1, bias=False)
        self.sp_sig  = nn.Sigmoid()

        # Fusion
        self.fuse = nn.Sequential(
            nn.LayerNorm(c),
        )
        self.proj = nn.Linear(cat, c, bias=False)

    def forward(self, x_up: torch.Tensor,
                x_skip: torch.Tensor) -> torch.Tensor:
        cat = torch.cat([x_up, x_skip], dim=1)            # B 2C H W

        # Channel gate
        e_c  = self.ch_fc(self.ch_gap(cat)).view(x_up.size(0), -1, 1, 1)  # B C 1 1

        # Spatial gate
        A_sp = self.sp_sig(self.sp_conv(F.gelu(self.sp_dil(cat))))         # B 1 H W

        # Apply composite gate to skip
        x_skip_g = x_skip * e_c * A_sp                    # B C H W

        # Project fused features
        fused = torch.cat([x_up, x_skip_g], dim=1)        # B 2C H W
        B, _, H, W = fused.shape
        fused = fused.permute(0, 2, 3, 1)                 # B H W 2C
        fused = self.fuse[0](self.proj(fused))             # B H W C
        return fused.permute(0, 3, 1, 2)                  # B C H W


# APFA - Adaptive Pyramid Feature Aggregation

class APFA(nn.Module):
    """
    Collects all decoder outputs, aligns to H/4Ã—W/4, applies three dilated
    DW-Sep Conv branches (d=1,2,4) with learnable softmax-normalised weights.
    """
    def __init__(self, ch_list: list, out_c: int,
                 dilations: tuple = (1, 2, 4)):
        super().__init__()
        n = len(ch_list)

        # Per-stage alignment to out_c channels
        self.align = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, out_c, 1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.SiLU(inplace=True),
            ) for c in ch_list
        ])

        cat_c = out_c * n

        # 3 dilated branches
        self.branches = nn.ModuleList([
            DWSepConv(cat_c, out_c, k=3, d=d) for d in dilations
        ])

        # Learnable branch weights
        self.log_w = nn.Parameter(torch.zeros(len(dilations)))

        # Final conv
        self.final = nn.Sequential(
            nn.Conv2d(out_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True),
        )

    def forward(self, feats: list) -> torch.Tensor:
        """feats: [F_d1(shallowest), F_d2, F_d3, F_d4(deepest)]"""
        H, W = feats[0].shape[2], feats[0].shape[3]

        aligned = []
        for f, align in zip(feats, self.align):
            if f.shape[2] != H or f.shape[3] != W:
                f = F.interpolate(f, (H, W), mode='bilinear', align_corners=False)
            aligned.append(align(f))

        x      = torch.cat(aligned, dim=1)                # B nÂ·out_c H W
        w      = F.softmax(self.log_w, dim=0)
        out    = sum(w[i] * b(x) for i, b in enumerate(self.branches))
        return self.final(out)


# Encoder stage builder

def _stage(dim: int, depth: int, d_state: int,
           expand: int, dpr: list) -> nn.Sequential:
    return nn.Sequential(*[
        VSSBlock(dim, d_state=d_state, expand=expand, drop_path=dpr[i])
        for i in range(depth)
    ])


# AdapMamba-UNet

class AdapMambaUNet(nn.Module):
    """
    AdapMamba-UNet for 2-D medical image segmentation.

    Args
    
    in_chans        Input image channels (3 for RGB, 1 for greyscale).
    num_classes     Segmentation output classes.
    embed_dim       Base channel width C (96 = VMamba-Small).
    depths          VSS block counts per stage [enc1, enc2, enc3, bottleneck].
    d_state         SSM state dimension.
    expand          SS2D expansion factor.
    drop_path_rate  Maximum stochastic-depth rate (linearly scaled).
    k_up            CARAFE kernel size in PAU.
    apfa_dilations  Dilation rates for the three APFA branches.
    """
    def __init__(
        self,
        in_chans:       int   = 3,
        num_classes:    int   = 9,
        embed_dim:      int   = 96,
        depths:         list  = (2, 2, 3, 2),
        d_state:        int   = 16,
        expand:         int   = 2,
        drop_path_rate: float = 0.1,
        k_up:           int   = 5,
        apfa_dilations: tuple = (1, 2, 4),
    ):
        super().__init__()
        C = embed_dim

        # Stochastic-depth schedule
        total = sum(depths)
        dpr   = [x.item() for x in torch.linspace(0, drop_path_rate, total)]
        i = 0
        sdpr = []
        for d in depths:
            sdpr.append(dpr[i:i+d]); i += d

        #  Patch Embedding 
        self.patch_embed = PatchEmbed(in_chans, C)

        # Encoder 
        self.enc1   = _stage(C,     depths[0], d_state, expand, sdpr[0])
        self.merge1 = PatchMerging(C)

        self.enc2   = _stage(2*C,   depths[1], d_state, expand, sdpr[1])
        self.merge2 = PatchMerging(2*C)

        self.enc3   = _stage(4*C,   depths[2], d_state, expand, sdpr[2])
        self.merge3 = PatchMerging(4*C)

        # Bottleneck 
        self.bottle = _stage(8*C,   depths[3], d_state, expand, sdpr[3])

        # Decoder (PAU → HCAG → VSS) 
        self.pau3  = PAU(8*C, 4*C, k_up=k_up)
        self.hcag3 = HCAG(4*C)
        self.dec3  = _stage(4*C, 2, d_state, expand, [0.0]*2)

        self.pau2  = PAU(4*C, 2*C, k_up=k_up)
        self.hcag2 = HCAG(2*C)
        self.dec2  = _stage(2*C, 2, d_state, expand, [0.0]*2)

        self.pau1  = PAU(2*C, C,   k_up=k_up)
        self.hcag1 = HCAG(C)
        self.dec1  = _stage(C,   2, d_state, expand, [0.0]*2)

        # APFA 
        # receives [F_d1(C), F_d2(2C), F_d3(4C), F_d4/bottle(8C)]
        self.apfa = APFA([C, 2*C, 4*C, 8*C], out_c=C,
                          dilations=apfa_dilations)

        # Final PAU → 4Ã— (two Ã—2 steps) + head 
        self.fpau1 = PAU(C,    C//4, k_up=k_up)           # H/4 → H/2
        self.fpau2 = PAU(C//4, C//4, k_up=k_up)           # H/2 → H
        self.head  = nn.Conv2d(C//4, num_classes, 1)

        self._init_weights()

    # Init 

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # Forward 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x   : B Ã— in_chans Ã— H Ã— W   (H,W must be divisible by 32)
        out : B Ã— num_classes Ã— H Ã— W
        """
        # Encoder
        t  = self.patch_embed(x)                           # B  C   H/4  W/4
        e1 = self.enc1(t)                                  # B  C   H/4  W/4
        e2 = self.enc2(self.merge1(e1))                    # B 2C   H/8  W/8
        e3 = self.enc3(self.merge2(e2))                    # B 4C  H/16 W/16
        bn = self.bottle(self.merge3(e3))                  # B 8C  H/32 W/32

        # Decoder
        d3 = self.dec3(self.hcag3(self.pau3(bn), e3))     # B 4C  H/16 W/16
        d2 = self.dec2(self.hcag2(self.pau2(d3), e2))     # B 2C   H/8  W/8
        d1 = self.dec1(self.hcag1(self.pau1(d2), e1))     # B  C   H/4  W/4

        # Aggregate + predict
        agg = self.apfa([d1, d2, d3, bn])                 # B  C   H/4  W/4
        out = self.fpau1(agg)                              # B C/4  H/2  W/2
        out = self.fpau2(out)                              # B C/4  H    W
        return self.head(out)                              # B cls  H    W


# Convenience constructors

def adapmamba_unet_small(num_classes: int = 9, **kw) -> AdapMambaUNet:
    """VMamba-Small capacity.  ~25 M params.  Trained at 224Ã—224."""
    return AdapMambaUNet(embed_dim=96,  depths=(2,2,3,2),
                          num_classes=num_classes, **kw)

def adapmamba_unet_base(num_classes: int = 9, **kw) -> AdapMambaUNet:
    """Wider model for larger GPU budgets.  ~44 M params."""
    return AdapMambaUNet(embed_dim=128, depths=(2,2,6,2),
                          num_classes=num_classes, drop_path_rate=0.2, **kw)


# Quick sanity check  (python model.py)

if __name__ == "__main__":
    model = adapmamba_unet_small(num_classes=9)
    x     = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        y = model(x)
    p = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Input : {tuple(x.shape)}")
    print(f"Output: {tuple(y.shape)}")
    print(f"Params: {p:.1f} M")
    assert y.shape == (2, 9, 224, 224), "shape mismatch"
    print("All checks passed")
