"""
AdapMamba-UNet (Section 3.1).

Encoder: VMamba-Small (embedding width C = 96) with stage depths [2, 2, 3, 2],
initialised from the ImageNet-1K VMamba-Small checkpoint. The published
VMamba-Small has depths [2, 2, 27, 2]; with [2, 2, 3, 2] every encoder tensor has
a counterpart in the checkpoint, and the first three blocks of stage 3 are used
(VM-UNet applies the same truncation, with nine blocks, to this checkpoint).

Decoder: F_d,4 = VSS(F_e,4). For i = 3, 2, 1:
    X_up = Up(F_d,i+1)             (PAU by default; halves channels, doubles resolution)
    X_fused = HCAG(X_up, F_e,i)
    F_d,i = VSS(X_fused)
APFA aggregates {F_d,1..F_d,4} at H/4; two stacked 2x PAU blocks restore full
resolution; a 1x1 convolution produces the logits.

Every proposed component can be switched off or replaced, which is how the
ablation tables are produced (see adapmamba/configs.py).
"""

import torch
import torch.nn as nn

from .ss2d import VSSBlock
from .modules import make_upsampler, HCAG, APFA


# ---------------------------------------------------------------------------
# Encoder (parameter names identical to VMamba v0)
# ---------------------------------------------------------------------------

class PatchEmbed2D(nn.Module):
    def __init__(self, in_chans=3, embed_dim=96, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):                      # (B, 3, H, W) -> (B, H/4, W/4, C)
        return self.norm(self.proj(x).permute(0, 2, 3, 1))


class PatchMerging2D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x):                      # (B, H, W, C) -> (B, H/2, W/2, 2C)
        x = torch.cat([x[:, 0::2, 0::2], x[:, 1::2, 0::2], x[:, 0::2, 1::2], x[:, 1::2, 1::2]], dim=-1)
        return self.reduction(self.norm(x))


class VSSLayer(nn.Module):
    def __init__(self, dim, depth, drop_paths, d_state, scan_backend, downsample):
        super().__init__()
        self.blocks = nn.ModuleList([
            VSSBlock(dim, drop_path=drop_paths[i], d_state=d_state, scan_backend=scan_backend)
            for i in range(depth)])
        self.downsample = PatchMerging2D(dim) if downsample else None


class VSSEncoder(nn.Module):
    def __init__(self, in_chans=3, embed_dim=96, depths=(2, 2, 3, 2), d_state=16,
                 drop_path_rate=0.0, scan_backend="auto"):
        super().__init__()
        self.patch_embed = PatchEmbed2D(in_chans, embed_dim)
        dpr = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        self.layers = nn.ModuleList()
        start = 0
        for i, depth in enumerate(depths):
            self.layers.append(VSSLayer(embed_dim * 2 ** i, depth, dpr[start:start + depth],
                                        d_state, scan_backend, downsample=i < len(depths) - 1))
            start += depth

    def forward(self, x):
        """Returns [F_e,1 .. F_e,4] in (B, C, H, W) layout."""
        x = self.patch_embed(x)
        feats = []
        for layer in self.layers:
            for blk in layer.blocks:
                x = blk(x)
            feats.append(x.permute(0, 3, 1, 2).contiguous())
            if layer.downsample is not None:
                x = layer.downsample(x)
        return feats


def load_vmamba_pretrained(encoder, path):
    """Initialise the encoder from a VMamba checkpoint by parameter name.

    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    src = ckpt.get("model", ckpt)
    own = encoder.state_dict()
    loaded, shape_mismatch = [], []
    for k, v in src.items():
        if k in own:
            if own[k].shape == v.shape:
                own[k] = v
                loaded.append(k)
            else:
                shape_mismatch.append((k, tuple(v.shape), tuple(own[k].shape)))
    missing = [k for k in own if k not in loaded]
    if shape_mismatch or missing:
        raise RuntimeError(f"pretrained load incomplete: {len(missing)} missing, "
                           f"shape mismatches {shape_mismatch[:5]}")
    encoder.load_state_dict(own, strict=True)
    unused = sorted({".".join(k.split(".")[:3]) for k in src if k not in own})
    return {"checkpoint": str(path), "encoder_tensors": len(own), "loaded": len(loaded),
            "missing": 0, "unused_checkpoint_groups": unused}


# ---------------------------------------------------------------------------
# AdapMamba-UNet
# ---------------------------------------------------------------------------

class AdapMambaUNet(nn.Module):
    def __init__(self, num_classes, in_chans=3, embed_dim=96, depths=(2, 2, 3, 2), d_state=16,
                 upsampler="pau", skip_fusion="hcag", hcag_dilation=2,
                 use_apfa=True, apfa_dilations=(1, 2, 4), apfa_learnable=True,
                 k_up=5, drop_path_rate=0.0, scan_backend="auto"):
        super().__init__()
        C = embed_dim
        fusion_mode = {"hcag": "both", "channel": "channel", "spatial": "spatial",
                       "concat_linear": "none"}[skip_fusion]
        self.config = dict(num_classes=num_classes, in_chans=in_chans, embed_dim=C, depths=list(depths),
                           d_state=d_state, upsampler=upsampler, skip_fusion=skip_fusion,
                           hcag_dilation=hcag_dilation, use_apfa=use_apfa,
                           apfa_dilations=list(apfa_dilations), apfa_learnable=apfa_learnable,
                           k_up=k_up, drop_path_rate=drop_path_rate)

        self.encoder = VSSEncoder(in_chans, C, depths, d_state, drop_path_rate, scan_backend)
        self.dec4 = VSSBlock(8 * C, d_state=d_state, scan_backend=scan_backend)
        self.up3 = make_upsampler(upsampler, 8 * C, 4 * C, k_up)
        self.fuse3 = HCAG(4 * C, dilation=hcag_dilation, mode=fusion_mode)
        self.dec3 = VSSBlock(4 * C, d_state=d_state, scan_backend=scan_backend)
        self.up2 = make_upsampler(upsampler, 4 * C, 2 * C, k_up)
        self.fuse2 = HCAG(2 * C, dilation=hcag_dilation, mode=fusion_mode)
        self.dec2 = VSSBlock(2 * C, d_state=d_state, scan_backend=scan_backend)
        self.up1 = make_upsampler(upsampler, 2 * C, C, k_up)
        self.fuse1 = HCAG(C, dilation=hcag_dilation, mode=fusion_mode)
        self.dec1 = VSSBlock(C, d_state=d_state, scan_backend=scan_backend)
        self.apfa = APFA([C, 2 * C, 4 * C, 8 * C], C, apfa_dilations, apfa_learnable) if use_apfa else None
        self.final_up1 = make_upsampler(upsampler, C, C, k_up)   # H/4 -> H/2
        self.final_up2 = make_upsampler(upsampler, C, C, k_up)   # H/2 -> H
        self.head = nn.Conv2d(C, num_classes, 1)
        self._init_decoder_weights()

    def _init_decoder_weights(self):
        """Kaiming-normal initialisation for every convolution and linear layer outside
        the encoder. State-space parameters (A, D, step-size and input projections)
        keep the standard Mamba initialisation set in SS2D."""
        for name, m in self.named_modules():
            if name.startswith("encoder"):
                continue
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def load_pretrained_encoder(self, path):
        return load_vmamba_pretrained(self.encoder, path)

    @staticmethod
    def _vss(block, x):  # (B, C, H, W) -> VSS -> (B, C, H, W)
        return block(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()

    def forward(self, x):
        e1, e2, e3, e4 = self.encoder(x)
        d4 = self._vss(self.dec4, e4)
        d3 = self._vss(self.dec3, self.fuse3(self.up3(d4), e3))
        d2 = self._vss(self.dec2, self.fuse2(self.up2(d3), e2))
        d1 = self._vss(self.dec1, self.fuse1(self.up1(d2), e1))
        y = self.apfa([d1, d2, d3, d4]) if self.apfa is not None else d1
        return self.head(self.final_up2(self.final_up1(y)))


def param_groups(model, weight_decay):
    """AdamW groups: A_logs and Ds are excluded from weight decay, as in VMamba."""
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (no_decay if getattr(p, "_no_weight_decay", False) else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]
