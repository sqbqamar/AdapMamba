

FULL = dict(upsampler="pau", skip_fusion="hcag", hcag_dilation=2,
            use_apfa=True, apfa_dilations=(1, 2, 4), apfa_learnable=True)


def _cfg(**overrides):
    c = dict(FULL)
    c.update(overrides)
    return c


BASELINE = dict(upsampler="patch_expand", skip_fusion="concat_linear", hcag_dilation=2,
                use_apfa=False, apfa_dilations=(1, 2, 4), apfa_learnable=True)

CONFIGS = {
    "full": FULL,
    "baseline": BASELINE,

    # Stepwise component ablation (Table: component-wise ablation)
    "step_pau": dict(BASELINE, upsampler="pau"),
    "step_pau_hcag": dict(BASELINE, upsampler="pau", skip_fusion="hcag"),
    "step_pau_hcag_apfa_fixed": dict(BASELINE, upsampler="pau", skip_fusion="hcag",
                                     use_apfa=True, apfa_learnable=False),
    # the final row of that table is "full"

    # 2x2x2 factorial over PAU, HCAG, APFA (Table: factorial ablation).
    # "Absent" means the baseline counterpart: patch expanding, concat+linear, single scale.
    **{f"fact_p{p}_h{h}_a{a}": dict(BASELINE,
                                     upsampler="pau" if p else "patch_expand",
                                     skip_fusion="hcag" if h else "concat_linear",
                                     use_apfa=bool(a))
       for p in (0, 1) for h in (0, 1) for a in (0, 1)},

    # Upsampling strategy (Table: upsampling comparison); HCAG and APFA fixed
    "up_bilinear": _cfg(upsampler="bilinear"),
    "up_transposed": _cfg(upsampler="transposed"),
    "up_patch_expand": _cfg(upsampler="patch_expand"),
    "up_carafe": _cfg(upsampler="carafe"),
    "up_pau": FULL,

    # Gate design (Table: HCAG gate design); PAU and APFA fixed
    "gate_none": _cfg(skip_fusion="concat_linear"),
    "gate_channel": _cfg(skip_fusion="channel"),
    "gate_spatial": _cfg(skip_fusion="spatial"),
    "gate_both_d1": _cfg(hcag_dilation=1),
    "gate_both_d2": FULL,
    "gate_both_d4": _cfg(hcag_dilation=4),

    # APFA dilation configuration (Table: APFA dilation rates)
    "apfa_d1": _cfg(apfa_dilations=(1,)),
    "apfa_d12": _cfg(apfa_dilations=(1, 2)),
    "apfa_d124": FULL,
    "apfa_d136": _cfg(apfa_dilations=(1, 3, 6)),
    "apfa_d1248": _cfg(apfa_dilations=(1, 2, 4, 8)),
}


def get_config(name):
    if name not in CONFIGS:
        raise KeyError(f"unknown config '{name}'. Available: {sorted(CONFIGS)}")
    return dict(CONFIGS[name])
