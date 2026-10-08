
import argparse, csv, glob, json, math, os

ROWS = {
    "Stepwise component ablation": ["baseline", "step_pau", "step_pau_hcag", "step_pau_hcag_apfa_fixed", "full"],
    "Factorial ablation": [f"fact_p{p}_h{h}_a{a}" for p in (0, 1) for h in (0, 1) for a in (0, 1)],
    "Upsampling strategy": ["up_bilinear", "up_transposed", "up_patch_expand", "up_carafe", "full"],
    "HCAG gate design": ["gate_none", "gate_channel", "gate_spatial", "gate_both_d1", "full", "gate_both_d4"],
    "APFA dilation rates": ["apfa_d1", "apfa_d12", "full", "apfa_d136", "apfa_d1248"],
}


def load(results):
    out = {}
    for path in glob.glob(os.path.join(results, "*", "*", "summary.json")):
        dataset, cfg = path.split(os.sep)[-3], path.split(os.sep)[-2]
        s = json.load(open(path))
        if "per_seed" in s:
            out[(dataset, cfg)] = s
    return out


def fmt(stat, key, scale=1.0, nd=2):
    m = stat["across_seeds"][key]
    return f"{scale * m['mean']:.{nd}f} ± {scale * m['std']:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--cost", default=None)
    ap.add_argument("--out", default="results/tables")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res = load(a.results)
    if not res:
        raise SystemExit(f"no summary.json files under {a.results}/<dataset>/<config>/")
    cost = {}
    if a.cost and os.path.exists(a.cost):
        cost = {r["config"]: r for r in csv.DictReader(open(a.cost)) if r["resolution"] == "224"}

    # 1. per-seed export
    keys = sorted({k for s in res.values() for seed in s["per_seed"].values() for k in seed})
    with open(os.path.join(a.out, "seed_results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "config", "seed"] + keys)
        for (ds, cfg), s in sorted(res.items()):
            for seed, vals in s["per_seed"].items():
                w.writerow([ds, cfg, seed] + [vals.get(k, "") for k in keys])

    # 2. tables
    md, tex = [], []
    for ds in sorted({d for d, _ in res}):
        for title, cfgs in ([("Main result", ["full"])] + list(ROWS.items())):
            present = [c for c in cfgs if (ds, c) in res]
            if not present:
                continue
            md += [f"## {ds}: {title}", "", "| Config | DSC (%) | HD95 (mm) | HD95 (TransUNet rule) | Params (M) | GFLOPs |",
                   "|---|---|---|---|---|---|"]
            for c in present:
                s = res[(ds, c)]
                p = cost.get(c, {})
                row = [c, fmt(s, "dice_mean", 100), fmt(s, "hd95mm_mean"), fmt(s, "hd95tu_mean"),
                       f"{float(p['params_M']):.2f}" if p else "-", f"{float(p['total_GFLOPs']):.2f}" if p else "-"]
                md.append("| " + " | ".join(row) + " |")
                tex.append(" & ".join(r.replace("±", "$\\pm$") for r in row) + r" \\")
            md.append("")
    # 3. factorial effect decomposition (main effects and interactions, DSC points)
    for ds in sorted({d for d, _ in res}):
        cells = {(p, h, f): res.get((ds, f"fact_p{p}_h{h}_a{f}")) for p in (0, 1) for h in (0, 1) for f in (0, 1)}
        if all(cells.values()):
            y = {k: 100 * v["across_seeds"]["dice_mean"]["mean"] for k, v in cells.items()}
            def effect(idx):
                return sum(v * math.prod(1 if k[i] else -1 for i in idx) for k, v in y.items()) / 4
            names = {"PAU": [0], "HCAG": [1], "APFA": [2], "PAU x HCAG": [0, 1], "PAU x APFA": [0, 2],
                     "HCAG x APFA": [1, 2], "PAU x HCAG x APFA": [0, 1, 2]}
            md += [f"## {ds}: factorial effect decomposition (DSC points)", "", "| Effect | Estimate |", "|---|---|"]
            md += [f"| {n} | {effect(i):+.3f} |" for n, i in names.items()]
            md += [f"| joint gain (all vs none) | {y[(1, 1, 1)] - y[(0, 0, 0)]:+.3f} |", ""]
    open(os.path.join(a.out, "tables.md"), "w").write("\n".join(md) + "\n")
    open(os.path.join(a.out, "tables.tex"), "w").write("\n".join(tex) + "\n")
    print(f"wrote {a.out}/seed_results.csv, tables.md, tables.tex ({len(res)} configuration results)")


if __name__ == "__main__":
    main()
