"""
Statistical comparison of two methods from per-case results (Section 4.2, SI S5).

Inputs are the ensemble_cases.csv files written by test.py (seed-averaged softmax
predictions of each method, same three seeds). Cases are paired by id; the
statistical unit is the case (12 Synapse volumes, 20 ACDC patients).

Reported for the mean DSC over classes:
    Shapiro-Wilk test on the paired differences
    paired two-tailed t-test (t, df, p)
    Wilcoxon signed-rank test, two-sided (exact for small n)
    Cohen's d for paired samples (mean difference / SD of differences)
    95% t-confidence interval of the mean DSC of the first method
    
Per class (Synapse organs or ACDC structures): paired t-test and Wilcoxon p-values,
with Holm-Bonferroni correction across classes.

Example
    python stats.py --a results/synapse/full/ensemble_cases.csv \
                    --b results/synapse/vmkla/external_cases.csv \
                    --name_a AdapMamba-UNet --name_b VMKLA-UNet --out results/synapse/stats
"""

import argparse
import csv
import json
import os

import numpy as np
from scipy import stats


def read_cases(path):
    with open(path) as f:
        return {r["case"]: {k: float(v) for k, v in r.items() if k != "case"} for r in csv.DictReader(f)}


def holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values (monotone, capped at 1)."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj


def paired_tests(x, y):
    d = x - y
    t = stats.ttest_rel(x, y)
    w = stats.wilcoxon(x, y, alternative="two-sided") if np.any(d != 0) else None
    return {"mean_a": float(x.mean()), "mean_b": float(y.mean()), "mean_diff": float(d.mean()),
            "t": float(t.statistic), "df": int(len(d) - 1), "p_t": float(t.pvalue),
            "p_wilcoxon": float(w.pvalue) if w is not None else 1.0,
            "cohens_d": float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="per-case CSV of the proposed method")
    ap.add_argument("--b", required=True, help="per-case CSV of the comparison method")
    ap.add_argument("--name_a", default="A")
    ap.add_argument("--name_b", default="B")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    A, B = read_cases(a.a), read_cases(a.b)
    cases = sorted(set(A) & set(B))
    if len(cases) != len(A) or len(cases) != len(B):
        raise SystemExit(f"case mismatch: {len(A)} vs {len(B)} cases, {len(cases)} shared")

    x = np.array([100 * A[c]["dice_mean"] for c in cases])
    y = np.array([100 * B[c]["dice_mean"] for c in cases])
    res = {"n_cases": len(cases), "cases": cases, "method_a": a.name_a, "method_b": a.name_b}
    sw = stats.shapiro(x - y)
    res["shapiro_wilk"] = {"W": float(sw.statistic), "p": float(sw.pvalue)}
    res["overall"] = paired_tests(x, y)
    ci = stats.t.interval(0.95, len(x) - 1, loc=x.mean(), scale=stats.sem(x))
    res["ci95_mean_dsc_a"] = [float(ci[0]), float(ci[1])]

    classes = [k[len("dice_"):] for k in A[cases[0]] if k.startswith("dice_") and k != "dice_mean"]
    per = []
    for c in classes:
        xc = np.array([100 * A[k][f"dice_{c}"] for k in cases])
        yc = np.array([100 * B[k][f"dice_{c}"] for k in cases])
        per.append({"class": c, **paired_tests(xc, yc)})
    for key in ["p_t", "p_wilcoxon"]:
        adj = holm([r[key] for r in per])
        for r, v in zip(per, adj):
            r[f"{key}_holm"] = float(v)
    res["per_class"] = per

    with open(os.path.join(a.out, "stats.json"), "w") as f:
        json.dump(res, f, indent=2)

    o = res["overall"]
    lines = [f"# {a.name_a} vs {a.name_b} (n = {len(cases)} cases)", "",
             f"- Mean DSC: {o['mean_a']:.2f} vs {o['mean_b']:.2f} (difference {o['mean_diff']:+.2f})",
             f"- Shapiro-Wilk on differences: W = {sw.statistic:.3f}, p = {sw.pvalue:.3f}",
             f"- Paired t-test: t = {o['t']:.2f}, df = {o['df']}, p = {o['p_t']:.4f}",
             f"- Wilcoxon signed-rank: p = {o['p_wilcoxon']:.4f}",
             f"- Cohen's d (paired): {o['cohens_d']:.2f}",
             f"- 95% CI of mean DSC ({a.name_a}): [{ci[0]:.2f}, {ci[1]:.2f}]", "",
             "| Class | Delta DSC | p (t) | p (t, Holm) | p (Wilcoxon) | p (Wilcoxon, Holm) |",
             "|---|---|---|---|---|---|"]
    for r in per:
        lines.append(f"| {r['class']} | {r['mean_diff']:+.2f} | {r['p_t']:.4f} | {r['p_t_holm']:.4f} | "
                     f"{r['p_wilcoxon']:.4f} | {r['p_wilcoxon_holm']:.4f} |")
    with open(os.path.join(a.out, "stats.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
