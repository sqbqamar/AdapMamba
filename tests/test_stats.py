"""stats.py test on synthetic per-case files (inputs are random test data, not results)."""
import sys, pathlib, csv, json, subprocess, tempfile, os
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stats import holm

def write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def main():
    # Holm against a direct reference computation
    p = [0.0018, 0.0041, 0.012, 0.021, 0.029, 0.038, 0.049, 0.62]
    ref = []
    order = sorted(range(8), key=lambda i: p[i]); run = 0; adj = [0]*8
    for r, i in enumerate(order):
        run = max(run, (8 - r) * p[i]); adj[i] = min(run, 1)
    assert np.allclose(holm(p), adj)
    try:
        from statsmodels.stats.multitest import multipletests
        assert np.allclose(holm(p), multipletests(p, method="holm")[1]); print("Holm matches statsmodels")
    except ImportError:
        print("Holm matches the direct step-down computation")
    rng = np.random.default_rng(1)
    organs = ["aorta", "gallbladder", "spleen", "kidney_l", "kidney_r", "liver", "stomach", "pancreas"]
    A, B = [], []
    for i in range(12):
        base = rng.uniform(0.6, 0.9, 8)
        a = np.clip(base + rng.normal(0.02, 0.02, 8), 0, 1)
        for rows, v in ((A, a), (B, base)):
            r = {"case": f"case{i:04d}", **{f"dice_{o}": x for o, x in zip(organs, v)}, "dice_mean": v.mean()}
            rows.append(r)
    d = tempfile.mkdtemp()
    write(os.path.join(d, "a.csv"), A); write(os.path.join(d, "b.csv"), B)
    subprocess.run([sys.executable, str(ROOT / "stats.py"), "--a", f"{d}/a.csv", "--b", f"{d}/b.csv",
                    "--out", f"{d}/out"], check=True, capture_output=True)
    res = json.load(open(f"{d}/out/stats.json"))
    assert res["n_cases"] == 12 and res["overall"]["df"] == 11 and len(res["per_class"]) == 8
    x = np.array([r["dice_mean"] for r in A]) * 100; y = np.array([r["dice_mean"] for r in B]) * 100
    from scipy import stats as st
    assert abs(res["overall"]["p_t"] - st.ttest_rel(x, y).pvalue) < 1e-12
    dd = x - y
    assert abs(res["overall"]["cohens_d"] - dd.mean() / dd.std(ddof=1)) < 1e-12
    print("paired t-test, Cohen's d, df and per-class output verified")
    print(open(f"{d}/out/stats.md").read().splitlines()[2])
    print("PASS")

if __name__ == "__main__":
    main()
