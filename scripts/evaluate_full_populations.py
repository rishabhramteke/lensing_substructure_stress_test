"""Full 1 000-lens populations for Families A and B (referee round 3, point 4): completeness per
0.5-dex bin with 68% Wilson intervals, confounder FPR with Wilson intervals, paired flip, and a
LaTeX table for the paper. Also reports the mean/std over the three 300-lens subsamples for comparison.

    python scripts/evaluate_full_populations.py -> results/full_populations.json, paper/tables/full_populations.tex
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BINS = [(8.0, 8.5), (8.5, 9.0), (9.0, 9.5), (9.5, 10.0), (10.0, 10.5), (10.5, 11.0)]


def wilson(k, n, z=1.0):
    if n == 0: return (np.nan, np.nan, np.nan)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def load(root, pop):
    p = ROOT / root / pop / "scan_results.jsonl"
    return [json.loads(l) for l in open(p)] if (ROOT / root / pop / "manifest.json").exists() else None


def evaluate(root):
    ns, t60, t15, a1, a3 = (load(root, p) for p in ("no_subhalo", "test_fixed60", "test_fixed15", "multipole_m4_a1", "multipole_m4_a3"))
    if not (ns and t60 and t15):
        return None
    rel_ns = [r for r in ns if r["reliable_fit"]]; thr = float(np.quantile([r["delta_chi2"] for r in rel_ns], 0.90))
    out = {"threshold": thr, "n_clean_reliable": len(rel_ns), "n_unreliable": {}}
    for name, recs in (("no_subhalo", ns), ("test_fixed60", t60), ("test_fixed15", t15), ("multipole_m4_a1", a1), ("multipole_m4_a3", a3)):
        if recs: out["n_unreliable"][name] = sum(1 for r in recs if not r["reliable_fit"])
    for key, recs in (("c60", t60), ("c15", t15)):
        rel = [r for r in recs if r["reliable_fit"] and r["has_subhalo"]]; d = {}
        for lo, hi in BINS:
            sel = [r for r in rel if lo <= r["log10_M200_true"] < hi]; k = sum(1 for r in sel if r["delta_chi2"] >= thr)
            p_, l_, h_ = wilson(k, len(sel)); d[f"{lo}-{hi}"] = {"n": len(sel), "k": k, "completeness": p_, "lo": l_, "hi": h_}
        out[f"completeness_{key}"] = d
    for name, recs in (("multipole_m4_a1", a1), ("multipole_m4_a3", a3)):
        if recs:
            rel = [r for r in recs if r["reliable_fit"]]; k = sum(1 for r in rel if r["delta_chi2"] >= thr)
            p_, l_, h_ = wilson(k, len(rel)); out[f"fpr_{name}"] = {"n": len(rel), "fpr": p_, "lo": l_, "hi": h_}
    r60 = {r["index"]: r for r in t60 if r["has_subhalo"] and r["reliable_fit"]}; r15 = {r["index"]: r for r in t15 if r["reliable_fit"]}
    idx = [i for i in r60 if i in r15]; d60 = np.array([r60[i]["delta_chi2"] >= thr for i in idx]); d15 = np.array([r15[i]["delta_chi2"] >= thr for i in idx])
    out["paired_flip"] = {"n_pairs": len(idx), "c60_not_c15": int((d60 & ~d15).sum()), "c15_not_c60": int((d15 & ~d60).sum())}
    return out


def fmt_pct(d):
    return f"{100*d['completeness']:.0f}$^{{+{100*(d['hi']-d['completeness']):.0f}}}_{{-{100*(d['completeness']-d['lo']):.0f}}}$"


def main():
    res = {"A": evaluate("results/baseline_a_full"), "B": evaluate("results/baseline_b_full/fitted")}
    (ROOT / "results/full_populations.json").write_text(json.dumps(res, indent=2))
    (ROOT / "paper/tables").mkdir(exist_ok=True)
    rows = []
    for fam, label in (("A", "A scan"), ("B", "B pot.\\ corr.")):
        d = res[fam]
        if d is None: continue
        for key, cl in (("c60", "$c{=}60$"), ("c15", "$c{=}15$")):
            rows.append(f"{label}, {cl} & " + " & ".join(fmt_pct(d[f'completeness_{key}'][f'{lo}-{hi}']) for lo, hi in BINS) + " \\\\")
    tex = ["\\begin{tabular}{@{}l" + "c" * 6 + "@{}}", "\\toprule", "family, population & " + " & ".join(f"{lo}--{hi}" for lo, hi in BINS) + " \\\\", "\\midrule"] + rows + ["\\bottomrule", "\\end{tabular}"]
    (ROOT / "paper/tables/full_populations.tex").write_text("\n".join(tex) + "\n")
    for fam, d in res.items():
        if d is None: print(fam, "not complete"); continue
        print(f"== {fam}: thr {d['threshold']:.2f}; unreliable {d['n_unreliable']}")
        for key in ("c60", "c15"):
            print("  ", key, {b: f"{100*v['completeness']:.0f}% (n={v['n']}, ±{50*(v['hi']-v['lo']):.0f})" for b, v in d[f"completeness_{key}"].items()})
        for name in ("fpr_multipole_m4_a1", "fpr_multipole_m4_a3"):
            if name in d: print("  ", name, f"{100*d[name]['fpr']:.1f}% [{100*d[name]['lo']:.1f}, {100*d[name]['hi']:.1f}] n={d[name]['n']}")
        print("   flip", d["paired_flip"])


if __name__ == "__main__":
    main()
