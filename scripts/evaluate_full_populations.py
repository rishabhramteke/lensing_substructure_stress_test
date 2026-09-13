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


def _completeness(recs, thr, strict, conservative):
    """Per-bin completeness. `conservative`: macro-fit failures stay in the denominator and
    count as misses (referee round 6, point 7) instead of being excluded."""
    pool = [r for r in recs if r["has_subhalo"] and (conservative or r["reliable_fit"])]
    d = {}
    for lo, hi in BINS:
        sel = [r for r in pool if lo <= r["log10_M200_true"] < hi]
        k = sum(1 for r in sel if r["reliable_fit"] and (r["delta_chi2"] > thr if strict else r["delta_chi2"] >= thr))
        p_, l_, h_ = wilson(k, len(sel))
        d[f"{lo}-{hi}"] = {"n": len(sel), "k": k, "completeness": p_, "lo": l_, "hi": h_}
    return d


def evaluate(root, floor=False):
    """`floor=True` scores the null-floored statistic max(Delta chi^2, 0) -- i.e. the grid with a
    zero-mass hypothesis added, whose clean distribution is degenerate at 0, so the operating point
    is Delta chi^2 > 0 rather than a 10%-FPR quantile (referee round 6, point 1)."""
    ns, t60, t15, a1, a3 = (load(root, p) for p in ("no_subhalo", "test_fixed60", "test_fixed15", "multipole_m4_a1", "multipole_m4_a3"))
    if not (ns and t60 and t15):
        return None
    rel_ns = [r for r in ns if r["reliable_fit"]]
    thr = 0.0 if floor else float(np.quantile([r["delta_chi2"] for r in rel_ns], 0.90))
    hit = (lambda r: r["delta_chi2"] > thr) if floor else (lambda r: r["delta_chi2"] >= thr)
    out = {"threshold": thr, "floored": floor, "n_clean_reliable": len(rel_ns),
           "fpr_clean": float(np.mean([hit(r) for r in rel_ns])), "n_unreliable": {}, "n_total": {}}
    for name, recs in (("no_subhalo", ns), ("test_fixed60", t60), ("test_fixed15", t15), ("multipole_m4_a1", a1), ("multipole_m4_a3", a3)):
        if recs:
            out["n_unreliable"][name] = sum(1 for r in recs if not r["reliable_fit"]); out["n_total"][name] = len(recs)
            sub = [r for r in recs if r["has_subhalo"]]
            if sub:
                out["n_unreliable"][name + "_subhalo_only"] = sum(1 for r in sub if not r["reliable_fit"])
                out["n_total"][name + "_subhalo_only"] = len(sub)
    for key, recs in (("c60", t60), ("c15", t15)):
        out[f"completeness_{key}"] = _completeness(recs, thr, floor, conservative=False)
        out[f"conservative_{key}"] = _completeness(recs, thr, floor, conservative=True)
    for name, recs in (("multipole_m4_a1", a1), ("multipole_m4_a3", a3)):
        if recs:
            rel = [r for r in recs if r["reliable_fit"]]; k = sum(1 for r in rel if hit(r))
            p_, l_, h_ = wilson(k, len(rel)); out[f"fpr_{name}"] = {"n": len(rel), "fpr": p_, "lo": l_, "hi": h_}
    r60 = {r["index"]: r for r in t60 if r["has_subhalo"] and r["reliable_fit"]}; r15 = {r["index"]: r for r in t15 if r["reliable_fit"]}
    idx = [i for i in r60 if i in r15]; d60 = np.array([hit(r60[i]) for i in idx]); d15 = np.array([hit(r15[i]) for i in idx])
    out["paired_flip"] = {"n_pairs": len(idx), "c60_not_c15": int((d60 & ~d15).sum()), "c15_not_c60": int((d15 & ~d60).sum())}
    return out


def fmt_pct(d):
    return f"{100*d['completeness']:.0f}$^{{+{100*(d['hi']-d['completeness']):.0f}}}_{{-{100*(d['completeness']-d['lo']):.0f}}}$"


def main():
    res = {"A": evaluate("results/baseline_a_full"), "B": evaluate("results/baseline_b_full/fitted"),
           "A_floored": evaluate("results/baseline_a_full", floor=True)}
    (ROOT / "results/full_populations.json").write_text(json.dumps(res, indent=2))
    (ROOT / "paper/tables").mkdir(exist_ok=True)
    rows = []
    blocks = [("A", "A scan", [("completeness", "reported"), ("conservative", "conservative"), (None, None)]),
              ("A_floored", "A scan, $\\dchi>0$", [("completeness", "reported"), ("conservative", "conservative"), (None, None)]),
              ("B", "B linear $\\delta\\psi$", [("completeness", "reported"), ("conservative", "conservative")])]
    for fam, label, crits in blocks:
        d = res.get(fam)
        if d is None:
            continue
        first = True
        for key, cl in (("c60", "$c{=}60$"), ("c15", "$c{=}15$")):
            for pref, cname in crits:
                if pref is None:
                    rows.append("\\addlinespace[2pt]"); continue
                rows.append(f"{label if first else ''} & {cl if cname == 'reported' else ''} & {cname} & "
                            + " & ".join(fmt_pct(d[f'{pref}_{key}'][f'{lo}-{hi}']) for lo, hi in BINS) + " \\\\")
                first = False
    tex = ["\\begin{tabular}{@{}lll" + "c" * 6 + "@{}}", "\\toprule",
           "family & population & fits & " + " & ".join(f"{lo}--{hi}" for lo, hi in BINS) + " \\\\", "\\midrule"] + rows + ["\\bottomrule", "\\end{tabular}"]
    (ROOT / "paper/tables/full_populations.tex").write_text("\n".join(tex) + "\n")
    for fam, d in res.items():
        if d is None: print(fam, "not complete"); continue
        print(f"== {fam}: thr {d['threshold']:.2f}; clean FPR {100*d['fpr_clean']:.2f}%")
        print("   excluded:", {k: f"{v}/{d['n_total'][k]} ({100*v/d['n_total'][k]:.0f}%)" for k, v in d["n_unreliable"].items()})
        for key in ("c60", "c15"):
            print("  ", key, "reported ", " ".join(f"{100*v['completeness']:.0f}" for v in d[f"completeness_{key}"].values()))
            print("  ", key, "conserv. ", " ".join(f"{100*v['completeness']:.0f}" for v in d[f"conservative_{key}"].values()))
        for name in ("fpr_multipole_m4_a1", "fpr_multipole_m4_a3"):
            if name in d: print("  ", name, f"{100*d[name]['fpr']:.1f}% [{100*d[name]['lo']:.1f}, {100*d[name]['hi']:.1f}] n={d[name]['n']}")
        print("   flip", d["paired_flip"])


if __name__ == "__main__":
    main()
