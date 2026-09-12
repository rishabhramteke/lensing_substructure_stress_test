"""Cost and reach of multipole freedom in Family A's macro-model (referee round 4).

Three macro-models on the same 300-lens subsamples (seed-0 selections): the bare EPL+shear used
throughout the paper, EPL + a free m=4 term (the post-Lange+2024 fix), and EPL + free m=3 and m=4
terms (the cost of freedom the truth does not need). Scored on the clean population (threshold at
10% FPR from each model's own clean scan), the two m=4 confounder populations, an m=3 confounder
population (the form the m=4 fix was NOT built for) and the c=60 subhalo population.

    python scripts/evaluate_multipole_orders.py
-> results/multipole_orders_summary.json, paper/tables/mpmacro.tex
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOC = 0.16
COLS = [("bare", "bare EPL", {"default": "results/baseline_a", "multipole_m3_a3": "results/baseline_a_m3truth"}),
        ("mp4", "$+\\,m{=}4$", {"default": "results/baseline_a_mpmacro"}),
        ("mp34", "$+\\,m{=}3{+}4$", {"default": "results/baseline_a_mp34"})]
POPS = ("no_subhalo", "multipole_m4_a1", "multipole_m4_a3", "multipole_m3_a3", "test_fixed60")
truth60 = [json.loads(l) for l in open(ROOT / "data/test_fixed60/truth.jsonl")]


def load(root, pop):
    p = ROOT / root / pop / "scan_results.jsonl"
    if not p.exists() or not (ROOT / root / pop / "manifest.json").exists():   # unfinished runs are still buffered
        return None
    return [json.loads(l) for l in open(p)]


def fmt(x, pct=True, nd=0):
    return "--" if x is None else (f"{100*x:.{nd}f}\\%" if pct else f"{x:.{nd}f}")


def pair(a, b):
    """'a / b%' with one decimal only below 10% (keeps the table inside the column)."""
    f = lambda v: "--" if v is None else (f"{100*v:.1f}" if 100 * v < 10 else f"{100*v:.0f}")
    return f"{f(a)} / {f(b)}\\%"


def main():
    out = {}
    for key, label, roots in COLS:
        R = {pop: load(roots.get(pop, roots["default"]), pop) for pop in POPS}
        d = {"label": label, "roots": roots, "n": {pop: (len(r) if r else 0) for pop, r in R.items()}}
        rel = {pop: ([x for x in r if x["reliable_fit"]] if r else None) for pop, r in R.items()}
        d["rejected"] = {pop: (len(r) - len(rel[pop]) if r else None) for pop, r in R.items()}
        thr = float(np.quantile([x["delta_chi2"] for x in rel["no_subhalo"]], 0.90)) if rel["no_subhalo"] else None
        d["threshold_10pct_fpr"] = thr
        for pop in ("multipole_m4_a1", "multipole_m4_a3", "multipole_m3_a3"):
            rr = rel[pop]
            d[pop] = None if (rr is None or thr is None) else {
                "chi2_dof_median": float(np.median([x["chi2_smooth_per_dof"] for x in R[pop]])),
                "fpr_10pct": float(np.mean([x["delta_chi2"] >= thr for x in rr])), "fpr_20": float(np.mean([x["delta_chi2"] >= 20 for x in rr])),
                "fpr_100": float(np.mean([x["delta_chi2"] >= 100 for x in rr])), "n_reliable": len(rr)}
        rr = rel["test_fixed60"]
        if rr is not None and thr is not None:
            pos = [x for x in rr if x["has_subhalo"]]
            comp = {}
            for lo, hi in [(9.0, 9.5), (9.5, 10.0), (10.0, 10.5), (10.5, 11.0)]:
                sel = [x for x in pos if lo <= x["log10_M200_true"] < hi]
                comp[f"{lo}-{hi}"] = float(np.mean([x["delta_chi2"] >= thr for x in sel])) if sel else None
            det = [x for x in pos if x["delta_chi2"] >= thr]
            off = np.array([np.hypot(x["best_x"] - truth60[x["index"]]["subhalo"]["x"], x["best_y"] - truth60[x["index"]]["subhalo"]["y"]) for x in det])
            d["c60"] = {"completeness": comp, "n_detected": len(det), "frac_localized": float((off < LOC).mean()) if len(off) else None,
                        "clean_fpr_20": float(np.mean([x["delta_chi2"] >= 20 for x in rel["no_subhalo"]]))}
        else:
            d["c60"] = None
        out[key] = d
    (ROOT / "results/multipole_orders_summary.json").write_text(json.dumps(out, indent=2))

    def cell(key, f):
        try:
            return f(out[key])
        except (TypeError, KeyError):
            return "--"
    rows = [
        ("$\\chi^2/{\\rm dof}$ (median), $m{=}4$ truth", lambda d: fmt(d["multipole_m4_a3"]["chi2_dof_median"], pct=False, nd=2)),
        ("$\\chi^2/{\\rm dof}$ (median), $m{=}3$ truth", lambda d: fmt(d["multipole_m3_a3"]["chi2_dof_median"], pct=False, nd=2)),
        ("FPR, $m{=}4$ truth, $a_m{=}0.01\\,\\thetaE$: cal.\\ / ${>}20$", lambda d: pair(d["multipole_m4_a1"]["fpr_10pct"], d["multipole_m4_a1"]["fpr_20"])),
        ("FPR, $m{=}4$ truth: cal.\\ / ${>}20$", lambda d: pair(d["multipole_m4_a3"]["fpr_10pct"], d["multipole_m4_a3"]["fpr_20"])),
        ("FPR, $m{=}3$ truth: cal.\\ / ${>}20$", lambda d: pair(d["multipole_m3_a3"]["fpr_10pct"], d["multipole_m3_a3"]["fpr_20"])),
        ("fits rejected (of 300): clean / $m{=}4$ / $m{=}3$", lambda d: " / ".join("--" if d["rejected"][k] is None else str(d["rejected"][k]) for k in ("no_subhalo", "multipole_m4_a3", "multipole_m3_a3"))),
        ("FPR, clean, ${>}20$", lambda d: fmt(d["c60"]["clean_fpr_20"], nd=1)),
        ("compl.\\ $c{=}60$: 9--9.5 / 9.5--10 / 10--10.5", lambda d: " / ".join(f"{100*d['c60']['completeness'][k]:.0f}" for k in ("9.0-9.5", "9.5-10.0", "10.0-10.5")) + "\\%"),
        ("localized ($\\le2$ px)", lambda d: fmt(d["c60"]["frac_localized"])),
    ]
    lines = ["\\begin{tabular}{@{}lccc@{}}", "\\toprule", " & " + " & ".join(lab for _, lab, _ in COLS) + " \\\\", "\\midrule"]
    for name, f in rows:
        lines.append(name + " & " + " & ".join(cell(k, f) for k, _, _ in COLS) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (ROOT / "paper/tables/mpmacro.tex").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
