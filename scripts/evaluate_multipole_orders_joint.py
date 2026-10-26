"""Multipole freedom in the JOINT scan's macro-model (referee round 19, 2026-09-18).

Table B.4 (evaluate_multipole_orders.py) tested a free m=4 term only on the fixed-macro scan. This evaluator does
the same for the joint scan, the method the headline multipole rates describe: the bare EPL+shear joint scan
(results/baseline_a_joint_full, restricted to the SAME 300-lens seed subsamples the freer runs used), the joint scan
with a free m=4 term (results/baseline_a_joint_mp4) and with free m=3 and m=4 terms (results/baseline_a_joint_mp34).
Each model's 10%-FPR threshold is calibrated on its own clean scan; the absolute Delta chi^2 >= 20 threshold is
also scored. Completeness is on the c=60 population; the paired flip c=60 -> 15 where the c=15 run exists.

    python scripts/evaluate_multipole_orders_joint.py
-> results/multipole_orders_joint_summary.json, paper/tables/mpmacro_joint.tex
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts"))
from paper_config import A_JOINT_FULL, A_JOINT_MP4, A_JOINT_MP34, A_ABS, BINS, LOC_ARCSEC, wilson  # noqa: E402

SEEDS = {"no_subhalo": 1, "multipole_m4_a1": 2, "multipole_m4_a3": 3, "multipole_m3_a3": 3, "test_fixed60": 0, "test_fixed15": 0}
POPS = ("no_subhalo", "multipole_m4_a1", "multipole_m4_a3", "multipole_m3_a3", "test_fixed60", "test_fixed15")
COLS = [("bare", "bare", A_JOINT_FULL), ("mp4", "$+m{=}4$", A_JOINT_MP4), ("mp34", "$+m{=}3{+}4$", A_JOINT_MP34)]
truth60 = [json.loads(l) for l in open(ROOT / "data/test_fixed60/truth.jsonl")]


def subsample_indices(pop, n_pop=1000, n_sub=300):
    """The indices run_baseline_a_joint.py (and run_baseline_a.py) draw for --seed s --n-subsample 300."""
    rng = np.random.default_rng(SEEDS[pop])
    return set(int(i) for i in np.sort(rng.choice(n_pop, size=n_sub, replace=False)))


def load(root, pop, restrict=False):
    p = ROOT / root / pop / "scan_results.jsonl"
    if not p.exists() or not (ROOT / root / pop / "manifest.json").exists():
        return None
    recs = [json.loads(l) for l in open(p)]
    if restrict:
        idx = subsample_indices(pop); recs = [r for r in recs if r["index"] in idx]
    return recs


def pct_str(v, force_nd=None):
    if v is None:
        return "--"
    x = 100 * v
    if force_nd is not None:
        return f"{x:.{force_nd}f}"
    return f"{x:.2f}" if x < 1 else (f"{x:.1f}" if x < 10 else f"{x:.0f}")


def main():
    out = {}
    for key, label, root in COLS:
        R = {pop: load(root, pop, restrict=(key == "bare")) for pop in POPS}
        rel = {pop: ([x for x in r if x["reliable_fit"]] if r else None) for pop, r in R.items()}
        d = {"label": label, "root": root, "n": {pop: (len(r) if r else 0) for pop, r in R.items()},
             "rejected": {pop: (len(r) - len(rel[pop]) if r else None) for pop, r in R.items()},
             "truncated": {pop: (sum(1 for x in r if x.get("truncated")) if r else None) for pop, r in R.items()}}
        thr = float(np.quantile([x["delta_chi2"] for x in rel["no_subhalo"]], 0.90)) if rel["no_subhalo"] else None
        d["threshold_10pct_fpr"] = thr
        if rel["no_subhalo"]:
            d["clean"] = {"chi2_dof_median": float(np.median([x["chi2_smooth_per_dof"] for x in R["no_subhalo"]])),
                          "fpr_20": float(np.mean([x["delta_chi2"] >= A_ABS for x in rel["no_subhalo"]])), "n_reliable": len(rel["no_subhalo"])}
        for pop in ("multipole_m4_a1", "multipole_m4_a3", "multipole_m3_a3"):
            rr = rel[pop]
            d[pop] = None if (rr is None or thr is None) else {
                "chi2_dof_median": float(np.median([x["chi2_smooth_per_dof"] for x in R[pop]])),
                "fpr_10pct": float(np.mean([x["delta_chi2"] >= thr for x in rr])), "fpr_20": float(np.mean([x["delta_chi2"] >= A_ABS for x in rr])),
                "fpr_100": float(np.mean([x["delta_chi2"] >= 100 for x in rr])), "n_reliable": len(rr),
                "dchi2_median": float(np.median([x["delta_chi2"] for x in rr]))}
        rr = rel["test_fixed60"]
        d["c60"] = None
        if rr is not None and thr is not None:
            pos = [x for x in rr if x["has_subhalo"]]
            comp = {}
            for name, t in (("cal", thr), ("abs", A_ABS)):
                comp[name] = {}
                for lo, hi in BINS:
                    sel = [x for x in pos if lo <= x["log10_M200_true"] < hi]
                    k = sum(1 for x in sel if x["delta_chi2"] >= t); p_, l_, h_ = wilson(k, len(sel))
                    comp[name][f"{lo}-{hi}"] = {"n": len(sel), "k": k, "p": p_, "lo": l_, "hi": h_}
            det = [x for x in pos if x["delta_chi2"] >= thr]
            off = np.array([np.hypot(x["best_x"] - truth60[x["index"]]["subhalo"]["x"], x["best_y"] - truth60[x["index"]]["subhalo"]["y"]) for x in det])
            merr = np.array([x["best_log10_m"] - truth60[x["index"]]["subhalo"]["log10_M200"] for x in det])
            d["c60"] = {"completeness": comp, "n_detected": len(det), "frac_localized": float((off < LOC_ARCSEC).mean()) if len(off) else None,
                        "mass_error_median": float(np.median(merr)) if len(merr) else None}
            if rel["test_fixed15"]:
                a = {x["index"]: x for x in pos}; b = {x["index"]: x for x in rel["test_fixed15"] if x["has_subhalo"]}
                common = sorted(set(a) & set(b))
                d["flip_60_to_15"] = {n: {"n_pairs": len(common), "lost": sum(1 for i in common if a[i]["delta_chi2"] >= t and b[i]["delta_chi2"] < t),
                                          "gained": sum(1 for i in common if b[i]["delta_chi2"] >= t and a[i]["delta_chi2"] < t)} for n, t in (("cal", thr), ("abs", A_ABS))}
        out[key] = d
    (ROOT / "results/multipole_orders_joint_summary.json").write_text(json.dumps(out, indent=2))

    def cell(key, f):
        try:
            v = f(out[key])
            return "--" if v is None else v
        except (TypeError, KeyError, ZeroDivisionError):
            return "--"
    def pair(a, b):
        return f"{pct_str(a)}\\,/\\,{pct_str(b)}\\%"
    def comp4(d, op):
        c = d["c60"]["completeness"][op]
        return "/".join(f"{100*c[k]['p']:.0f}" for k in ("8.5-9.0", "9.0-9.5", "9.5-10.0", "10.0-10.5")) + "\\%"
    rows = [
        ("threshold at 10\\% FPR", lambda d: f"${d['threshold_10pct_fpr']:.1f}$"),
        ("$\\chi^2/{\\rm dof}$, $m{=}4$\\,/\\,$m{=}3$ truth", lambda d: f"{d['multipole_m4_a3']['chi2_dof_median']:.2f}\\,/\\,{d['multipole_m3_a3']['chi2_dof_median']:.2f}"),
        ("FPR, clean, $\\dchi\\ge20$", lambda d: pct_str(d["clean"]["fpr_20"]) + "\\%"),
        ("FPR, $m{=}4$, $a_4{=}0.01\\,\\thetaE$: cal.\\,/\\,${\\ge}20$", lambda d: pair(d["multipole_m4_a1"]["fpr_10pct"], d["multipole_m4_a1"]["fpr_20"])),
        ("FPR, $m{=}4$, $a_4{=}0.03\\,\\thetaE$: cal.\\,/\\,${\\ge}20$", lambda d: pair(d["multipole_m4_a3"]["fpr_10pct"], d["multipole_m4_a3"]["fpr_20"])),
        ("FPR, $m{=}3$, $a_3{=}0.03\\,\\thetaE$: cal.\\,/\\,${\\ge}20$", lambda d: pair(d["multipole_m3_a3"]["fpr_10pct"], d["multipole_m3_a3"]["fpr_20"])),
        ("median $\\dchi$, $m{=}4$, $a_4{=}0.03\\,\\thetaE$", lambda d: f"{d['multipole_m4_a3']['dchi2_median']:.0f}"),
        ("removed by the fit-quality cut (of 300): clean/$m{=}4$/$m{=}3$/$c{=}60$", lambda d: "/".join("--" if d["rejected"][k] is None else str(d["rejected"][k]) for k in ("no_subhalo", "multipole_m4_a3", "multipole_m3_a3", "test_fixed60"))),
        ("compl.\\ $c{=}60$, 10\\% cal.$^{a}$", lambda d: comp4(d, "cal")),
        ("compl.\\ $c{=}60$, $\\dchi\\ge20$$^{a}$", lambda d: comp4(d, "abs")),
        ("paired flip $c{=}60\\to15$, $\\dchi\\ge20$", lambda d: f"{d['flip_60_to_15']['abs']['lost']}\\,:\\,{d['flip_60_to_15']['abs']['gained']}"),
        ("localized ($\\le2$ px)", lambda d: pct_str(d["c60"]["frac_localized"]) + "\\%"),
        ("mass error, median (dex)", lambda d: f"${d['c60']['mass_error_median']:+.2f}$"),
    ]
    lines = ["\\begin{tabular}{@{}lccc@{}}", "\\toprule", " & " + " & ".join(lab for _, lab, _ in COLS) + " \\\\", "\\midrule"]
    for name, f in rows:
        lines.append(name + " & " + " & ".join(cell(k, f) for k, _, _ in COLS) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (ROOT / "paper/tables/mpmacro_joint.tex").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    for k, d in out.items():
        print(k, "n", d["n"], "rejected", d["rejected"], "truncated", d["truncated"], "thr", d["threshold_10pct_fpr"])


if __name__ == "__main__":
    main()
