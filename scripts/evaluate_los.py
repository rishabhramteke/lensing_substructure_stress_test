"""Referee round 7, point M7: a line-of-sight halo population.

Sengul et al. (2022) reanalysed the field's first "dark perturber" as a line-of-sight halo
rather than a subhalo, and Gilman et al. (2020) show the line-of-sight population dominates
in some configurations. `data/los_halo_z*` contains NO subhalo at the lens plane and one
halo on a second lens plane (multi-plane ray tracing), drawn from the same mass range and
the same projected annulus, with the field concentration-mass relation since a line-of-sight
halo is not tidally stripped.

Family A is run unchanged: it assumes any perturber lies at the lens redshift. So the
question is not only "is it flagged" (it is) but "what mass is inferred when the redshift
assumption is wrong", which is the quantity a substructure mass function is built from.

    python scripts/evaluate_los.py
-> results/los_halo.json, paper/tables/los.tex
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BINS = [(8.0, 9.0), (9.0, 9.5), (9.5, 10.0), (10.0, 10.5), (10.5, 11.0)]
LOC = 0.16
ARMS = {"z025": ("results/baseline_a_los/los_z025", "data/los_halo_z025"),
        "z075": ("results/baseline_a_los/los_z075", "data/los_halo_z075")}


def wilson(k, n, z=1.0):
    if not n:
        return (np.nan,) * 3
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def main():
    clean = [json.loads(l) for l in open(ROOT / "results/baseline_a/no_subhalo/scan_results.jsonl")]
    rel_clean = [r for r in clean if r["reliable_fit"]]
    thr = float(np.quantile([r["delta_chi2"] for r in rel_clean], 0.90))
    res = {"threshold_10pct": thr, "clean_fpr_floor": float(np.mean([r["delta_chi2"] > 0 for r in rel_clean])), "arms": {}}
    for arm, (root, data) in ARMS.items():
        p = ROOT / root / "scan_results.jsonl"
        if not p.exists():
            continue
        recs = [json.loads(l) for l in open(p)]
        truths = [json.loads(l) for l in open(ROOT / data / "truth.jsonl")]
        rel = [r for r in recs if r["reliable_fit"]]
        d = {"n": len(recs), "n_reliable": len(rel), "n_excluded": len(recs) - len(rel),
             "z_halo": truths[0]["los_halo"]["z_halo"]}
        for name, f in (("cal", lambda r: r["delta_chi2"] >= thr), ("floor", lambda r: r["delta_chi2"] > 0)):
            d[f"flag_rate_{name}"] = float(np.mean([f(r) for r in rel]))
            b = {}
            for lo, hi in BINS:
                sel = [r for r in rel if lo <= truths[r["index"]]["los_halo"]["log10_M200"] < hi]
                k = sum(1 for r in sel if f(r)); p_, l_, h_ = wilson(k, len(sel))
                b[f"{lo}-{hi}"] = {"n": len(sel), "k": k, "p": p_, "lo": l_, "hi": h_}
            d[f"bins_{name}"] = b
        # mass and position error among floored detections, treating the halo as if it were a subhalo
        det = [r for r in rel if r["delta_chi2"] > 0]
        if det:
            off = np.array([np.hypot(r["best_x"] - truths[r["index"]]["los_halo"]["x"],
                                     r["best_y"] - truths[r["index"]]["los_halo"]["y"]) for r in det])
            merr = np.array([r["best_log10_m"] - truths[r["index"]]["los_halo"]["log10_M200"] for r in det])
            loc = off < LOC
            d["detections"] = {"n": len(det), "frac_localized": float(loc.mean()), "n_localized": int(loc.sum()),
                               "median_offset": float(np.median(off)),
                               "mass_error_localized_median": float(np.median(merr[loc])) if loc.any() else None,
                               "mass_error_all_median": float(np.median(merr)),
                               "n_low": int((merr < 0).sum())}
        res["arms"][arm] = d
    (ROOT / "results/los_halo.json").write_text(json.dumps(res, indent=2))

    a, b = res["arms"].get("z025"), res["arms"].get("z075")
    lines = [r"\begin{tabular}{@{}lcc@{}}", r"\toprule",
             r" & foreground & background \\", r"line-of-sight halo redshift & $z=0.25$ & $z=0.75$ \\", r"\midrule",
             f"flagged at $\\dchi>0$ & {100*a['flag_rate_floor']:.0f}\\% & {100*b['flag_rate_floor']:.0f}\\% \\\\",
             f"flagged at the 10\\% threshold & {100*a['flag_rate_cal']:.0f}\\% & {100*b['flag_rate_cal']:.0f}\\% \\\\",
             f"macro fits rejected (of {a['n']}) & {a['n_excluded']} & {b['n_excluded']} \\\\"]
    for lo, hi in BINS[1:]:
        k = f"{lo}-{hi}"
        lines.append(f"\\quad flagged, $10^{{{lo}}}$--$10^{{{hi}}}\\Msun$ & {100*a['bins_floor'][k]['p']:.0f}\\% & {100*b['bins_floor'][k]['p']:.0f}\\% \\\\")
    lines.append(r"\addlinespace[2pt]")
    lines.append(f"localized ($\\le2$ px) & {100*a['detections']['frac_localized']:.0f}\\% & {100*b['detections']['frac_localized']:.0f}\\% \\\\")
    lines.append(f"inferred minus true $\\log_{{10}}M$ (dex) & ${a['detections']['mass_error_all_median']:+.2f}$ & ${b['detections']['mass_error_all_median']:+.2f}$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "paper/tables/los.tex").write_text("\n".join(lines) + "\n")
    print(f"threshold {thr:.1f} | clean floored FPR {100*res['clean_fpr_floor']:.2f}%")
    for arm, d in res["arms"].items():
        print(f"== {arm} (z={d['z_halo']}): flagged floor {100*d['flag_rate_floor']:.0f}%  cal {100*d['flag_rate_cal']:.0f}%  excluded {d['n_excluded']}/{d['n']}")
        print("   by mass:", {k: f"{100*v['p']:.0f}%(n{v['n']})" for k, v in d["bins_floor"].items()})
        de = d["detections"]
        print(f"   detections {de['n']}: localized {100*de['frac_localized']:.0f}% ({de['n_localized']}), median offset {de['median_offset']:.2f}\", mass err all {de['mass_error_all_median']:+.2f} dex, localized {de['mass_error_localized_median']}")


if __name__ == "__main__":
    main()
