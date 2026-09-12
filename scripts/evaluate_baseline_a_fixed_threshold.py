"""Re-score Family A at *absolute* Delta-chi^2 thresholds -- the literature's own kind of
operating point -- instead of (or alongside) the 10%-FPR calibration used everywhere else.

Why (self-review, 2026-09-11): forcing a 10%-FPR calibration onto Delta-chi^2 put the
threshold at -10.6, so 18 of 108 "detections" had actually made the fit *worse*. No
practitioner uses such a threshold; parametric scans report Delta ln Z > 10 (Nightingale+2024)
or Delta log E >= 50 (Despali+2022). Treating Delta ln L ~ Delta chi^2 / 2 and ignoring the
Occam penalty (which only makes evidence thresholds *stricter*), those correspond to
Delta chi^2 > 20 and > 100. This script reports FPR, completeness, localization and mass bias
at those thresholds from the scan results already on disk -- no new compute.

    source ~/myenv/bin/activate
    python scripts/evaluate_baseline_a_fixed_threshold.py --root results/baseline_a --thresholds 20 100

Writes <root>/summary_fixed_threshold/results.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MASS_BINS = [(8.0, 8.5), (8.5, 9.0), (9.0, 9.5), (9.5, 10.0), (10.0, 10.5), (10.5, 11.0)]
LOC_ARCSEC = 2.0 * 0.08


def load(root: Path, name: str):
    p = root / name / "scan_results.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else None


def score(root: Path, thr: float, truth60, truth15):
    out = {"threshold_delta_chi2": thr}
    t60, t15, ns, a1, a3 = (load(root, n) for n in ("test_fixed60", "test_fixed15", "no_subhalo", "multipole_m4_a1", "multipole_m4_a3"))

    def fpr(recs):
        rel = [r for r in recs if r["reliable_fit"]] if recs else []
        return (float(np.mean([r["delta_chi2"] >= thr for r in rel])) if rel else None), len(rel)

    for name, recs in (("no_subhalo", ns), ("multipole_m4_a1", a1), ("multipole_m4_a3", a3)):
        f, n = fpr(recs)
        out[f"fpr_{name}"] = f
        out[f"n_reliable_{name}"] = n

    def completeness(recs, truths):
        comp = {}
        rel = [r for r in recs if r["reliable_fit"] and r["has_subhalo"]]
        for lo, hi in MASS_BINS:
            sel = [r for r in rel if lo <= r["log10_M200_true"] < hi]
            comp[f"{lo}-{hi}"] = {"n": len(sel), "completeness": float(np.mean([r["delta_chi2"] >= thr for r in sel])) if sel else None}
        return comp

    out["completeness_c60_by_mass_bin"] = completeness(t60, truth60)
    if t15:
        out["completeness_c15_by_mass_bin"] = completeness(t15, truth15)

    # localization + mass bias among detections on the c=60 population
    det = [r for r in t60 if r["reliable_fit"] and r["has_subhalo"] and r["delta_chi2"] >= thr]
    d = np.array([np.hypot(r["best_x"] - truth60[r["index"]]["subhalo"]["x"], r["best_y"] - truth60[r["index"]]["subhalo"]["y"]) for r in det])
    loc = d < LOC_ARCSEC if len(d) else np.array([], dtype=bool)
    merr = np.array([r["best_log10_m"] - truth60[r["index"]]["subhalo"]["log10_M200"] for r in det])
    out["c60_detections"] = {
        "n_detected": len(det), "n_positive_reliable": sum(1 for r in t60 if r["reliable_fit"] and r["has_subhalo"]),
        "frac_localized_2px": float(loc.mean()) if len(d) else None, "n_localized": int(loc.sum()),
        "median_offset_arcsec": float(np.median(d)) if len(d) else None,
        "mass_error_dex_localized": {"n": int(loc.sum()),
                                     "median": float(np.median(merr[loc])) if loc.any() else None,
                                     "n_underestimated": int((merr[loc] < 0).sum()),
                                     "n_within_0p5dex": int((np.abs(merr[loc]) < 0.5).sum())},
        "mass_error_dex_all_detected": {"median": float(np.median(merr)) if len(merr) else None,
                                        "n_underestimated": int((merr < 0).sum())},
    }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=ROOT / "results/baseline_a")
    p.add_argument("--thresholds", type=float, nargs="+", default=[20.0, 100.0])
    p.add_argument("--truth60", type=Path, default=ROOT / "data/test_fixed60/truth.jsonl")
    p.add_argument("--truth15", type=Path, default=ROOT / "data/test_fixed15/truth.jsonl")
    args = p.parse_args()
    truth60 = [json.loads(l) for l in open(args.truth60)]
    truth15 = [json.loads(l) for l in open(args.truth15)] if args.truth15.exists() else None

    # reference: the 10%-FPR calibrated threshold, recomputed the same way evaluate_baseline_a.py does
    ns = load(args.root, "no_subhalo")
    neg = np.array([r["delta_chi2"] for r in ns if r["reliable_fit"]])
    thr10 = float(np.quantile(neg, 0.90))

    results = {"root": str(args.root), "reference_10pct_fpr_threshold": thr10,
               "note": "Delta chi^2 > 20 / > 100 approximate Delta ln Z > 10 (Nightingale+2024) / Delta log E >= 50 (Despali+2022) via Delta ln L ~ Delta chi^2/2, with no Occam penalty (which would only tighten them).",
               "at_threshold": [score(args.root, t, truth60, truth15) for t in [thr10] + list(args.thresholds)]}
    out = args.root / "summary_fixed_threshold"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2))

    for r in results["at_threshold"]:
        c = r["c60_detections"]
        print(f"thr={r['threshold_delta_chi2']:8.1f} | FPR no_sub={r['fpr_no_subhalo']!s:>6} a1={r['fpr_multipole_m4_a1']!s:>6} a3={r['fpr_multipole_m4_a3']!s:>6} | "
              f"c60 detected {c['n_detected']}/{c['n_positive_reliable']}  localized {c['frac_localized_2px']}  "
              f"mass err (localized) median {c['mass_error_dex_localized']['median']}  under {c['mass_error_dex_localized']['n_underestimated']}/{c['mass_error_dex_localized']['n']}")
    print(f"\nwrote {out}/results.json")


if __name__ == "__main__":
    main()
