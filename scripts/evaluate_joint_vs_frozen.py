"""Frozen macro-model scan vs joint-refit scan on the SAME lenses (first 100 of the seed-0/1/3
subsamples). Answers the referee's question: does the fit-then-scan shortcut also drive the
confounder false-positive rate and the 15% localization, or only the mass?

    python scripts/evaluate_joint_vs_frozen.py                 # c=15 (paper headline) + c=60 mass check
Writes results/baseline_a_joint_c15/summary_vs_frozen.json
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOC = 0.16
BINS = [(8.0, 8.5), (8.5, 9.0), (9.0, 9.5), (9.5, 10.0), (10.0, 10.5), (10.5, 11.0)]
truth60 = [json.loads(l) for l in open(ROOT / "data/test_fixed60/truth.jsonl")]


def load(p):
    return {r["index"]: r for r in (json.loads(l) for l in open(p))} if p.exists() else None


def evaluate(root_joint, root_frozen, label):
    out = {"label": label}
    J = {n: load(ROOT / root_joint / n / "scan_results.jsonl") for n in ("no_subhalo", "multipole_m4_a3", "test_fixed60")}
    F = {n: load(ROOT / root_frozen / n / "scan_results.jsonl") for n in ("no_subhalo", "multipole_m4_a3", "test_fixed60")}
    for kind, R in (("joint", J), ("frozen_same_lenses", F)):
        res = {}
        # restrict the frozen run to the lenses the joint run has
        idx = {n: sorted(J[n].keys()) if J[n] else [] for n in J}
        ns = [R["no_subhalo"][i] for i in idx["no_subhalo"] if i in R["no_subhalo"]] if R["no_subhalo"] else []
        rel_ns = [r for r in ns if r["reliable_fit"]]
        thr = float(np.quantile([r["delta_chi2"] for r in rel_ns], 0.90)) if rel_ns else None
        res["threshold_10pct_fpr"] = thr
        res["n_no_subhalo_reliable"] = len(rel_ns)
        if R["multipole_m4_a3"] and thr is not None:
            mp = [R["multipole_m4_a3"][i] for i in idx["multipole_m4_a3"] if i in R["multipole_m4_a3"]]
            rel = [r for r in mp if r["reliable_fit"]]
            res["fpr_multipole_a3"] = {"at_10pct": float(np.mean([r["delta_chi2"] >= thr for r in rel])),
                                       "at_20": float(np.mean([r["delta_chi2"] >= 20 for r in rel])),
                                       "at_100": float(np.mean([r["delta_chi2"] >= 100 for r in rel])), "n_reliable": len(rel)}
            res["fpr_clean"] = {"at_20": float(np.mean([r["delta_chi2"] >= 20 for r in rel_ns])), "at_100": float(np.mean([r["delta_chi2"] >= 100 for r in rel_ns]))}
            res["multipole_median_dchi2"] = float(np.median([r["delta_chi2"] for r in rel]))
        if R["test_fixed60"] and thr is not None:
            t = [R["test_fixed60"][i] for i in idx["test_fixed60"] if i in R["test_fixed60"]]
            rel = [r for r in t if r["reliable_fit"] and r["has_subhalo"]]
            det = [r for r in rel if r["delta_chi2"] >= thr]
            d = np.array([np.hypot(r["best_x"] - truth60[r["index"]]["subhalo"]["x"], r["best_y"] - truth60[r["index"]]["subhalo"]["y"]) for r in det])
            loc = d < LOC
            merr = np.array([r["best_log10_m"] - truth60[r["index"]]["subhalo"]["log10_M200"] for r in det])
            comp = {}
            for lo, hi in BINS:
                sel = [r for r in rel if lo <= r["log10_M200_true"] < hi]
                comp[f"{lo}-{hi}"] = {"n": len(sel), "completeness": float(np.mean([r["delta_chi2"] >= thr for r in sel])) if sel else None}
            res["c60"] = {"n_positive_reliable": len(rel), "n_detected": len(det), "frac_localized_2px": float(loc.mean()) if len(d) else None,
                          "n_localized": int(loc.sum()), "median_offset_arcsec": float(np.median(d)) if len(d) else None,
                          "completeness": comp,
                          "mass_error_localized": {"n": int(loc.sum()), "median": float(np.median(merr[loc])) if loc.any() else None,
                                                   "n_low": int((merr[loc] < 0).sum()), "n_within_0p25": int((np.abs(merr[loc]) < 0.25).sum()), "n_within_0p5": int((np.abs(merr[loc]) < 0.5).sum())},
                          "mass_error_all_detected": {"median": float(np.median(merr)) if len(merr) else None, "n_low": int((merr < 0).sum()), "n": len(merr)},
                          "median_dchi2_detected": float(np.median([r["delta_chi2"] for r in det])) if det else None}
        out[kind] = res
    # paired per-lens delta: joint minus frozen Delta chi2 on the multipole population
    if J["multipole_m4_a3"] and F["multipole_m4_a3"]:
        common = [i for i in J["multipole_m4_a3"] if i in F["multipole_m4_a3"] and J["multipole_m4_a3"][i]["reliable_fit"] and F["multipole_m4_a3"][i]["reliable_fit"]]
        gain = np.array([J["multipole_m4_a3"][i]["delta_chi2"] - F["multipole_m4_a3"][i]["delta_chi2"] for i in common])
        out["multipole_joint_minus_frozen_dchi2"] = {"n": len(common), "median": float(np.median(gain)), "frac_positive": float((gain > 0).mean())}
    if J["no_subhalo"] and F["no_subhalo"]:
        common = [i for i in J["no_subhalo"] if i in F["no_subhalo"] and J["no_subhalo"][i]["reliable_fit"] and F["no_subhalo"][i]["reliable_fit"]]
        gain = np.array([J["no_subhalo"][i]["delta_chi2"] - F["no_subhalo"][i]["delta_chi2"] for i in common])
        out["clean_joint_minus_frozen_dchi2"] = {"n": len(common), "median": float(np.median(gain))}
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--joint-root", default="results/baseline_a_joint_c15"); ap.add_argument("--tag", default="")
    args = ap.parse_args()
    results = {"c15": evaluate(args.joint_root, "results/baseline_a", f"joint vs frozen, c=15, first 100 lenses of the seed-0 subsamples ({args.joint_root})")}
    # null-control fields, if the joint run carries them
    jr = ROOT / args.joint_root / "multipole_m4_a3" / "scan_results.jsonl"
    if jr.exists():
        recs = [json.loads(l) for l in open(jr)]
        if "chi2_polish_gain" in recs[0]:
            rel = [r for r in recs if r["reliable_fit"]]
            results["null_control"] = {"median_polish_gain_multipole": float(np.median([r["chi2_polish_gain"] for r in rel])),
                                       "median_polish_gain_clean": float(np.median([r["chi2_polish_gain"] for r in (json.loads(l) for l in open(ROOT / args.joint_root / "no_subhalo" / "scan_results.jsonl")) if r["reliable_fit"]])),
                                       "median_dchi2_vs_unpolished_multipole": float(np.median([r["delta_chi2_vs_unpolished_smooth"] for r in rel])),
                                       "median_dchi2_vs_polished_multipole": float(np.median([r["delta_chi2"] for r in rel]))}
    if (ROOT / "results/baseline_a_joint_c60/test_fixed60/scan_results.jsonl").exists():
        results["c60_mass_only"] = evaluate("results/baseline_a_joint_c60", "results/baseline_a_c60", "joint vs frozen, c=60, test_fixed60 only (threshold from c15 joint run not applicable)")
    (ROOT / args.joint_root / f"summary_vs_frozen{args.tag}.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
