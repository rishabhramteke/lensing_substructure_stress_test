"""Data-only versus near-truth initialization of the JOINT scan (referee round 20).

results/baseline_a_joint_blind/<pop> re-runs the joint scan on the 300-lens seed-0 subsamples of the clean, c=60 and
a4/thetaE=0.03 populations with the single smooth start taken from the data alone (fit._blind_init_vec). The
reference is the paper's own joint scan (results/baseline_a_joint_full) restricted to the SAME lenses. Both are
scored identically: gate = best smooth or best joint chi^2/dof < 10 (regate_joint.py), threshold = 90th percentile
of the gated clean statistic of that run, plus the absolute Delta chi^2 >= 20. Writes results/joint_blind.json and
prints the numbers the paper quotes.

    python scripts/regate_joint.py --roots results/baseline_a_joint_blind && python scripts/evaluate_joint_blind.py
"""
import json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from paper_config import A_JOINT_FULL, BINS, LOC_ARCSEC, A_ABS, load_scan  # noqa: E402

POPS = ("no_subhalo", "test_fixed60", "multipole_m4_a3")
BLIND = "results/baseline_a_joint_blind"


def score(recs, truths, thr_cal):
    rel = [r for r in recs if r["reliable_fit"]]
    out = {"n": len(recs), "n_gated": len(recs) - len(rel),
           "n_gated_first_start": sum(1 for r in recs if not r.get("reliable_fit_initial", r["reliable_fit"]))}
    for op, thr in (("cal", thr_cal), ("abs", A_ABS)):
        flag = [r for r in rel if r["delta_chi2"] >= thr]
        d = {"flag_rate": len(flag) / max(1, len(rel))}
        sub = [r for r in rel if r["has_subhalo"]]
        if sub:
            comp = {}
            for lo, hi in BINS:
                sel = [r for r in sub if lo <= r["log10_M200_true"] < hi]
                comp[f"{lo}-{hi}"] = {"n": len(sel), "p": float(np.mean([r["delta_chi2"] >= thr for r in sel])) if sel else None}
            d["completeness"] = comp
            det = [r for r in sub if r["delta_chi2"] >= thr]
            off = [np.hypot(r["best_x"] - truths[r["index"]]["subhalo"]["x"], r["best_y"] - truths[r["index"]]["subhalo"]["y"]) for r in det]
            d["n_det"] = len(det)
            d["frac_localized"] = float(np.mean(np.array(off) < LOC_ARCSEC)) if det else None
            d["mass_error_median"] = float(np.median([r["best_log10_m"] - r["log10_M200_true"] for r in det])) if det else None
        out[op] = d
    return out


def main():
    res = {"threshold": {}, "blind": {}, "near_truth": {}, "same_decision": {}}
    blind = {p: load_scan(BLIND, p, require_manifest=True) for p in POPS}
    if any(v is None for v in blind.values()):
        print("blind runs unfinished:", [p for p, v in blind.items() if v is None]); return
    ref_all = {p: {r["index"]: r for r in load_scan(A_JOINT_FULL, p)} for p in POPS}
    ref = {p: [ref_all[p][r["index"]] for r in blind[p]] for p in POPS}
    truths = {p: [json.loads(l) for l in open(ROOT / "data" / p / "truth.jsonl")] for p in POPS}
    for tag, S in (("blind", blind), ("near_truth", ref)):
        clean = [r["delta_chi2"] for r in S["no_subhalo"] if r["reliable_fit"]]
        thr = float(np.quantile(clean, 0.90)); res["threshold"][tag] = thr
        res[tag] = {p: score(S[p], truths[p], thr) for p in POPS}
    # lens-by-lens agreement of the Delta chi^2 >= 20 decision on lenses both runs retain
    for p in POPS:
        both = [(b, r) for b, r in zip(blind[p], ref[p]) if b["reliable_fit"] and r["reliable_fit"]]
        agree = sum(1 for b, r in both if (b["delta_chi2"] >= A_ABS) == (r["delta_chi2"] >= A_ABS))
        res["same_decision"][p] = {"n_both_retained": len(both), "n_agree_at_20": agree,
                                   "median_abs_log_ratio": float(np.median([abs(np.log10(max(b["delta_chi2"], 1e-3) / max(r["delta_chi2"], 1e-3))) for b, r in both]))}
    (ROOT / "results/joint_blind.json").write_text(json.dumps(res, indent=2))
    print("thresholds (10% cal):", {k: round(v, 2) for k, v in res["threshold"].items()})
    for p in POPS:
        for tag in ("near_truth", "blind"):
            d = res[tag][p]
            line = f"{p:16s} {tag:10s} gated {d['n_gated']:3d} (first start {d['n_gated_first_start']:3d})  flag cal {100*d['cal']['flag_rate']:.1f}%  >=20 {100*d['abs']['flag_rate']:.2f}%"
            if "completeness" in d["cal"]:
                line += "  compl cal " + "/".join(f"{100*v['p']:.0f}" for v in d["cal"]["completeness"].values() if v["p"] is not None)
                line += "  >=20 " + "/".join(f"{100*v['p']:.0f}" for v in d["abs"]["completeness"].values() if v["p"] is not None)
                line += f"  loc {100*d['cal']['frac_localized']:.1f}%  merr {d['cal']['mass_error_median']:+.2f}"
            print(line)
        s = res["same_decision"][p]
        print(f"{'':16s} same >=20 decision on {s['n_agree_at_20']}/{s['n_both_retained']} lenses both retain; median |log10 ratio| {s['median_abs_log_ratio']:.3f}")


if __name__ == "__main__":
    main()
