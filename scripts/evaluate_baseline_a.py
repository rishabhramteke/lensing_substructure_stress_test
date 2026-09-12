"""Turn Family-A scan_results.jsonl files into the same RQ2/RQ4 numbers as
scripts/evaluate_detector.py, for a direct cross-family (RQ1) comparison.

    source ~/myenv/bin/activate
    python scripts/evaluate_baseline_a.py --root results/baseline_a --out results/baseline_a/summary

Threshold calibrated for 10% FPR on `no_subhalo`'s delta_chi2 distribution,
exactly mirroring the U-Net evaluation protocol. Unreliable fits (see
run_baseline_a.py) are excluded from FPR/completeness denominators and
reported separately -- silently keeping them would let optimizer failures
masquerade as either detections or non-detections.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MASS_BINS = [(8.0, 8.5), (8.5, 9.0), (9.0, 9.5), (9.5, 10.0), (10.0, 10.5), (10.5, 11.0)]


def load(root: Path, name: str):
    recs = [json.loads(l) for l in open(root / name / "scan_results.jsonl")]
    manifest = json.loads((root / name / "manifest.json").read_text())
    return recs, manifest


def completeness_by_bin(recs, thr):
    out = {}
    reliable = [r for r in recs if r["reliable_fit"]]
    for lo, hi in MASS_BINS:
        sel = [r for r in reliable if r["has_subhalo"] and lo <= r["log10_M200_true"] < hi]
        out[f"{lo}-{hi}"] = {"n": len(sel), "completeness": (float(np.mean([r["delta_chi2"] >= thr for r in sel])) if sel else None)}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("results/baseline_a"))
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    test60, m60 = load(args.root, "test_fixed60")
    test15, m15 = load(args.root, "test_fixed15")
    no_sub, m_ns = load(args.root, "no_subhalo")
    mp_a1, m_a1 = load(args.root, "multipole_m4_a1")
    mp_a3, m_a3 = load(args.root, "multipole_m4_a3")

    no_sub_reliable = [r for r in no_sub if r["reliable_fit"]]
    neg_scores = np.array([r["delta_chi2"] for r in no_sub_reliable])
    thr = float(np.quantile(neg_scores, 0.90))
    fpr_calib = float((neg_scores >= thr).mean())

    comp60 = completeness_by_bin(test60, thr)
    comp15 = completeness_by_bin(test15, thr)

    # paired flip -- test_fixed60/test_fixed15 share --seed, so the SAME subset
    # indices were drawn from both; match by "index" field (position in the
    # original population) to pair correctly even if reliability differs.
    r60_by_idx = {r["index"]: r for r in test60 if r["has_subhalo"] and r["reliable_fit"]}
    r15_by_idx = {r["index"]: r for r in test15 if r["reliable_fit"]}
    paired_idx = [i for i in r60_by_idx if i in r15_by_idx]
    det60 = np.array([r60_by_idx[i]["delta_chi2"] >= thr for i in paired_idx])
    det15 = np.array([r15_by_idx[i]["delta_chi2"] >= thr for i in paired_idx])
    paired_flip = {
        "n_pairs": len(paired_idx), "detected_at_c60": int(det60.sum()), "detected_at_c15": int(det15.sum()),
        "detected_at_c60_not_c15": int((det60 & ~det15).sum()), "detected_at_c15_not_c60": int((det15 & ~det60).sum()),
    }

    def fpr_of(recs):
        rel = [r for r in recs if r["reliable_fit"]]
        return float(np.mean([r["delta_chi2"] >= thr for r in rel])), len(rel), len(recs) - len(rel)

    fpr_a1, n_a1, nu_a1 = fpr_of(mp_a1)
    fpr_a3, n_a3, nu_a3 = fpr_of(mp_a3)

    results = {
        "threshold_delta_chi2_at_10pct_fpr": thr, "fpr_at_threshold_calibration_set": fpr_calib,
        "n_no_subhalo_reliable": len(no_sub_reliable), "n_no_subhalo_unreliable": len(no_sub) - len(no_sub_reliable),
        "completeness_c60_by_mass_bin": comp60, "completeness_c15_by_mass_bin": comp15,
        "paired_concentration_flip": paired_flip,
        "confounder_fpr": {
            "no_subhalo (calibration set)": fpr_calib,
            "multipole m=4, a=0.01*thetaE": fpr_a1,
            "multipole m=4, a=0.03*thetaE": fpr_a3,
        },
        "n_unreliable_fits": {"test_fixed60": sum(1 for r in test60 if not r["reliable_fit"]),
                              "test_fixed15": sum(1 for r in test15 if not r["reliable_fit"]),
                              "no_subhalo": len(no_sub) - len(no_sub_reliable), "multipole_a1": nu_a1, "multipole_a3": nu_a3},
        "mean_time_per_image_s": float(np.mean([r["t_fit_s"] + r["t_scan_s"] for r in test60])),
    }
    (args.out / "results.json").write_text(json.dumps(results, indent=2))

    mids = [(lo + hi) / 2 for lo, hi in MASS_BINS]
    c60 = [comp60[f"{lo}-{hi}"]["completeness"] for lo, hi in MASS_BINS]
    c15 = [comp15[f"{lo}-{hi}"]["completeness"] for lo, hi in MASS_BINS]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(mids, c60, "o-", label="c = 60 (Tsang+2024 fiducial)", color="#1e64a8")
    ax.plot(mids, c15, "s--", label="c = 15 (their low-c ablation)", color="#c8177a")
    ax.axhline(0.10, color="gray", ls=":", lw=1, label="FPR operating point")
    ax.set_xlabel("log10(subhalo M200 / Msun)"); ax.set_ylabel("completeness @ FPR=10%")
    ax.set_title("Family A (parametric scan) — completeness vs. concentration")
    ax.legend(fontsize=8); ax.set_ylim(-0.02, 1.02); fig.tight_layout()
    fig.savefig(args.out / "completeness_vs_mass.png", dpi=140); plt.close(fig)

    print(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}/results.json + completeness_vs_mass.png")


if __name__ == "__main__":
    main()
