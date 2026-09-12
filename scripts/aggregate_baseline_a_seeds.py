"""Aggregate Family A's 2 seeds (0 and 200) into mean +/- std, mirroring
scripts/aggregate_seeds.py's treatment of the U-Net's 4 seeds.

    source ~/myenv/bin/activate
    python scripts/aggregate_baseline_a_seeds.py

n=2 is a minimal repeat, not a powered study -- enough to say whether the
single-run numbers were a fluke of that particular 300-image draw, not enough
for a tight confidence interval. Each seed's threshold is calibrated on that
seed's OWN no_subhalo draw (matching evaluate_baseline_a.py's protocol
exactly), not shared across seeds -- so seed-to-seed spread in the final
numbers already includes threshold-calibration noise, not just scan noise.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

MASS_BINS = [(8.0, 8.5), (8.5, 9.0), (9.0, 9.5), (9.5, 10.0), (10.0, 10.5), (10.5, 11.0)]
SEED_DIRS = {k: v for k, v in {0: Path("results/baseline_a"), 200: Path("results/baseline_a_seed1"), 300: Path("results/baseline_a_seed300")}.items()
             if (v / "multipole_m4_a3" / "scan_results.jsonl").exists()}   # a third seed (300) is picked up automatically once its five populations exist


def load(root: Path, name: str):
    recs = [json.loads(l) for l in open(root / name / "scan_results.jsonl")]
    return recs


def completeness_by_bin(recs, thr):
    out = {}
    reliable = [r for r in recs if r["reliable_fit"]]
    for lo, hi in MASS_BINS:
        sel = [r for r in reliable if r["has_subhalo"] and lo <= r["log10_M200_true"] < hi]
        out[f"{lo}-{hi}"] = float(np.mean([r["delta_chi2"] >= thr for r in sel])) if sel else None
    return out


def one_seed(root: Path):
    test60 = load(root, "test_fixed60")
    test15 = load(root, "test_fixed15")
    no_sub = load(root, "no_subhalo")
    mp_a1 = load(root, "multipole_m4_a1")
    mp_a3 = load(root, "multipole_m4_a3")

    no_sub_reliable = [r for r in no_sub if r["reliable_fit"]]
    neg_scores = np.array([r["delta_chi2"] for r in no_sub_reliable])
    thr = float(np.quantile(neg_scores, 0.90))

    comp60 = completeness_by_bin(test60, thr)
    comp15 = completeness_by_bin(test15, thr)

    r60_by_idx = {r["index"]: r for r in test60 if r["has_subhalo"] and r["reliable_fit"]}
    r15_by_idx = {r["index"]: r for r in test15 if r["reliable_fit"]}
    paired_idx = [i for i in r60_by_idx if i in r15_by_idx]
    det60 = np.array([r60_by_idx[i]["delta_chi2"] >= thr for i in paired_idx])
    det15 = np.array([r15_by_idx[i]["delta_chi2"] >= thr for i in paired_idx])

    def fpr_of(recs):
        rel = [r for r in recs if r["reliable_fit"]]
        return float(np.mean([r["delta_chi2"] >= thr for r in rel]))

    return {
        "threshold": thr,
        "completeness_c60": comp60, "completeness_c15": comp15,
        "n_pairs": len(paired_idx), "c60_not_c15": int((det60 & ~det15).sum()), "c15_not_c60": int((det15 & ~det60).sum()),
        "fpr_no_subhalo": fpr_of(no_sub), "fpr_mp_a1": fpr_of(mp_a1), "fpr_mp_a3": fpr_of(mp_a3),
    }


def mean_std(vals):
    a = np.array([v for v in vals if v is not None], dtype=float)
    if len(a) == 0:
        return None, None
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def main():
    per_seed = {seed: one_seed(root) for seed, root in SEED_DIRS.items()}
    print("per-seed thresholds:", {s: r["threshold"] for s, r in per_seed.items()})

    agg = {"n_seeds": len(per_seed), "seeds": list(per_seed.keys())}
    for key in ("completeness_c60", "completeness_c15"):
        agg[key] = {}
        for lo, hi in MASS_BINS:
            k = f"{lo}-{hi}"
            m, s = mean_std([per_seed[seed][key][k] for seed in per_seed])
            agg[key][k] = {"mean": m, "std": s, "values": [per_seed[seed][key][k] for seed in per_seed]}

    for key in ("fpr_no_subhalo", "fpr_mp_a1", "fpr_mp_a3"):
        vals = [per_seed[seed][key] for seed in per_seed]
        m, s = mean_std(vals)
        agg[key] = {"mean": m, "std": s, "values": vals}

    c60_not_c15 = [per_seed[seed]["c60_not_c15"] for seed in per_seed]
    c15_not_c60 = [per_seed[seed]["c15_not_c60"] for seed in per_seed]
    agg["paired_flip"] = {
        "n_pairs": [per_seed[seed]["n_pairs"] for seed in per_seed],
        "c60_not_c15_mean": float(np.mean(c60_not_c15)), "c60_not_c15_values": c60_not_c15,
        "c15_not_c60_mean": float(np.mean(c15_not_c60)), "c15_not_c60_values": c15_not_c60,
    }

    out = Path("results/baseline_a/aggregate")
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(agg, indent=2))
    print(json.dumps(agg, indent=2))
    print(f"\nwrote {out}/summary.json")


if __name__ == "__main__":
    main()
