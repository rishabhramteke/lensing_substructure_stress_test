"""Family B (free-form residual statistic, fitted macro-model, lambda = 1e5) on the stress arms it is scored on in
referee round 21: the 0.15" PSF, the 600 s and 135 s depths, the combined realism arm and the two line-of-sight
populations. Each arm is calibrated at 10% FPR on its own clean control (the line-of-sight arms on the Tier-0 clean
control, as for Family A). 300-lens subsamples with Family A's subsample seeds.

    python scripts/evaluate_family_b_arms.py        -> results/family_b_arms.json
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BINS = [(8.0, 8.5), (8.5, 9.0), (9.0, 9.5), (9.5, 10.0), (10.0, 10.5), (10.5, 11.0)]
ARMS = {
    "psf015": ("results/baseline_b_psf015/fitted", "psf015_no_subhalo", "psf015_fixed60", "psf015_fixed15", "psf015_multipole_m4_a3"),
    "shallow600": ("results/baseline_b_shallow600/fitted", "test_shallow600_no_subhalo", "test_shallow600_fixed60", "test_shallow600_fixed15", "test_shallow600_multipole_m4_a3"),
    "shallow": ("results/baseline_b_shallow/fitted", "test_shallow_no_subhalo", "test_shallow_fixed60", "test_shallow_fixed15", "test_shallow_multipole_m4_a3"),
    "clumpy": ("results/baseline_b_clumpy/fitted", "clumpy_no_subhalo", "clumpy_fixed60", "none", "clumpy_multipole_m4_a3"),
    "invcrime": ("results/baseline_b_invcrime/fitted", "invcrime_no_subhalo", "invcrime_fixed60", "none", "invcrime_multipole_m4_a3"),
    "realistic": ("results/baseline_b_realistic_full/fitted", "realistic_no_subhalo", "realistic_fixed60", "realistic_fixed15", "realistic_multipole_m4_a3"),
}


def wilson(k, n, z=1.0):
    if not n:
        return None, None, None
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def load(root, pop):
    p = ROOT / root / pop / "scan_results.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else None


def comp(recs, thr):
    pos = [r for r in recs if r["reliable_fit"] and r["has_subhalo"]]
    out = {}
    for lo, hi in BINS:
        s = [r for r in pos if lo <= r["log10_M200_true"] < hi]
        k = sum(r["delta_chi2"] >= thr for r in s); p, l, h = wilson(k, len(s))
        out[f"{lo}-{hi}"] = {"n": len(s), "k": int(k), "p": p, "lo": l, "hi": h}
    return out


def main():
    out = {}
    for arm, (root, clean, c60, c15, mp) in ARMS.items():
        cl = load(root, clean)
        if cl is None:
            continue
        rel = [r for r in cl if r["reliable_fit"]]
        thr = float(np.quantile([r["delta_chi2"] for r in rel], 0.90))
        d = {"threshold": thr, "n_clean": len(cl), "n_clean_gated": len(cl) - len(rel)}
        for key, pop in (("c60", c60), ("c15", c15)):
            R = load(root, pop)
            if R:
                d[f"completeness_{key}"] = comp(R, thr); d[f"n_gated_{key}"] = sum(not r["reliable_fit"] for r in R)
        M = load(root, mp)
        if M:
            mr = [r for r in M if r["reliable_fit"]]
            d["fpr_multipole"] = float(np.mean([r["delta_chi2"] >= thr for r in mr])); d["n_multipole"] = len(mr)
        out[arm] = d
    # line-of-sight: calibrated on the Tier-0 clean control of the full-population run, as Family A's LOS arms are
    ns = [r for r in (json.loads(l) for l in open(ROOT / "results/baseline_b_full/fitted/no_subhalo/scan_results.jsonl")) if r["reliable_fit"]]
    thr0 = float(np.quantile([r["delta_chi2"] for r in ns], 0.90))
    for z, pop in (("z025", "los_halo_z025"), ("z075", "los_halo_z075")):
        R = load("results/baseline_b_los/fitted", pop)
        if R:
            rr = [r for r in R if r["reliable_fit"]]
            out[f"los_{z}"] = {"threshold": thr0, "n": len(rr), "flag_rate": float(np.mean([r["delta_chi2"] >= thr0 for r in rr]))}
    # decoys injected into the realistic clean population, at the realistic arm's own calibration
    if "realistic" in out:
        t = out["realistic"]["threshold"]; dd = {}
        for pop in ("realistic_decoy_gaussian_s42_a10", "realistic_decoy_gaussian_s43_a10", "realistic_decoy_dipole_s42_a10"):
            R = load("results/baseline_b_realistic_full/fitted", pop)
            if R:
                rr = [r for r in R if r["reliable_fit"]]; dd[pop] = float(np.mean([r["delta_chi2"] >= t for r in rr]))
        out["realistic"]["decoys"] = dd
    (ROOT / "results/family_b_arms.json").write_text(json.dumps(out, indent=2))
    for arm, d in out.items():
        if "completeness_c60" in d:
            f = lambda c: " ".join("--" if v["p"] is None else f"{100*v['p']:.0f}" for v in c.values())  # noqa: E731
            print(f"{arm:11s} thr {d['threshold']:.3g}  c60 {f(d['completeness_c60'])}  c15 {f(d.get('completeness_c15', {}))}  "
                  f"mp {100*d.get('fpr_multipole', np.nan):.0f}%  gated clean {d['n_clean_gated']}/{d['n_clean']} c60 {d.get('n_gated_c60')}")
        else:
            print(f"{arm:11s} flagged {100*d['flag_rate']:.0f}% of {d['n']}")


if __name__ == "__main__":
    main()
