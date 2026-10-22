"""Expected true and false subhalo detections per lens at the Euclid-like operating point (Sect. 5.2).

True detections: the joint scan's completeness per 0.5-dex bin on the Euclid-like c=60 and c=15 populations, weighted by
the number of subhalos a lens carries in the placement annulus under dN/dM ~ M^-1.9, normalized as in the sub-threshold
population arm (45 subhalos of 1e7-10^8.5 Msun within 1.5 theta_E), counted from 1e9 Msun, below which the
Euclid-like completeness is not resolved from zero. False detections: the clean null plus the rate at
which a lens carrying a 1% m=4 multipole, the amplitude measured in real lenses, is flagged.

    python scripts/evaluate_survey_purity.py      -> results/survey_purity.json
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EU = ROOT / "results/baseline_a_joint_euclid"
EDGES = np.arange(8.0, 11.01, 0.5)
SLOPE = 1.9
N_POP, POP_RANGE, POP_RADIUS = 45.0, (7.0, 8.5), 1.5        # the population arm's normalization (Sect. 4.5)
ANNULUS = (0.6, 1.3)                                         # the placement annulus, in theta_E


def n_between(lo, hi):
    """Number of subhalos with log10 M in [lo, hi) for dN/dM ~ M^-SLOPE, up to the normalization."""
    return (10 ** (-(SLOPE - 1) * lo) - 10 ** (-(SLOPE - 1) * hi)) / (SLOPE - 1)


def rows(pop):
    return [json.loads(l) for l in open(EU / pop / "scan_results.jsonl") if json.loads(l)["reliable_fit"]]


def main():
    area = (ANNULUS[1] ** 2 - ANNULUS[0] ** 2) / POP_RADIUS ** 2
    norm = N_POP / n_between(*POP_RANGE) * area
    n_bin = np.array([norm * n_between(lo, hi) for lo, hi in zip(EDGES[:-1], EDGES[1:])])
    out = {"subhalos_per_lens_in_annulus_by_bin": n_bin.tolist(), "subhalos_per_lens_above_1e8": float(n_bin.sum())}
    for T in (20, 48):
        clean = np.mean([r["delta_chi2"] >= T for r in rows("no_subhalo")])
        mp = np.mean([r["delta_chi2"] >= T for r in rows("multipole_m4_a1")])
        R = {"clean_fpr": float(clean), "multipole_1pc_fpr": float(mp), "n_clean": len(rows("no_subhalo")), "n_multipole": len(rows("multipole_m4_a1"))}
        for c in ("test_fixed60", "test_fixed15"):
            rs = [r for r in rows(c) if r["has_subhalo"]]
            m = np.array([r["log10_M200_true"] for r in rs]); det = np.array([r["delta_chi2"] >= T for r in rs])
            comp = np.array([det[(m >= lo) & (m < hi)].mean() for lo, hi in zip(EDGES[:-1], EDGES[1:])])
            true = float(np.sum((n_bin * comp)[EDGES[:-1] >= 9.0]))   # below 1e9 Msun the Euclid-like completeness is unresolved from zero
            false = float(clean + mp)
            R[c] = {"completeness": comp.tolist(), "true_per_lens": true, "false_per_lens": false, "purity": true / (true + false)}
        out[f"dchi2_{T}"] = R
    (ROOT / "results/survey_purity.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
