"""Misspecification tolerance (referee round 29, M6): the joint scan's clean false-positive rate at the absolute
thresholds Delta chi^2 >= 20 and >= 48, and the median clean Delta chi^2, against the amplitude of a PSF error (the
data's PSF wider than the fitters assume by 2, 5 or 10%) and of an unmodelled source structure (two clumps of 0.5, 1, 2
or 4% of the source flux each), at three operating points (Tier 0, 600 s, Euclid-like), 200 clean lenses each.
Amplitude zero is each operating point's own clean scan.

    python scripts/evaluate_misspec_grid.py      -> results/misspec_grid.json
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
BASE = {"t0": R / "baseline_a_joint_full/no_subhalo", "600": R / "baseline_a_joint_shallow600/no_subhalo", "euclid": R / "baseline_a_joint_euclid/no_subhalo"}
GRID = {"psf": [(0.02, "psf2"), (0.05, "psf5"), (0.10, "psf10")], "clump": [(0.005, "clump0p5"), (0.01, "clump1"), (0.02, "clump2"), (0.04, "clump4")]}


def wilson(k, n, z=1.0):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def summ(path):
    f = Path(path) / "scan_results.jsonl"
    if not f.exists():
        return None
    r = [json.loads(l) for l in open(f)]
    d = np.array([x["delta_chi2"] for x in r if x["reliable_fit"]]); n = len(d)
    out = {"n": n, "n_scanned": len(r), "median_dchi2": float(np.median(d)) if n else None,
           "median_chi2dof": float(np.median([x["chi2_smooth_per_dof"] for x in r if x["reliable_fit"]])) if n else None}
    for t in (20, 48):
        k = int((d >= t).sum()); out[f"fpr_{t}"] = k / n if n else None; out[f"fpr_{t}_68"] = wilson(k, n)
    return out


def main():
    res = {}
    for b, base in BASE.items():
        res[b] = {"base": summ(base)}
        for kind, pts in GRID.items():
            res[b][kind] = []
            for amp, tag in pts:
                path = R / "baseline_a_joint_psferr200/no_subhalo" if (b, tag) == ("t0", "psf10") else R / f"baseline_a_joint_misspec/misspec_{b}_{tag}_no_subhalo"
                s = summ(path)
                if s:
                    res[b][kind].append(dict(amp=amp, **s))
    (R / "misspec_grid.json").write_text(json.dumps(res, indent=2))
    for b in res:
        print(b, "base", {k: res[b]["base"][k] for k in ("n", "fpr_20", "median_dchi2")} if res[b]["base"] else None)
        for kind in GRID:
            for x in res[b][kind]:
                print(f"   {kind} {x['amp']:.3f}: n {x['n']}/{x['n_scanned']} FPR20 {100 * x['fpr_20']:.0f}% FPR48 {100 * x['fpr_48']:.0f}% med dchi2 {x['median_dchi2']:.1f} chi2dof {x['median_chi2dof']:.2f}")


if __name__ == "__main__":
    main()
