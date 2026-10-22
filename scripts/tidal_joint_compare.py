"""Appendix A, joint-scan sentence: fixed-tau vs tidal-track completeness on the SAME 200 lenses (seed-0 subsample of the
matched populations, first 200), at Delta chi^2 >= 20 and at the core 10% calibration, for c=15 and c=30.

    python scripts/tidal_joint_compare.py
"""
import json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0, str(ROOT / "scripts"))
from paper_config import A_JOINT_FULL, A_JOINT_TIDAL, A_ABS, load_scan

idx = [int(i) for i in np.sort(np.random.default_rng(0).choice(1000, size=300, replace=False))][:200]
clean = [r for r in load_scan(A_JOINT_FULL, "no_subhalo") if r["reliable_fit"]]
thr = float(np.quantile([r["delta_chi2"] for r in clean], 0.90))
BINS = [(8.0, 8.5), (8.5, 9.0), (9.0, 9.5)]
def comp(recs, t):
    rel = [r for r in recs if r["reliable_fit"] and r["has_subhalo"]]
    return [(len([r for r in rel if lo <= r["log10_M200_true"] < hi]), round(100 * np.mean([r["delta_chi2"] >= t for r in rel if lo <= r["log10_M200_true"] < hi]))) for lo, hi in BINS]
for c, fixed_pop, tidal_pop in ((15, "test_fixed15", "test_tidal_c15"), (30, "test_fixed30", "test_tidal_c30")):
    fixed = [r for r in load_scan(A_JOINT_FULL, fixed_pop) if r["index"] in set(idx)]
    tidal = load_scan(A_JOINT_TIDAL, tidal_pop)
    tidx = set(r["index"] for r in tidal); fixed_m = [r for r in fixed if r["index"] in tidx]
    print(f"c={c}: tidal n={len(tidal)}, fixed-tau matched n={len(fixed_m)}")
    for name, t in (("abs", A_ABS), ("cal", thr)):
        print(f"   {name}: fixed tau {comp(fixed_m, t)}  ->  tidal track {comp(tidal, t)}")
