"""Grid scan restricted to the free-position subsample (referee round 19): the like-for-like numbers Sect. 4.4 quotes,
the grid's localized fraction and its two lowest-bin completenesses at Delta chi^2 >= 20 on the SAME 300 lenses the
free-position scan used (seed 0 of test_fixed60), plus the free-position scan's own numbers.

    python scripts/freepos_vs_grid.py
"""
import json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0, str(ROOT / "scripts"))
from paper_config import A_JOINT_FULL, A_ABS, LOC_ARCSEC, load_scan

truth = [json.loads(l) for l in open(ROOT / "data/test_fixed60/truth.jsonl")]
idx = set(int(i) for i in np.sort(np.random.default_rng(0).choice(1000, size=300, replace=False)))
core_clean = [r for r in load_scan(A_JOINT_FULL, "no_subhalo") if r["reliable_fit"]]
thr_core = float(np.quantile([r["delta_chi2"] for r in core_clean], 0.90))
grid = [r for r in load_scan(A_JOINT_FULL, "test_fixed60") if r["index"] in idx and r["reliable_fit"] and r["has_subhalo"]]
fp_clean = [r for r in load_scan("results/baseline_a_joint_freepos", "no_subhalo") if r["reliable_fit"]]
thr_fp = float(np.quantile([r["delta_chi2"] for r in fp_clean], 0.90))
free = [r for r in load_scan("results/baseline_a_joint_freepos", "test_fixed60") if r["reliable_fit"] and r["has_subhalo"]]
def loc(recs, thr):
    det = [r for r in recs if r["delta_chi2"] >= thr]
    off = np.array([np.hypot(r["best_x"] - truth[r["index"]]["subhalo"]["x"], r["best_y"] - truth[r["index"]]["subhalo"]["y"]) for r in det])
    return len(det), float((off < LOC_ARCSEC).mean()) if len(off) else None, float(np.median(off)) if len(off) else None, float((off < 0.08).mean()) if len(off) else None
def comp(recs, thr, lo, hi):
    sel = [r for r in recs if lo <= r["log10_M200_true"] < hi]
    return len(sel), float(np.mean([r["delta_chi2"] >= thr for r in sel])) if sel else None
print(f"core clean threshold {thr_core:.2f}; free-position clean threshold {thr_fp:.2f}; grid subsample n={len(grid)} (of {len(idx)}), free n={len(free)}")
for name, recs, thr in (("grid (same 300 lenses)", grid, thr_core), ("free position", free, thr_fp)):
    n, f, med, f1 = loc(recs, thr); print(f"{name:24s} cal: det {n} localized {100*f:.1f}% median offset {med:.3f} within 1 px {100*f1:.0f}%")
    n, f, med, f1 = loc(recs, A_ABS); print(f"{'':24s} abs: det {n} localized {100*f:.1f}% median offset {med:.3f}")
    print(f"{'':24s} compl at >=20, bins 8-8.5 / 8.5-9:", comp(recs, A_ABS, 8.0, 8.5), comp(recs, A_ABS, 8.5, 9.0))
mp = [r for r in load_scan("results/baseline_a_joint_freepos", "multipole_m4_a3") if r["reliable_fit"]]
print("free-position multipole a3 FPR: cal", round(100 * np.mean([r["delta_chi2"] >= thr_fp for r in mp]), 1), "abs", round(100 * np.mean([r["delta_chi2"] >= A_ABS for r in mp]), 1), "n", len(mp))
print("free-position clean: max dchi2", round(max(r["delta_chi2"] for r in fp_clean), 1), "frac >= 20", round(100 * np.mean([r["delta_chi2"] >= A_ABS for r in fp_clean]), 2), "%")
det_mp = [r for r in mp if r["delta_chi2"] >= thr_fp]
moved = np.array([np.hypot(r["best_x"] - min(r["cells"], key=lambda c: c["chi2_joint"])["node_x"], r["best_y"] - min(r["cells"], key=lambda c: c["chi2_joint"])["node_y"]) for r in det_mp])
print("multipole false alarms: median displacement from node", round(float(np.median(moved)), 2), "frac at box edge (>=0.49)", round(float(np.mean(moved >= 0.49)), 2))
