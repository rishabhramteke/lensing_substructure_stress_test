"""Family B (fixed-source variant) on Tier 1 (COSMOS sources): threshold at 10% FPR on the Tier-1
clean population, multipole FPR, completeness by mass bin; gated and gate-lifted (the single-Sersic
source is misspecified on Tier 1, so most fits exceed chi2/N = 10, exactly as for Family A there).

    python scripts/evaluate_family_b_tier1.py -> results/baseline_b_tier1/summary.json
"""
import json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results/baseline_b_tier1/fitted"
BINS = [(8.0, 8.5), (8.5, 9.0), (9.0, 9.5), (9.5, 10.0), (10.0, 10.5), (10.5, 11.0)]
def load(pop):
    p = R / pop / "scan_results.jsonl"; return [json.loads(l) for l in open(p)] if p.exists() else None
ns, t60, mp = load("no_subhalo"), load("test_fixed60"), load("multipole_m4_a3")
out = {}
for gate, label in ((True, "gated"), (False, "lifted")):
    sel = (lambda recs: [r for r in recs if r["reliable_fit"]]) if gate else (lambda recs: list(recs))
    thr = float(np.quantile([r["delta_chi2"] for r in sel(ns)], 0.90))
    o = {"threshold": thr, "n_clean_used": len(sel(ns)), "n_unreliable_clean": sum(1 for r in ns if not r["reliable_fit"]),
         "median_chi2_per_dof_clean": float(np.median([r["chi2_smooth_per_dof"] for r in ns]))}
    if mp: o["fpr_multipole_a3"] = float(np.mean([r["delta_chi2"] >= thr for r in sel(mp)]))
    if t60:
        pos = [r for r in sel(t60) if r["has_subhalo"]]
        o["completeness_c60"] = {f"{lo}-{hi}": {"n": len([r for r in pos if lo <= r["log10_M200_true"] < hi]),
                                                 "completeness": (float(np.mean([r["delta_chi2"] >= thr for r in pos if lo <= r["log10_M200_true"] < hi])) if any(lo <= r["log10_M200_true"] < hi for r in pos) else None)} for lo, hi in BINS}
        det = [r for r in pos if r["delta_chi2"] >= thr]
        o["auc_like_frac_detected"] = len(det) / max(1, len(pos))
    out[label] = o
(ROOT / "results/baseline_b_tier1/summary.json").write_text(json.dumps(out, indent=2)); print(json.dumps(out, indent=1))
