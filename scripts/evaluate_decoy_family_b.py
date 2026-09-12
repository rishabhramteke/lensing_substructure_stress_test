"""Family B on the non-physical decoys (referee round 5): FPR at its own 10%-FPR threshold
(from results/baseline_b/fitted/summary/results.json, lambda=1e5) on the materialised decoy
populations, in the same JSON layout as noise_decoy_control.py so the figure code can overlay it.

    python scripts/evaluate_decoy_family_b.py
-> results/noise_decoy_control_familyB/results.json
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
thr = json.loads((ROOT / "results/baseline_b/fitted/summary/results.json").read_text())["threshold_delta_chi2_at_10pct_fpr"]
out = {"config": {"threshold_peak_abs_dkappa_lam1e5": thr, "source": "results/baseline_b_decoy/fitted", "note": "same decoys as noise_decoy_control.py (seed 42)"}}
for shape, seed in (("gaussian", 42), ("dipole", 42), ("gaussian", 43)):
    key = shape if seed == 42 else f"{shape}_seed{seed}"     # 'gaussian'/'dipole' = seed 42 (as in the A/C runs); 'gaussian_seed43' = the replicate
    for amp in (3, 6, 10):
        p = ROOT / f"results/baseline_b_decoy/fitted/decoy_{shape}_s{seed}_a{amp}/scan_results.jsonl"
        if not p.exists():
            continue
        recs = [json.loads(l) for l in open(p)]
        rel = [r for r in recs if r["reliable_fit"]]
        out.setdefault(key, {})[str(float(amp))] = {"b_fpr": float(np.mean([r["delta_chi2"] >= thr for r in rel])) if rel else None,
                                                    "n_reliable": len(rel), "n": len(recs),
                                                    "median_peak": float(np.median([r["delta_chi2"] for r in rel])) if rel else None}
(ROOT / "results/noise_decoy_control_familyB").mkdir(exist_ok=True)
(ROOT / "results/noise_decoy_control_familyB/results.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=1))
