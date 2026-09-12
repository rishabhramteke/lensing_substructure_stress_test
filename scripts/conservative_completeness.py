"""Conservative completeness: excluded ("unreliable") fits counted as MISSES, not dropped.

Families A and B exclude lenses whose smooth macro fit has chi2/N >= 10 from every
denominator (metrics_definitions.md section 6). The exclusions are concentrated at high
subhalo mass (a massive perturber makes the smooth fit bad), so dropping them flatters
completeness in exactly the bins where it is highest. This script re-computes completeness
with every excluded subhalo-bearing lens counted as a miss, at each family's own 10%-FPR
threshold (calibrated on RELIABLE no_subhalo fits, as before -- the threshold is not the
issue, the denominator is).

    python scripts/conservative_completeness.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BINS = [(8.0, 8.5), (8.5, 9.0), (9.0, 9.5), (9.5, 10.0), (10.0, 10.5), (10.5, 11.0)]
RUNS = {
    "A c=15 seed 0": "results/baseline_a", "A c=15 seed 200": "results/baseline_a_seed1", "A c=15 seed 300": "results/baseline_a_seed300",
    "A c=60": "results/baseline_a_c60", "A c-M": "results/baseline_a_ccm",
    "B fitted seed 0": "results/baseline_b/fitted", "B fitted seed 200": "results/baseline_b_seed200/fitted", "B fitted seed 300": "results/baseline_b_seed300/fitted",
}


def load(p):
    return [json.loads(l) for l in open(p)] if p.exists() else None


def main():
    out = {}
    for label, rel in RUNS.items():
        root = ROOT / rel
        ns, t60, t15 = (load(root / f"{n}/scan_results.jsonl") for n in ("no_subhalo", "test_fixed60", "test_fixed15"))
        if not (ns and t60):
            continue
        thr = float(np.quantile([r["delta_chi2"] for r in ns if r["reliable_fit"]], 0.90))
        res = {"threshold": thr}
        for name, recs in (("c60", t60), ("c15", t15)):
            if not recs:
                continue
            pos = [r for r in recs if r["has_subhalo"]]
            bins = {}
            for lo, hi in BINS:
                sel = [r for r in pos if lo <= r["log10_M200_true"] < hi]
                rel_ = [r for r in sel if r["reliable_fit"]]
                det = sum(1 for r in rel_ if r["delta_chi2"] >= thr)
                bins[f"{lo}-{hi}"] = {"n_all": len(sel), "n_reliable": len(rel_), "n_excluded": len(sel) - len(rel_), "n_detected": det,
                                     "completeness_reported": det / len(rel_) if rel_ else None,
                                     "completeness_conservative": det / len(sel) if sel else None}
            res[name] = bins
            res[f"{name}_n_excluded_total"] = sum(1 for r in pos if not r["reliable_fit"])
            res[f"{name}_n_positive_total"] = len(pos)
        out[label] = res
    (ROOT / "results/conservative_completeness.json").write_text(json.dumps(out, indent=2))
    for label, res in out.items():
        b = res["c60"]
        print(f"{label:18s} thr {res['threshold']:7.2f}  excluded {res['c60_n_excluded_total']:3d}/{res['c60_n_positive_total']:3d}  " +
              "  ".join(f"{k}: {100*v['completeness_reported']:.0f}->{100*v['completeness_conservative']:.0f}% ({v['n_excluded']} excl)" if v['completeness_reported'] is not None else f"{k}: --" for k, v in b.items()))


if __name__ == "__main__":
    main()
