"""What the 900 s wall-clock guard did to every joint-scan number (referee round 20).

complete_truncated_joint.py finished every lens the guard had cut short and kept the pre-completion statistic under
`completion.before`. For each population this reports how many lenses were truncated, how many decisions at the
population's own 10% calibration and at Delta chi^2 >= 20 changed, and the flagged fraction before and after.
The calibration threshold is recomputed from the clean control of the same arm before and after. Writes
results/guard_completion.json.

    python scripts/evaluate_guard_completion.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ABS = 20.0
CLEAN = {"lenslight_no_subhalo": "lenslight_no_subhalo"}


def before_delta(r):
    return r["completion"]["before"]["delta_chi2"] if r.get("completion") else r["delta_chi2"]


def main():
    out, tot = {}, {"n": 0, "n_truncated": 0, "n_completed": 0, "changed_abs": 0, "changed_cal": 0}
    for root in sorted((ROOT / "results").glob("baseline_a_joint_*")):
        if root.name in ("baseline_a_joint_rescue", "baseline_a_joint_basin_c15", "baseline_a_joint_c15", "baseline_a_joint_null_c15", "baseline_a_joint_blind"):
            continue
        pops = {p.name: [json.loads(l) for l in open(p / "scan_results.jsonl")] for p in sorted(root.glob("*/")) if (p / "scan_results.jsonl").exists()}
        clean_key = next((k for k in pops if k.endswith("no_subhalo")), None)
        thr = {}
        if clean_key:
            rel = [r for r in pops[clean_key] if r["reliable_fit"]]
            thr = {"after": float(np.quantile([r["delta_chi2"] for r in rel], 0.9)),
                   "before": float(np.quantile([before_delta(r) for r in rel], 0.9))}
        for pop, recs in pops.items():
            rel = [r for r in recs if r["reliable_fit"]]
            trunc = [r for r in recs if r.get("truncated") or r.get("truncated_by_guard")]
            comp = [r for r in recs if r.get("completion")]
            d = {"n": len(recs), "n_truncated": len(trunc), "n_completed": len(comp), "n_still_truncated": sum(1 for r in recs if r.get("truncated")),
                 "flag_abs_before": float(np.mean([before_delta(r) >= ABS for r in rel])) if rel else None,
                 "flag_abs_after": float(np.mean([r["delta_chi2"] >= ABS for r in rel])) if rel else None,
                 "changed_abs": sum(1 for r in comp if r["reliable_fit"] and (r["delta_chi2"] >= ABS) != (before_delta(r) >= ABS))}
            if thr:
                d["threshold_before"], d["threshold_after"] = thr["before"], thr["after"]
                d["flag_cal_before"] = float(np.mean([before_delta(r) >= thr["before"] for r in rel])) if rel else None
                d["flag_cal_after"] = float(np.mean([r["delta_chi2"] >= thr["after"] for r in rel])) if rel else None
                d["changed_cal"] = sum(1 for r in rel if (r["delta_chi2"] >= thr["after"]) != (before_delta(r) >= thr["before"]))
            out[f"{root.name}/{pop}"] = d
            for k in ("n", "n_truncated", "n_completed", "changed_abs"):
                tot[k] += d[k]
            tot["changed_cal"] += d.get("changed_cal", 0)
    out["_total"] = tot
    (ROOT / "results/guard_completion.json").write_text(json.dumps(out, indent=1))
    for k, d in out.items():
        if k == "_total" or not d["n_truncated"]:
            continue
        cal = f"  cal {100*d['flag_cal_before']:.1f}->{100*d['flag_cal_after']:.1f}% (thr {d['threshold_before']:.2f}->{d['threshold_after']:.2f}, {d['changed_cal']} changed)" if "flag_cal_before" in d and d["flag_cal_before"] is not None else ""
        print(f"{k:55s} trunc {d['n_truncated']:3d}/{d['n']:4d} done {d['n_completed']:3d}  >=20 {100*d['flag_abs_before']:.1f}->{100*d['flag_abs_after']:.1f}% ({d['changed_abs']} changed){cal}")
    print("TOTAL", tot)


if __name__ == "__main__":
    main()
