"""Fold the multi-start rescue pass back into the core joint-scan populations (referee round 19, 2026-09-18).

results/baseline_a_joint_rescue/<pop>/scan_results.jsonl holds re-fits (six smooth starts, then the full joint scan
and basin control) of the lenses that results/baseline_a_joint_full/<pop> had gated. Each rescued record replaces the
original by index; the original's gate-relevant numbers are kept inside the new record (`prerescue`) so the change is
traceable, and `rescued_record: true` marks it. A record is replaced only if the re-fit improved the best smooth or
joint chi^2 (it never makes a lens worse). regate_joint.py must run afterwards to recompute the gate flags.

    python scripts/merge_rescue.py && python scripts/regate_joint.py --roots results/baseline_a_joint_full --report
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FULL, RESCUE = ROOT / "results/baseline_a_joint_full", ROOT / "results/baseline_a_joint_rescue"
NPIX = 4096


def main():
    for pop_dir in sorted(RESCUE.glob("*/")):
        pop = pop_dir.name
        src, dst = pop_dir / "scan_results.jsonl", FULL / pop / "scan_results.jsonl"
        if not src.exists() or not (pop_dir / "manifest.json").exists() or not dst.exists():
            print(f"skip {pop} (rescue unfinished or no target)"); continue
        new = {r["index"]: r for r in (json.loads(l) for l in open(src))}
        recs = [json.loads(l) for l in open(dst)]
        n_rep = n_better = 0
        for k, r in enumerate(recs):
            i = r["index"]
            if i not in new:
                continue
            nr = new[i]
            old_smooth = r["chi2_smooth_basin_per_dof"]; old_joint = old_smooth - r["delta_chi2"] / NPIX
            new_smooth = nr["chi2_smooth_basin_per_dof"]; new_joint = new_smooth - nr["delta_chi2"] / NPIX
            if min(new_smooth, new_joint) < min(old_smooth, old_joint) - 1e-9:
                nr = dict(nr)
                nr["rescued_record"] = True
                nr["prerescue"] = {"chi2_smooth_per_dof": r["chi2_smooth_per_dof"], "chi2_smooth_basin_per_dof": old_smooth,
                                   "chi2_joint_best_per_dof": old_joint, "delta_chi2": r["delta_chi2"], "reliable_fit": r.get("reliable_fit_initial", r["reliable_fit"])}
                for key in ("reliable_fit_initial", "gate_rescued_by", "chi2_joint_best_per_dof"):
                    nr.pop(key, None)
                nr["reliable_fit_initial"] = False   # by construction: every rescued lens had failed the first-start gate
                recs[k] = nr; n_rep += 1
                if (new_smooth < 10) or (new_joint < 10):
                    n_better += 1
            else:
                r["rescue_attempted"] = True
        with open(dst, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        man = FULL / pop / "manifest.json"
        m = json.loads(man.read_text())
        m["rescue"] = {"n_attempted": len(new), "n_replaced": n_rep, "n_now_passing_gate": n_better, "source": str(pop_dir.relative_to(ROOT)),
                       "recipe": "six smooth starts (near-truth x1 at 15% jitter, x4 at 30% jitter, one data-only), full joint scan + basin control"}
        man.write_text(json.dumps(m, indent=2))
        print(f"{pop:14s} attempted {len(new):3d}  replaced {n_rep:3d}  now passing gate {n_better:3d}")


if __name__ == "__main__":
    main()
