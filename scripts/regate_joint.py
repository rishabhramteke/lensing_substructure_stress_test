"""Re-gate the joint scan's macro-fit failures from the fits already on disk (referee round 19, 2026-09-18).

The joint scan gated a lens as a macro-fit failure when its FIRST smooth fit had chi^2/dof >= 10, and the
paper then bracketed the choice ("retained" vs "excluded as misses"). But every record also stores two later
fits: the best smooth chi^2 over the 24 basin re-optimisations (`chi2_smooth_basin_per_dof`) and, through
`delta_chi2`, the best joint chi^2. A lens whose smooth model fits from one of those starts was an optimizer
failure, not a macro-fit failure; a lens whose JOINT model fits while no smooth start does is a lens the
smooth model cannot describe and the perturber can, i.e. the strongest possible detection, not a failure.

New gate:  reliable_fit = (best smooth chi^2/dof < 10) or (best joint chi^2/dof < 10).
The original flag is kept as `reliable_fit_initial`; `gate_rescued_by` says which fit rescued a lens. Records
whose neither model fits stay excluded. Idempotent: everything is derived from stored fields, so running it
twice changes nothing. Frozen-scan roots have no basin/joint fields and are left alone.

    python scripts/regate_joint.py            # every results/baseline_a_joint_*/<pop>/scan_results.jsonl
    python scripts/regate_joint.py --roots results/baseline_a_joint_full --report
"""
import argparse, json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GATE = 10.0


def regate_file(path, report=False):
    recs = [json.loads(l) for l in open(path)]
    if not recs or "chi2_smooth_basin_per_dof" not in recs[0]:
        return None
    npix = None
    for r in recs:
        # best joint chi^2 = best smooth chi^2 - delta_chi2 (delta_chi2 is measured against the best smooth start)
        if npix is None:
            npix = 4096 if r.get("n_pix") is None else r["n_pix"]
        smooth_best = r["chi2_smooth_basin_per_dof"]
        joint_best = smooth_best - r["delta_chi2"] / npix
        r["chi2_joint_best_per_dof"] = float(joint_best)
        init = r.get("reliable_fit_initial", (r["chi2_smooth_per_dof"] < GATE))
        r["reliable_fit_initial"] = bool(init)
        new = bool(smooth_best < GATE or joint_best < GATE)
        r["gate_rescued_by"] = None if init or not new else ("basin" if smooth_best < GATE else "joint")
        r["reliable_fit"] = new
    with open(path, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    man = path.parent / "manifest.json"
    if man.exists():
        m = json.loads(man.read_text())
        m["n_unreliable_fits_initial"] = sum(1 for r in recs if not r["reliable_fit_initial"])
        m["n_unreliable_fits"] = sum(1 for r in recs if not r["reliable_fit"])
        m["gate"] = f"reliable if best smooth chi2/dof < {GATE:g} over all starts (basin control) OR best joint chi2/dof < {GATE:g}; regate_joint.py 2026-09-18"
        man.write_text(json.dumps(m, indent=2))
    out = {"n": len(recs), "unreliable_initial": sum(1 for r in recs if not r["reliable_fit_initial"]),
           "unreliable_now": sum(1 for r in recs if not r["reliable_fit"]),
           "rescued_basin": sum(1 for r in recs if r["gate_rescued_by"] == "basin"),
           "rescued_joint": sum(1 for r in recs if r["gate_rescued_by"] == "joint")}
    if report:
        sub = [r for r in recs if r["has_subhalo"] and not r["reliable_fit"]]
        if sub:
            m = np.array([r["log10_M200_true"] for r in sub])
            out["still_excluded_subhalo_mass_bins"] = {f"{lo}-{lo+0.5}": int(((m >= lo) & (m < lo + 0.5)).sum()) for lo in (8.0, 8.5, 9.0, 9.5, 10.0, 10.5)}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--roots", nargs="*", default=None)
    p.add_argument("--report", action="store_true")
    a = p.parse_args()
    roots = [ROOT / r for r in a.roots] if a.roots else sorted((ROOT / "results").glob("baseline_a_joint_*"))
    summary = {}
    for root in roots:
        if not root.is_dir():
            continue
        for f in sorted(root.glob("*/scan_results.jsonl")):
            # a population without a manifest is still being written by its scan: rewriting it would interleave the two
            # writers and corrupt the file (2026-09-30), so it is skipped until the run finishes
            if not (f.parent / "manifest.json").exists():
                print(f"{str(f.parent.relative_to(ROOT)):62s} running, skipped")
                continue
            r = regate_file(f, a.report)
            if r:
                summary[str(f.parent.relative_to(ROOT))] = r
                print(f"{str(f.parent.relative_to(ROOT)):62s} n={r['n']:4d} gated {r['unreliable_initial']:3d} -> {r['unreliable_now']:3d}  (basin {r['rescued_basin']}, joint {r['rescued_joint']})" + (f"  still excluded subhalo bins {r.get('still_excluded_subhalo_mass_bins')}" if a.report and r.get('still_excluded_subhalo_mass_bins') else ""))
    (ROOT / "results/regate_joint.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
