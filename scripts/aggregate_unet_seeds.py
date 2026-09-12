"""Aggregate any set of U-Net evaluation runs (results/<run>/results.json) into
mean +/- std -- a parameterized version of aggregate_seeds.py, written so the
30,000-image scale-up (self-review fix, 2026-09-11) can be summarized with the
exact same statistics as the original 8,000-image runs.

    source ~/myenv/bin/activate
    python scripts/aggregate_unet_seeds.py --runs detector_30k_seed0 detector_30k_seed1 --out results/aggregate_30k

Writes <out>/summary.json and summary.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MASS_BINS = [(8.0, 8.5), (8.5, 9.0), (9.0, 9.5), (9.5, 10.0), (10.0, 10.5), (10.5, 11.0)]


def stats(vals):
    v = np.asarray(vals, dtype=float)
    return {"values": v.tolist(), "mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    runs = [json.loads((ROOT / "results" / r / "results.json").read_text()) for r in args.runs]
    out = ROOT / args.out if not args.out.is_absolute() else args.out
    out.mkdir(parents=True, exist_ok=True)

    conds = list(runs[0]["confounder_fpr"].keys())
    fpr = {c: stats([r["confounder_fpr"][c] for r in runs]) for c in conds}
    comp60 = {f"{lo}-{hi}": stats([r["completeness_c60_by_mass_bin"][f"{lo}-{hi}"]["completeness"] for r in runs]) for lo, hi in MASS_BINS}
    comp15 = {f"{lo}-{hi}": stats([r["completeness_c15_by_mass_bin"][f"{lo}-{hi}"]["completeness"] for r in runs]) for lo, hi in MASS_BINS}
    flips = np.array([[r["paired_concentration_flip"]["detected_at_c60_not_c15"], r["paired_concentration_flip"]["detected_at_c15_not_c60"]] for r in runs], dtype=float)
    summary = {
        "n_seeds": len(runs), "seed_runs": args.runs,
        "auc": stats([r["roc_auc_fixed60_vs_no_subhalo"] for r in runs]),
        "confounder_fpr": fpr, "completeness_c60": comp60, "completeness_c15": comp15,
        "paired_flip": {"c60_not_c15": stats(flips[:, 0]), "c15_not_c60": stats(flips[:, 1]),
                        "ratio_of_means": float(flips[:, 0].mean() / max(flips[:, 1].mean(), 1e-9))},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    lines = [f"# U-Net aggregate over {len(runs)} runs: {', '.join(args.runs)}\n",
             f"AUC: {summary['auc']['mean']:.3f} ± {summary['auc']['std']:.3f}\n",
             "| mass bin | c=60 | c=15 |", "|---|---|---|"]
    for k in comp60:
        lines.append(f"| {k} | {100*comp60[k]['mean']:.1f}% ± {100*comp60[k]['std']:.1f}% | {100*comp15[k]['mean']:.1f}% ± {100*comp15[k]['std']:.1f}% |")
    lines += ["", "| confounder | FPR |", "|---|---|"] + [f"| {c} | {100*fpr[c]['mean']:.1f}% ± {100*fpr[c]['std']:.1f}% |" for c in conds]
    pf = summary["paired_flip"]
    lines += ["", f"paired flip c60→missed at c15: {pf['c60_not_c15']['mean']:.1f} ± {pf['c60_not_c15']['std']:.1f} vs reverse {pf['c15_not_c60']['mean']:.1f} ± {pf['c15_not_c60']['std']:.1f} (ratio {pf['ratio_of_means']:.1f}:1)"]
    (out / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwrote {out}/summary.json + summary.md")


if __name__ == "__main__":
    main()
