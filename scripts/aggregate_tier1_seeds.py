"""Aggregate the Tier-1 (COSMOS source) U-Net seeds into mean +/- std.

    source ~/myenv/bin/activate
    python scripts/aggregate_tier1_seeds.py

n=2 is a pilot (matching Family A's seed count for this same comparison, not
the Tier-0 U-Net's n=4) -- treat std as indicative. Mirrors
`aggregate_seeds.py`'s statistics exactly, pointed at the two Tier-1 runs, so
the Tier-0 and Tier-1 numbers in `first_results.md` are computed identically
and safe to put side by side.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUNS = ["detector_tier1_seed0", "detector_tier1_seed1"]
MASS_BINS = [(8.0, 8.5), (8.5, 9.0), (9.0, 9.5), (9.5, 10.0), (10.0, 10.5), (10.5, 11.0)]


def load(name):
    return json.loads((Path("results") / name / "results.json").read_text())


def main():
    runs = [load(r) for r in RUNS]
    out = Path("results/aggregate_tier1")
    out.mkdir(parents=True, exist_ok=True)

    conds = list(runs[0]["confounder_fpr"].keys())
    fpr_stats = {}
    for c in conds:
        vals = np.array([r["confounder_fpr"][c] for r in runs])
        fpr_stats[c] = {"values": vals.tolist(), "mean": float(vals.mean()), "std": float(vals.std(ddof=1))}

    comp60_stats, comp15_stats = {}, {}
    for lo, hi in MASS_BINS:
        k = f"{lo}-{hi}"
        v60 = np.array([r["completeness_c60_by_mass_bin"][k]["completeness"] for r in runs])
        v15 = np.array([r["completeness_c15_by_mass_bin"][k]["completeness"] for r in runs])
        comp60_stats[k] = {"mean": float(v60.mean()), "std": float(v60.std(ddof=1)), "values": v60.tolist()}
        comp15_stats[k] = {"mean": float(v15.mean()), "std": float(v15.std(ddof=1)), "values": v15.tolist()}

    flips = np.array([[r["paired_concentration_flip"]["detected_at_c60_not_c15"],
                        r["paired_concentration_flip"]["detected_at_c15_not_c60"]] for r in runs])
    flip_stats = {
        "c60_not_c15_mean": float(flips[:, 0].mean()), "c60_not_c15_std": float(flips[:, 0].std(ddof=1)),
        "c15_not_c60_mean": float(flips[:, 1].mean()), "c15_not_c60_std": float(flips[:, 1].std(ddof=1)),
        "ratio_of_means": float(flips[:, 0].mean() / max(flips[:, 1].mean(), 1e-9)),
        "per_seed": flips.tolist(),
    }

    auc_vals = np.array([r["roc_auc_fixed60_vs_no_subhalo"] for r in runs])

    summary = {
        "n_seeds": len(runs), "seed_runs": RUNS,
        "auc": {"values": auc_vals.tolist(), "mean": float(auc_vals.mean()), "std": float(auc_vals.std(ddof=1))},
        "confounder_fpr": fpr_stats,
        "completeness_c60": comp60_stats, "completeness_c15": comp15_stats,
        "paired_flip": flip_stats,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    mids = [(lo + hi) / 2 for lo, hi in MASS_BINS]
    m60 = [comp60_stats[f"{lo}-{hi}"]["mean"] for lo, hi in MASS_BINS]
    s60 = [comp60_stats[f"{lo}-{hi}"]["std"] for lo, hi in MASS_BINS]
    m15 = [comp15_stats[f"{lo}-{hi}"]["mean"] for lo, hi in MASS_BINS]
    s15 = [comp15_stats[f"{lo}-{hi}"]["std"] for lo, hi in MASS_BINS]
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    ax.errorbar(mids, m60, yerr=s60, fmt="o-", label="c = 60 (Tsang+2024 fiducial)", color="#1e64a8", capsize=3)
    ax.errorbar(mids, m15, yerr=s15, fmt="s--", label="c = 15 (their low-c ablation)", color="#c8177a", capsize=3)
    ax.axhline(0.10, color="gray", ls=":", lw=1, label="FPR operating point")
    ax.set_xlabel("log10(subhalo M200 / Msun)"); ax.set_ylabel("completeness @ FPR=10%")
    ax.set_title(f"Tier 1 (COSMOS source) RQ4 — n={len(runs)} seeds, mean ± std")
    ax.legend(fontsize=8); ax.set_ylim(-0.02, 1.02); fig.tight_layout()
    fig.savefig(out / "completeness_vs_mass_aggregate.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4))
    names = conds
    means = [fpr_stats[c]["mean"] for c in names]
    stds = [fpr_stats[c]["std"] for c in names]
    colors = ["#8e93a3", "#ffb86b", "#ff6fb0"]
    ax.bar(range(len(names)), means, yerr=stds, capsize=4, color=colors)
    ax.axhline(0.10, color="k", ls=":", lw=1, label="nominal 10% operating point")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("false positive rate"); ax.set_title(f"Tier 1 (COSMOS source) RQ2 — n={len(runs)} seeds, mean ± std")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(out / "confounder_fpr_aggregate.png", dpi=140); plt.close(fig)

    lines = [f"# Tier-1 (COSMOS source) multi-seed aggregate (n={len(runs)})\n",
              f"AUC across seeds: {auc_vals.mean():.3f} ± {auc_vals.std(ddof=1):.3f}  (values: {[round(v,3) for v in auc_vals]})\n",
              "## RQ2 — confounder FPR, mean ± std across seeds\n", "| condition | mean FPR | std |", "|---|---|---|"]
    for c in conds:
        lines.append(f"| {c} | {fpr_stats[c]['mean']:.1%} | {fpr_stats[c]['std']:.1%} |")
    lines += ["", "## RQ4 — completeness by mass bin, mean ± std across seeds\n",
              "| mass bin | c=60 mean±std | c=15 mean±std |", "|---|---|---|"]
    for lo, hi in MASS_BINS:
        k = f"{lo}-{hi}"
        lines.append(f"| {lo}-{hi} | {comp60_stats[k]['mean']:.1%} ± {comp60_stats[k]['std']:.1%} | {comp15_stats[k]['mean']:.1%} ± {comp15_stats[k]['std']:.1%} |")
    lines += ["", f"**Paired flip, averaged over seeds:** c60-not-c15 = {flip_stats['c60_not_c15_mean']:.1f} ± {flip_stats['c60_not_c15_std']:.1f}, "
              f"c15-not-c60 = {flip_stats['c15_not_c60_mean']:.1f} ± {flip_stats['c15_not_c60_std']:.1f}  "
              f"(ratio of means: {flip_stats['ratio_of_means']:.1f}:1)"]
    (out / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwrote {out}/summary.json, summary.md, and 2 plots")


if __name__ == "__main__":
    main()
