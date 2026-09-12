"""Aggregate the 4 independent training seeds into mean +/- std for RQ2 and RQ4.

    source ~/myenv/bin/activate
    python scripts/aggregate_seeds.py

n=4 is a pilot, not a powered study -- treat the std as indicative, not a
formal confidence interval. Writes results/aggregate/{summary.json,summary.md}
and a plot with error bars for each figure in first_results.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUNS = ["detector_v0_best", "detector_v1_seed1", "detector_v2_seed2", "detector_v3_seed3"]
MASS_BINS = [(8.0, 8.5), (8.5, 9.0), (9.0, 9.5), (9.5, 10.0), (10.0, 10.5), (10.5, 11.0)]


def load(name):
    return json.loads((Path("results") / name / "results.json").read_text())


def main():
    runs = [load(r) for r in RUNS]
    out = Path("results/aggregate")
    out.mkdir(parents=True, exist_ok=True)

    # ---- RQ2: confounder FPR across seeds ----
    conds = list(runs[0]["confounder_fpr"].keys())
    fpr_stats = {}
    for c in conds:
        vals = np.array([r["confounder_fpr"][c] for r in runs])
        fpr_stats[c] = {"values": vals.tolist(), "mean": float(vals.mean()), "std": float(vals.std(ddof=1))}

    baseline = fpr_stats[conds[0]]  # "no_subhalo (calibration set)" -- always exactly 0.10 by construction
    resolved = {}
    for c in conds[1:]:
        diff = fpr_stats[c]["mean"] - baseline["mean"]
        pooled_std = np.sqrt(fpr_stats[c]["std"] ** 2 + baseline["std"] ** 2)
        # baseline std is 0 (FPR on the calibration set is exactly 0.10 every time by
        # construction of the threshold), so this reduces to the confounder's own std.
        z = diff / fpr_stats[c]["std"] if fpr_stats[c]["std"] > 0 else float("inf")
        resolved[c] = {"mean_diff_from_baseline": diff, "seed_std": fpr_stats[c]["std"], "z_like": z, "resolved_at_2sigma": abs(z) >= 2}

    # ---- RQ4: completeness by mass bin, c=60 vs c=15 ----
    comp60_stats, comp15_stats = {}, {}
    for lo, hi in MASS_BINS:
        k = f"{lo}-{hi}"
        v60 = np.array([r["completeness_c60_by_mass_bin"][k]["completeness"] for r in runs])
        v15 = np.array([r["completeness_c15_by_mass_bin"][k]["completeness"] for r in runs])
        comp60_stats[k] = {"mean": float(v60.mean()), "std": float(v60.std(ddof=1)), "values": v60.tolist()}
        comp15_stats[k] = {"mean": float(v15.mean()), "std": float(v15.std(ddof=1)), "values": v15.tolist()}

    # ---- paired flip counts across seeds ----
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
        "confounder_fpr": fpr_stats, "confounder_resolved_vs_baseline": resolved,
        "completeness_c60": comp60_stats, "completeness_c15": comp15_stats,
        "paired_flip": flip_stats,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    # ---- plot: completeness vs mass, with seed error bars ----
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
    ax.set_title(f"RQ4 — completeness vs. concentration (n={len(runs)} seeds, mean ± std)")
    ax.legend(fontsize=8); ax.set_ylim(-0.02, 1.02); fig.tight_layout()
    fig.savefig(out / "completeness_vs_mass_aggregate.png", dpi=140); plt.close(fig)

    # ---- plot: confounder FPR with seed error bars ----
    fig, ax = plt.subplots(figsize=(6.2, 4))
    names = conds
    means = [fpr_stats[c]["mean"] for c in names]
    stds = [fpr_stats[c]["std"] for c in names]
    colors = ["#8e93a3", "#ffb86b", "#ff6fb0"]
    ax.bar(range(len(names)), means, yerr=stds, capsize=4, color=colors)
    ax.axhline(0.10, color="k", ls=":", lw=1, label="nominal 10% operating point")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("false positive rate"); ax.set_title(f"RQ2 — confounder FPR (n={len(runs)} seeds, mean ± std)")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(out / "confounder_fpr_aggregate.png", dpi=140); plt.close(fig)

    lines = [f"# Multi-seed aggregate (n={len(runs)} independent training runs)\n",
              f"AUC across seeds: {auc_vals.mean():.3f} ± {auc_vals.std(ddof=1):.3f}  (values: {[round(v,3) for v in auc_vals]})\n",
              "## RQ2 — confounder FPR, mean ± std across seeds\n", "| condition | mean FPR | std | resolved at 2σ from baseline? |", "|---|---|---|---|"]
    for c in conds:
        r = resolved.get(c)
        res_str = "—" if r is None else ("**yes**" if r["resolved_at_2sigma"] else "no")
        lines.append(f"| {c} | {fpr_stats[c]['mean']:.1%} | {fpr_stats[c]['std']:.1%} | {res_str} |")
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
