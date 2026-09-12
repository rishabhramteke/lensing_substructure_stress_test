"""Re-weight already-computed per-mass-bin completeness by the real CDM
subhalo mass function, instead of the flat/uniform mix our test populations
happen to contain.

Motivation (2026-09-11, user question): "is the subhalo mass distribution a
proven fact or a simulation assumption -- can we stress-test it?" Every
completeness number reported so far (RQ4, RQ1) is a per-mass-bin rate; the
population's own mass prior is uniform-in-log-mass (config.py's
`log10_mass_range`, matching Tsang+2024's own convention) purely so every
bin gets enough test images -- it is not meant to represent how many
subhalos of each mass actually exist in nature. Real CDM subhalo populations
follow dN/dM ~ M^-alpha, alpha~1.9 (Dhanasingham+2025's own prior, cited in
fulltext_findings.md), heavily dominated by the smallest masses. This
recombines the SAME per-bin completeness numbers already on disk with that
realistic weighting -- no new simulation or detector training needed -- to
ask "what completeness would a real subhalo population actually see."

    source ~/myenv/bin/activate
    python scripts/reweight_by_mass_function.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BINS = [(8.0, 8.5), (8.5, 9.0), (9.0, 9.5), (9.5, 10.0), (10.0, 10.5), (10.5, 11.0)]
ALPHA = 1.9  # dN/dM ~ M^-alpha


def bin_weight(lo, hi):
    m_lo, m_hi = 10 ** lo, 10 ** hi
    return (m_hi ** (1 - ALPHA) - m_lo ** (1 - ALPHA)) / (1 - ALPHA)


def weighted_and_naive(comp_dict, weights):
    vals = np.array([comp_dict[f"{lo}-{hi}"]["mean"] for lo, hi in BINS])
    return float((vals * weights).sum()), float(vals.mean())


def main():
    weights = np.array([bin_weight(lo, hi) for lo, hi in BINS])
    weights = weights / weights.sum()

    d_unet = json.loads((ROOT / "results/aggregate/summary.json").read_text())
    d_a = json.loads((ROOT / "results/baseline_a/aggregate/summary.json").read_text())

    rows = []
    entries = [
        ("U-Net (Family C), c=60", d_unet["completeness_c60"]),
        ("U-Net (Family C), c=15", d_unet["completeness_c15"]),
        ("Family A, c=60", d_a["completeness_c60"]),
        ("Family A, c=15", d_a["completeness_c15"]),
    ]
    b_agg = next((ROOT / f"results/baseline_b/aggregate_fitted_{k}seeds.json" for k in (4, 3, 2) if (ROOT / f"results/baseline_b/aggregate_fitted_{k}seeds.json").exists()), None)
    if b_agg is not None:   # Family B (fitted variant), added 2026-09-12 -- pure arithmetic on numbers already in hand
        d_b = json.loads(b_agg.read_text())
        entries += [("Family B, c=60", d_b["completeness_c60"]), ("Family B, c=15", d_b["completeness_c15"])]
    for label, comp in entries:
        w, u = weighted_and_naive(comp, weights)
        rows.append({"label": label, "naive_unweighted": u, "cdm_mass_function_weighted": w})

    out = {
        "alpha": ALPHA,
        "bin_weights": {f"{lo}-{hi}": float(w) for (lo, hi), w in zip(BINS, weights)},
        "results": rows,
    }
    out_dir = ROOT / "results/mass_function_reweighting"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(out, indent=2))

    print("CDM mass-function bin weights (dN/dM ~ M^-1.9):")
    for (lo, hi), w in zip(BINS, weights):
        print(f"  {lo}-{hi}: {w:.1%}")
    print()
    for row in rows:
        print(f"{row['label']}: naive unweighted avg = {row['naive_unweighted']:.1%}  "
              f"->  CDM-mass-function-weighted = {row['cdm_mass_function_weighted']:.1%}")
    print(f"\nwrote {out_dir}/results.json")


if __name__ == "__main__":
    main()
