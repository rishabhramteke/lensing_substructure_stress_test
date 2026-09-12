"""mean ± std(ddof=1) across Family B seed sets (metrics_definitions.md §7 convention).

    python scripts/aggregate_family_b_seeds.py --summaries results/baseline_b/summary_family_b.json \
        results/baseline_b_seed200/summary_family_b.json --variant fitted --lam 1e5 \
        --out results/baseline_b/aggregate_fitted_2seeds.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BINS = ["8.0-8.5", "8.5-9.0", "9.0-9.5", "9.5-10.0", "10.0-10.5", "10.5-11.0"]


def ms(vals):
    v = np.array([x for x in vals if x is not None], dtype=float)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if v.size > 1 else None, "values": [float(x) for x in v], "n_seeds": int(v.size)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summaries", nargs="+", type=Path, required=True)
    p.add_argument("--variant", default="fitted")
    p.add_argument("--lam", default="1e5")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    S = [json.load(open(f))[a.variant] for f in a.summaries]
    P = [s["by_lambda"][a.lam] for s in S]
    out = {"variant": a.variant, "lam": a.lam, "n_seeds": len(S), "summaries": [str(f) for f in a.summaries],
           "threshold": ms([q["threshold"] for q in P]),
           "completeness_c60": {b: ms([q["completeness_c60"][b]["completeness"] for q in P]) for b in BINS},
           "completeness_c15": {b: ms([q["completeness_c15"][b]["completeness"] for q in P]) for b in BINS},
           "paired_flip": {"c60_not_c15": ms([q["paired_flip"]["detected_at_c60_not_c15"] for q in P]),
                           "c15_not_c60": ms([q["paired_flip"]["detected_at_c15_not_c60"] for q in P]),
                           "n_pairs": [q["paired_flip"]["n_pairs"] for q in P]},
           "confounder_fpr": {k: ms([q["confounder_fpr"][k]["fpr"] for q in P]) for k in ["multipole_m4_a1", "multipole_m4_a3"]},
           "localization_frac_within_0p16": ms([s["localization_lam1e5_10pct_fpr"].get("frac_within_2px_0p16") for s in S]),
           "localization_frac_within_0p32": ms([s["localization_lam1e5_10pct_fpr"].get("frac_within_2meshpx_0p32") for s in S]),
           "mass_error_median_dex_0p16": ms([s["localization_lam1e5_10pct_fpr"].get("mass_error_dex_localized_0p16", {}).get("median") for s in S]),
           "n_unreliable": [s["n_unreliable"] for s in S]}
    a.out.write_text(json.dumps(out, indent=2))
    f = lambda d: f"{100*d['mean']:.1f}±{100*d['std']:.1f}" if d["std"] is not None else f"{100*d['mean']:.1f}"
    print(f"{a.variant} lam={a.lam}, {len(S)} seeds")
    print("c60  " + "  ".join(f"{b}: {f(out['completeness_c60'][b])}" for b in BINS))
    print("c15  " + "  ".join(f"{b}: {f(out['completeness_c15'][b])}" for b in BINS))
    fl = out["paired_flip"]; print(f"flip c60-only {fl['c60_not_c15']['mean']:.0f}±{fl['c60_not_c15']['std']:.0f} : c15-only {fl['c15_not_c60']['mean']:.0f}±{fl['c15_not_c60']['std']:.0f}  (pairs {fl['n_pairs']})")
    print(f"confounder FPR a=0.01 {f(out['confounder_fpr']['multipole_m4_a1'])}  a=0.03 {f(out['confounder_fpr']['multipole_m4_a3'])}")
    print(f"localized <=0.16\" {f(out['localization_frac_within_0p16'])}  <=0.32\" {f(out['localization_frac_within_0p32'])}  mass err median {out['mass_error_median_dex_0p16']['mean']:+.2f}±{out['mass_error_median_dex_0p16']['std']:.2f} dex")
    print("unreliable per seed:", out["n_unreliable"])


if __name__ == "__main__":
    main()
