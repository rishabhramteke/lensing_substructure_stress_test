"""Family-B-specific summary on top of the shared evaluators.

The shared protocol numbers (10%-FPR threshold on `no_subhalo`, completeness per 0.5-dex
bin at c=60/c=15, paired flip, confounder FPR) come from `evaluate_baseline_a.py --root
results/baseline_b/<variant>` unchanged. This script adds what the shared evaluators do
not know about Family B:

  * the regularization sensitivity band (same protocol re-run on the lam=1e4 and lam=1e6
    scores stored alongside the primary lam=1e5 score),
  * localization at BOTH the 2-data-pixel criterion (0.16", the paper's) and the
    2-mesh-pixel criterion (0.32", the natural resolution of a factor-2 dpsi mesh),
  * aperture-mass error among localized detections (the sign of the raw proxy included),
  * a macro-fit-error diagnostic for the `fitted` variant: how much of the null
    distribution of max|dkappa| is driven by the smooth-fit residual rather than noise
    (Spearman rank correlation with chi2_smooth/N, and the null shift relative to `oracle`),
  * published-style absolute operating points: the 99th percentile and the maximum of the
    null scores (~1% FPR and "no null exceedance" at n~300 -- 0.1% is not resolvable at n=300
    and is not claimed),
  * a side-by-side with Family A (results/baseline_a/summary) and the U-Net 4-seed aggregate
    (results/aggregate/summary.json) on the identical images and protocol.

    python scripts/summarize_family_b.py --root results/baseline_b --out results/baseline_b/summary_family_b.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
BINS = [(8.0, 8.5), (8.5, 9.0), (9.0, 9.5), (9.5, 10.0), (10.0, 10.5), (10.5, 11.0)]
POPS = ["test_fixed60", "test_fixed15", "no_subhalo", "multipole_m4_a1", "multipole_m4_a3"]
LAM_KEYS = {"1e4": "peak_abs_dkappa_lam1e4", "1e5": "peak_abs_dkappa", "1e6": "peak_abs_dkappa_lam1e6"}


def load(root, variant):
    out = {}
    for pop in POPS:
        f = root / variant / pop / "scan_results.jsonl"
        out[pop] = [json.loads(l) for l in open(f)] if f.exists() else []
    return out


def completeness(recs, key, thr):
    res = {}
    for lo, hi in BINS:
        sel = [r for r in recs if r["reliable_fit"] and r["has_subhalo"] and lo <= r["log10_M200_true"] < hi]
        res[f"{lo}-{hi}"] = {"n": len(sel), "completeness": (float(np.mean([r[key] >= thr for r in sel])) if sel else None)}
    return res


def protocol(recs_by_pop, key):
    null = np.array([r[key] for r in recs_by_pop["no_subhalo"] if r["reliable_fit"]])
    thr = float(np.percentile(null, 90))
    r60 = {r["index"]: r for r in recs_by_pop["test_fixed60"] if r["reliable_fit"] and r["has_subhalo"]}
    r15 = {r["index"]: r for r in recs_by_pop["test_fixed15"] if r["reliable_fit"] and r["has_subhalo"]}
    common = sorted(set(r60) & set(r15))
    d60 = np.array([r60[i][key] >= thr for i in common]); d15 = np.array([r15[i][key] >= thr for i in common])
    fpr = {}
    for pop in ["no_subhalo", "multipole_m4_a1", "multipole_m4_a3"]:
        rel = [r for r in recs_by_pop[pop] if r["reliable_fit"]]
        fpr[pop] = {"n_reliable": len(rel), "fpr": float(np.mean([r[key] >= thr for r in rel])) if rel else None}
    return {"threshold": thr, "n_null": int(null.size),
            "completeness_c60": completeness(recs_by_pop["test_fixed60"], key, thr),
            "completeness_c15": completeness(recs_by_pop["test_fixed15"], key, thr),
            "paired_flip": {"n_pairs": len(common), "detected_at_c60": int(d60.sum()), "detected_at_c15": int(d15.sum()),
                            "detected_at_c60_not_c15": int((d60 & ~d15).sum()), "detected_at_c15_not_c60": int((d15 & ~d60).sum())},
            "confounder_fpr": fpr,
            "null_percentiles": {"p50": float(np.percentile(null, 50)), "p90": thr, "p99": float(np.percentile(null, 99)), "max": float(null.max())}}


def localization(recs_by_pop, truths60, thr, key="peak_abs_dkappa"):
    det = [r for r in recs_by_pop["test_fixed60"] if r["reliable_fit"] and r["has_subhalo"] and r[key] >= thr and r.get("best_x") is not None]
    if not det:
        return {"n_detected": 0}
    off = np.array([np.hypot(r["best_x"] - truths60[r["index"]]["subhalo"]["x"], r["best_y"] - truths60[r["index"]]["subhalo"]["y"]) for r in det])
    true_m = np.array([truths60[r["index"]]["subhalo"]["log10_M200"] for r in det])
    proxy = np.array([r["aperture_mass_proxy_Msun"] for r in det])
    out = {"n_detected": len(det), "median_offset_arcsec": float(np.median(off)),
           "frac_within_2px_0p16": float(np.mean(off <= 0.16)), "n_within_2px_0p16": int((off <= 0.16).sum()),
           "frac_within_2meshpx_0p32": float(np.mean(off <= 0.32)), "n_within_2meshpx_0p32": int((off <= 0.32).sum()),
           "frac_aperture_mass_negative_among_detected": float(np.mean(proxy <= 0))}
    for tag, lim in [("0p16", 0.16), ("0p32", 0.32)]:
        loc = off <= lim
        if loc.sum() and (proxy[loc] > 0).any():
            ok = loc & (proxy > 0)
            err = np.log10(proxy[ok]) - true_m[ok]
            out[f"mass_error_dex_localized_{tag}"] = {"n": int(ok.sum()), "n_negative_proxy_excluded": int((loc & (proxy <= 0)).sum()),
                                                      "median": float(np.median(err)), "mad": float(1.4826 * np.median(np.abs(err - np.median(err)))),
                                                      "min": float(err.min()), "max": float(err.max()),
                                                      "n_within_0p5dex": int((np.abs(err) <= 0.5).sum()), "n_underestimated": int((err < 0).sum())}
    return out


def fit_error_diagnostic(fitted, oracle):
    f = [r for r in fitted["no_subhalo"] if r["reliable_fit"]]
    o = {r["index"]: r for r in oracle["no_subhalo"] if r["reliable_fit"]} if oracle["no_subhalo"] else {}
    rho, p = spearmanr([r["chi2_smooth_per_dof"] for r in f], [r["peak_abs_dkappa"] for r in f])
    out = {"n": len(f), "spearman_rho_score_vs_chi2dof": float(rho), "spearman_p": float(p),
           "median_chi2_smooth_per_dof": float(np.median([r["chi2_smooth_per_dof"] for r in f])),
           "median_null_score_fitted": float(np.median([r["peak_abs_dkappa"] for r in f]))}
    if o:
        common = [r for r in f if r["index"] in o]
        ratio = np.array([r["peak_abs_dkappa"] / o[r["index"]]["peak_abs_dkappa"] for r in common])
        out.update({"median_null_score_oracle": float(np.median([o[r["index"]]["peak_abs_dkappa"] for r in common])),
                    "median_ratio_fitted_over_oracle_same_image": float(np.median(ratio)),
                    "frac_images_where_fitted_null_exceeds_oracle": float(np.mean(ratio > 1))})
        thr_o = float(np.percentile([o[i]["peak_abs_dkappa"] for i in o], 90))
        out["fpr_of_fitted_at_oracle_threshold"] = float(np.mean([r["peak_abs_dkappa"] >= thr_o for r in f]))
    return out


def timing(recs_by_pop):
    r = recs_by_pop["test_fixed60"]
    return {"mean_t_fit_s": float(np.mean([x["t_fit_s"] for x in r])), "mean_t_inv_s_3lams": float(np.mean([x["t_inv_s"] for x in r])),
            "mean_total_s": float(np.mean([x["t_fit_s"] + x["t_inv_s"] for x in r])), "n_errors": sum(1 for x in r if "error" in x)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=ROOT / "results/baseline_b")
    p.add_argument("--out", type=Path, default=ROOT / "results/baseline_b/summary_family_b.json")
    p.add_argument("--truth60", type=Path, default=ROOT / "data/test_fixed60/truth.jsonl")
    args = p.parse_args()
    truths60 = [json.loads(l) for l in open(args.truth60)]

    summary = {"root": str(args.root)}
    data = {v: load(args.root, v) for v in ["fitted", "oracle"]}
    for v, recs in data.items():
        if not recs["no_subhalo"]:
            continue
        s = {"n_unreliable": {pop: sum(1 for r in recs[pop] if not r["reliable_fit"]) for pop in POPS if recs[pop]},
             "n_records": {pop: len(recs[pop]) for pop in POPS}, "timing_test_fixed60": timing(recs), "by_lambda": {}}
        for lam, key in LAM_KEYS.items():
            if all(key in r for pop in POPS for r in recs[pop]):
                s["by_lambda"][lam] = protocol(recs, key)
        thr = s["by_lambda"]["1e5"]["threshold"]
        s["localization_lam1e5_10pct_fpr"] = localization(recs, truths60, thr)
        # published-style absolute operating points from the null distribution itself
        s["absolute_operating_points"] = {}
        for name, t in [("p99_null (~1% FPR)", s["by_lambda"]["1e5"]["null_percentiles"]["p99"]), ("max_null (0/n FPR)", s["by_lambda"]["1e5"]["null_percentiles"]["max"] + 1e-9)]:
            pr = protocol_at(recs, "peak_abs_dkappa", t)
            pr["localization"] = localization(recs, truths60, t)
            s["absolute_operating_points"][name] = pr
        summary[v] = s
    if data["fitted"]["no_subhalo"]:
        summary["fit_error_diagnostic_fitted"] = fit_error_diagnostic(data["fitted"], data["oracle"])

    # side-by-side with A and C on the same protocol
    ref = {}
    fa = ROOT / "results/baseline_a/summary/results.json"; fu = ROOT / "results/aggregate/summary.json"; fl = ROOT / "results/localization/summary.json"
    if fa.exists():
        a = json.load(open(fa))
        ref["family_a_seed0"] = {"completeness_c60": {k: v["completeness"] for k, v in a["completeness_c60_by_mass_bin"].items()},
                                 "completeness_c15": {k: v["completeness"] for k, v in a["completeness_c15_by_mass_bin"].items()},
                                 "paired_flip": a["paired_concentration_flip"], "confounder_fpr": a["confounder_fpr"], "mean_time_per_image_s": a["mean_time_per_image_s"]}
    if fu.exists():
        u = json.load(open(fu))
        ref["family_c_unet_4seeds"] = {"completeness_c60": {k: v["mean"] for k, v in u["completeness_c60"].items()},
                                       "completeness_c15": {k: v["mean"] for k, v in u["completeness_c15"].items()},
                                       "paired_flip": u["paired_flip"], "confounder_fpr": {k: v["mean"] for k, v in u["confounder_fpr"].items()}}
    if fl.exists():
        ref["localization_2px"] = json.load(open(fl))
    summary["reference"] = ref
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(render(summary))


def protocol_at(recs_by_pop, key, thr):
    fpr = {pop: float(np.mean([r[key] >= thr for r in recs_by_pop[pop] if r["reliable_fit"]])) for pop in ["no_subhalo", "multipole_m4_a1", "multipole_m4_a3"] if recs_by_pop[pop]}
    return {"threshold": thr, "fpr": fpr, "completeness_c60": completeness(recs_by_pop["test_fixed60"], key, thr),
            "completeness_c15": completeness(recs_by_pop["test_fixed15"], key, thr)}


def pct(x):
    return "  --  " if x is None else f"{100*x:5.1f}%"


def render(s):
    lines = []
    for v in ["fitted", "oracle"]:
        if v not in s:
            continue
        b = s[v]["by_lambda"]
        lines.append(f"\n## Family B / {v}   (unreliable: {s[v]['n_unreliable']}, errors: {s[v]['timing_test_fixed60']['n_errors']}, "
                     f"t_fit {s[v]['timing_test_fixed60']['mean_t_fit_s']:.2f}s + t_inv(3 lams) {s[v]['timing_test_fixed60']['mean_t_inv_s_3lams']:.2f}s per lens)")
        lines.append("lam    thr(10%FPR)  " + "  ".join(f"c60 {lo}-{hi}" for lo, hi in BINS) + "   flip c60>c15 : c15>c60   FPR a=0.01  a=0.03")
        for lam, pr in b.items():
            c = pr["completeness_c60"]; f = pr["confounder_fpr"]; fl = pr["paired_flip"]
            lines.append(f"{lam:5s}  {pr['threshold']:.4f}      " + "  ".join(pct(c[f'{lo}-{hi}']['completeness']).rjust(11) for lo, hi in BINS)
                         + f"   {fl['detected_at_c60_not_c15']:3d} : {fl['detected_at_c15_not_c60']:<3d} (n={fl['n_pairs']})   "
                         + f"{pct(f['multipole_m4_a1']['fpr'])}  {pct(f['multipole_m4_a3']['fpr'])}")
        c15 = b["1e5"]["completeness_c15"]
        lines.append("1e5 c15         " + "  ".join(pct(c15[f'{lo}-{hi}']['completeness']).rjust(11) for lo, hi in BINS))
        L = s[v]["localization_lam1e5_10pct_fpr"]
        if L.get("n_detected"):
            lines.append(f"localization (lam=1e5, 10% FPR): n_det={L['n_detected']}, median offset {L['median_offset_arcsec']:.3f}\", "
                         f"within 0.16\" {pct(L['frac_within_2px_0p16'])} ({L['n_within_2px_0p16']}), within 0.32\" {pct(L['frac_within_2meshpx_0p32'])} ({L['n_within_2meshpx_0p32']}), "
                         f"aperture mass <=0 among detected {pct(L['frac_aperture_mass_negative_among_detected'])}")
            for tag in ["0p16", "0p32"]:
                m = L.get(f"mass_error_dex_localized_{tag}")
                if m:
                    lines.append(f"  mass error (localized {tag}): n={m['n']} (+{m['n_negative_proxy_excluded']} negative-proxy excluded) median {m['median']:+.2f} dex, MAD {m['mad']:.2f}, "
                                 f"range [{m['min']:+.2f},{m['max']:+.2f}], within 0.5 dex {m['n_within_0p5dex']}, underestimated {m['n_underestimated']}")
        for name, pr in s[v]["absolute_operating_points"].items():
            c = pr["completeness_c60"]
            lines.append(f"abs. op. point {name}: thr={pr['threshold']:.4f}  FPR ns/a1/a3 = {pct(pr['fpr']['no_subhalo'])}/{pct(pr['fpr']['multipole_m4_a1'])}/{pct(pr['fpr']['multipole_m4_a3'])}"
                         f"  c60 = " + " ".join(pct(c[f'{lo}-{hi}']['completeness']).strip() for lo, hi in BINS)
                         + (f"  loc0.16 {pct(pr['localization'].get('frac_within_2px_0p16'))}" if pr['localization'].get('n_detected') else ""))
    if "fit_error_diagnostic_fitted" in s:
        d = s["fit_error_diagnostic_fitted"]
        lines.append(f"\nfit-error diagnostic (no_subhalo, fitted): Spearman(score, chi2/N) rho={d['spearman_rho_score_vs_chi2dof']:+.2f} (p={d['spearman_p']:.1e}), "
                     f"median chi2/N {d['median_chi2_smooth_per_dof']:.2f}; median null score fitted {d['median_null_score_fitted']:.4f}"
                     + (f" vs oracle {d['median_null_score_oracle']:.4f} (median ratio {d['median_ratio_fitted_over_oracle_same_image']:.2f}, "
                        f"fitted>oracle on {pct(d['frac_images_where_fitted_null_exceeds_oracle'])} of images; FPR of fitted at oracle's threshold {pct(d['fpr_of_fitted_at_oracle_threshold'])})" if "median_null_score_oracle" in d else ""))
    r = s.get("reference", {})
    if r:
        lines.append("\n## same images, same protocol (10% FPR on no_subhalo)")
        lines.append("family                 " + "  ".join(f"c60 {lo}-{hi}" for lo, hi in BINS) + "   flip        FPR a=0.01  a=0.03")
        if "fitted" in s:
            pr = s["fitted"]["by_lambda"]["1e5"]; c = pr["completeness_c60"]; fl = pr["paired_flip"]; f = pr["confounder_fpr"]
            lines.append("B fitted (lam=1e5)     " + "  ".join(pct(c[f'{lo}-{hi}']['completeness']).rjust(11) for lo, hi in BINS) + f"   {fl['detected_at_c60_not_c15']:3d}:{fl['detected_at_c15_not_c60']:<3d}     {pct(f['multipole_m4_a1']['fpr'])}  {pct(f['multipole_m4_a3']['fpr'])}")
        if "oracle" in s:
            pr = s["oracle"]["by_lambda"]["1e5"]; c = pr["completeness_c60"]; fl = pr["paired_flip"]; f = pr["confounder_fpr"]
            lines.append("B oracle (lam=1e5)     " + "  ".join(pct(c[f'{lo}-{hi}']['completeness']).rjust(11) for lo, hi in BINS) + f"   {fl['detected_at_c60_not_c15']:3d}:{fl['detected_at_c15_not_c60']:<3d}     {pct(f['multipole_m4_a1']['fpr'])}  {pct(f['multipole_m4_a3']['fpr'])}")
        if "family_a_seed0" in r:
            a = r["family_a_seed0"]; fl = a["paired_flip"]; f = a["confounder_fpr"]
            lines.append("A parametric (seed 0)  " + "  ".join(pct(a['completeness_c60'][f'{lo}-{hi}']).rjust(11) for lo, hi in BINS) + f"   {fl['detected_at_c60_not_c15']:3d}:{fl['detected_at_c15_not_c60']:<3d}     {pct(f['multipole m=4, a=0.01*thetaE'])}  {pct(f['multipole m=4, a=0.03*thetaE'])}")
        if "family_c_unet_4seeds" in r:
            u = r["family_c_unet_4seeds"]; fl = u["paired_flip"]; f = u["confounder_fpr"]
            lines.append("C U-Net (4-seed mean)  " + "  ".join(pct(u['completeness_c60'][f'{lo}-{hi}']).rjust(11) for lo, hi in BINS) + f"   {fl['c60_not_c15_mean']:5.0f}:{fl['c15_not_c60_mean']:<4.0f}   {pct(f['multipole m=4, a=0.01*thetaE'])}  {pct(f['multipole m=4, a=0.03*thetaE'])}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
