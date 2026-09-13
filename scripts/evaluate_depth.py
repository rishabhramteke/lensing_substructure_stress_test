"""Referee round 7, point M6: does any of this survive at survey depth?

Every result in the paper is measured at the deep HST operating point the reproduced papers
use: median arc signal-to-noise ~1.3e3, peak pixel ~240 sigma. `data/test_shallow_*` re-renders
the IDENTICAL lenses, sources and subhalos (same generation seeds) with the exposure time cut
from 5400 s to 135 s, taking the median arc S/N to ~2.4e2 and the peak pixel to ~41 sigma. Only
depth differs, so the comparison is a matched pair like the concentration one.

    python scripts/evaluate_depth.py
-> results/depth.json, paper/tables/depth.tex
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from detector.dataset import LensPatchDataset  # noqa: E402
from detector.unet import UNet  # noqa: E402

BINS = [(9.0, 9.5), (9.5, 10.0), (10.0, 10.5), (10.5, 11.0)]
SEEDS = {"detector_v0_best": "checkpoints/unet_v0", "detector_v1_seed1": "checkpoints/unet_v1_seed1",
         "detector_v2_seed2": "checkpoints/unet_v2_seed2", "detector_v3_seed3": "checkpoints/unet_v3_seed3"}
# (label, scan root, population folder suffixes)
ARMS = {"deep": ("results/baseline_a", {"no_subhalo": "no_subhalo", "c60": "test_fixed60", "c15": "test_fixed15", "mp": "multipole_m4_a3"}),
        "shallow": ("results/baseline_a_shallow", {"no_subhalo": "no_subhalo", "c60": "test_fixed60", "c15": "test_fixed15", "mp": "multipole_m4_a3"})}
DATA = {"deep": {"c60": "test_fixed60", "c15": "test_fixed15", "no_subhalo": "no_subhalo"},
        "shallow": {"c60": "test_shallow_fixed60", "c15": "test_shallow_fixed15", "no_subhalo": "test_shallow_no_subhalo"}}


def wilson(k, n, z=1.0):
    if not n:
        return (np.nan,) * 3
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def load(root, pop):
    p = ROOT / root / pop / "scan_results.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else None


def family_a(arm):
    root, pops = ARMS[arm]
    R = {k: load(root, v) for k, v in pops.items()}
    if not R["no_subhalo"]:
        return None
    rel = [r for r in R["no_subhalo"] if r["reliable_fit"]]
    thr = float(np.quantile([r["delta_chi2"] for r in rel], 0.90))
    out = {"threshold_10pct": thr, "clean_fpr_floor": float(np.mean([r["delta_chi2"] > 0 for r in rel])),
           "n_clean_reliable": len(rel), "excluded": {k: (sum(1 for r in v if not r["reliable_fit"]), len(v)) for k, v in R.items() if v}}
    for key in ("c60", "c15"):
        if not R[key]:
            continue
        pos = [r for r in R[key] if r["reliable_fit"] and r["has_subhalo"]]
        for name, f in (("cal", lambda r: r["delta_chi2"] >= thr), ("floor", lambda r: r["delta_chi2"] > 0)):
            d = {}
            for lo, hi in BINS:
                sel = [r for r in pos if lo <= r["log10_M200_true"] < hi]
                k = sum(1 for r in sel if f(r)); p_, l_, h_ = wilson(k, len(sel))
                d[f"{lo}-{hi}"] = {"n": len(sel), "k": k, "p": p_, "lo": l_, "hi": h_}
            out[f"{key}_{name}"] = d
    # matched-rate comparison: recalibrate on this arm's own clean control at the DEEP arm's
    # floored false-positive rate (0.68%), so completeness is compared at equal false alarms
    q = 1.0 - 0.0068
    thr_match = float(np.quantile([r["delta_chi2"] for r in rel], q))
    out["threshold_matched_rate"] = thr_match
    for key in ("c60", "c15"):
        if not R[key]:
            continue
        pos = [r for r in R[key] if r["reliable_fit"] and r["has_subhalo"]]
        d = {}
        for lo, hi in BINS:
            sel = [r for r in pos if lo <= r["log10_M200_true"] < hi]
            k = sum(1 for r in sel if r["delta_chi2"] >= thr_match); p_, l_, h_ = wilson(k, len(sel))
            d[f"{lo}-{hi}"] = {"n": len(sel), "k": k, "p": p_, "lo": l_, "hi": h_}
        out[f"{key}_matched"] = d
    if R["mp"]:
        rel_mp = [r for r in R["mp"] if r["reliable_fit"]]
        out["mp_cal"] = float(np.mean([r["delta_chi2"] >= thr for r in rel_mp]))
        out["mp_floor"] = float(np.mean([r["delta_chi2"] > 0 for r in rel_mp]))
        out["mp_n"] = len(rel_mp)
    # paired flip on the common indices
    if R["c60"] and R["c15"]:
        a = {r["index"]: r for r in R["c60"] if r["reliable_fit"] and r["has_subhalo"]}
        b = {r["index"]: r for r in R["c15"] if r["reliable_fit"]}
        common = sorted(set(a) & set(b))
        for name, f in (("cal", lambda r: r["delta_chi2"] >= thr), ("floor", lambda r: r["delta_chi2"] > 0)):
            out[f"flip_{name}"] = {"n": len(common),
                                   "lost": sum(1 for i in common if f(a[i]) and not f(b[i])),
                                   "gained": sum(1 for i in common if f(b[i]) and not f(a[i]))}
    return out


def family_c(arm):
    """U-Net completeness in each arm. The network is trained on deep images, so applying its
    deep threshold to shallow ones measures a distribution shift, not detection: we therefore
    report both that naive transfer and a threshold recalibrated on the arm's OWN clean control
    at the same 10% false-positive rate."""
    dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    per_seed = []
    for run, ckdir in SEEDS.items():
        ck_p, res_p = ROOT / ckdir / "model_best.pt", ROOT / "results" / run / "results.json"
        if not ck_p.exists():
            continue
        ck = torch.load(ck_p, map_location=dev)
        model = UNet(in_ch=1, base=ck.get("base", 16)).to(dev)
        model.load_state_dict(ck["model_state"]); model.eval()
        thr_deep = json.loads(res_p.read_text())["threshold_at_10pct_fpr"]
        scores, masses = {}, {}
        for key, pop in DATA[arm].items():
            ds = LensPatchDataset(ROOT / "data" / pop, scale=ck["scale"])
            sc, ms = [], []
            with torch.no_grad():
                for i in range(len(ds)):
                    sub = ds.truths[i].get("subhalo")
                    if key != "no_subhalo" and not sub:
                        continue
                    sc.append(float(torch.sigmoid(model(ds[i][0].unsqueeze(0).to(dev))).max()))
                    ms.append(sub["log10_M200"] if sub else np.nan)
            scores[key], masses[key] = np.array(sc), np.array(ms)
        thr_own = float(np.quantile(scores["no_subhalo"], 0.90))
        row = {"threshold_deep": thr_deep, "threshold_own": thr_own,
               "fpr_clean_at_deep_threshold": float((scores["no_subhalo"] >= thr_deep).mean())}
        for key in ("c60", "c15"):
            sc, ms = scores[key], masses[key]
            for name, thr in (("naive", thr_deep), ("recal", thr_own)):
                row[f"{key}_{name}"] = {f"{lo}-{hi}": float((sc[(ms >= lo) & (ms < hi)] >= thr).mean()) for lo, hi in BINS}
        per_seed.append(row)
    if not per_seed:
        return None
    out = {"n_seeds": len(per_seed),
           "fpr_clean_at_deep_threshold": float(np.mean([r["fpr_clean_at_deep_threshold"] for r in per_seed]))}
    for key in ("c60", "c15"):
        for name in ("naive", "recal"):
            out[f"{key}_{name}"] = {f"{lo}-{hi}": {
                "mean": float(np.mean([r[f"{key}_{name}"][f"{lo}-{hi}"] for r in per_seed])),
                "std": float(np.std([r[f"{key}_{name}"][f"{lo}-{hi}"] for r in per_seed]))} for lo, hi in BINS}
    return out


def main():
    res = {arm: {"A": family_a(arm), "C": family_c(arm)} for arm in ("deep", "shallow")}
    (ROOT / "results/depth.json").write_text(json.dumps(res, indent=2))
    D, S = res["deep"], res["shallow"]
    lines = [r"\begin{tabular}{@{}lcc@{}}", r"\toprule",
             r" & deep & shallow \\", r"median arc S/N & $1.3\times10^{3}$ & $2.4\times10^{2}$ \\", r"\midrule",
             r"\multicolumn{3}{@{}l}{Family A, parametric scan} \\",
             f"\\quad clean false positives at $\\dchi>0$ & {100*D['A']['clean_fpr_floor']:.1f}\\% & {100*S['A']['clean_fpr_floor']:.0f}\\% \\\\",
             f"\\quad macro fits rejected, $c{{=}}60$ & {D['A']['excluded']['c60'][0]}/{D['A']['excluded']['c60'][1]} & {S['A']['excluded']['c60'][0]}/{S['A']['excluded']['c60'][1]} \\\\",
             r"\quad \emph{at a matched 0.68\% false-positive rate:} \\"]
    for lo, hi in BINS:
        k = f"{lo}-{hi}"
        lines.append(f"\\quad\\quad compl.\\ $c{{=}}60$, $10^{{{lo}}}$--$10^{{{hi}}}\\Msun$ & {100*D['A']['c60_matched'][k]['p']:.0f}\\% & {100*S['A']['c60_matched'][k]['p']:.0f}\\% \\\\")
    lines.append(f"\\quad FPR, multipole $a_m{{=}}0.03\\,\\thetaE$, $\\dchi>0$ & {100*D['A']['mp_floor']:.0f}\\% & {100*S['A']['mp_floor']:.0f}\\% \\\\")
    lines.append(r"\addlinespace[2pt]")
    lines.append(r"\multicolumn{3}{@{}l}{Family C, U-Net (4 seeds, trained on deep images)} \\")
    lines.append(f"\\quad clean FPR at its deep threshold & 10\\% & {100*S['C']['fpr_clean_at_deep_threshold']:.0f}\\% \\\\")
    lines.append(r"\quad \emph{threshold recalibrated to 10\% on each arm:} \\")
    for lo, hi in BINS:
        k = f"{lo}-{hi}"
        lines.append(f"\\quad\\quad compl.\\ $c{{=}}60$, $10^{{{lo}}}$--$10^{{{hi}}}\\Msun$ & {100*D['C']['c60_recal'][k]['mean']:.0f}\\% & {100*S['C']['c60_recal'][k]['mean']:.0f}\\% \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "paper/tables/depth.tex").write_text("\n".join(lines) + "\n")
    for arm in ("deep", "shallow"):
        A, C = res[arm]["A"], res[arm]["C"]
        print(f"== {arm}: 10% thr {A['threshold_10pct']:.1f} | matched-rate thr {A['threshold_matched_rate']:.1f} | clean floor FPR {100*A['clean_fpr_floor']:.2f}% | excluded {A['excluded']}")
        print("   A c60 @matched-rate ", " ".join(f"{100*A['c60_matched'][f'{lo}-{hi}']['p']:3.0f}" for lo, hi in BINS),
              "| c15 ", " ".join(f"{100*A['c15_matched'][f'{lo}-{hi}']['p']:3.0f}" for lo, hi in BINS), f"| mp floor {100*A['mp_floor']:.0f}%")
        if C:
            print(f"   C clean FPR at deep thr {100*C['fpr_clean_at_deep_threshold']:.0f}% | own thr recal")
            print("   C c60 naive ", " ".join(f"{100*C['c60_naive'][f'{lo}-{hi}']['mean']:3.0f}" for lo, hi in BINS),
                  "| c60 recal ", " ".join(f"{100*C['c60_recal'][f'{lo}-{hi}']['mean']:3.0f}" for lo, hi in BINS),
                  "| c15 recal ", " ".join(f"{100*C['c15_recal'][f'{lo}-{hi}']['mean']:3.0f}" for lo, hi in BINS))


if __name__ == "__main__":
    main()
