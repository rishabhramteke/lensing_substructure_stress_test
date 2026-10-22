"""Depth stress (referee round 7, M6; revised 2026-09-15 for the joint scan and the 30k U-Net).

`data/test_shallow_*` re-renders the IDENTICAL lenses, sources and subhalos with the exposure time
cut from 5400 s to 135 s (median arc S/N 1.3e3 -> 2.4e2). Family A is the joint scan on both arms
(deep: the full 1 000-lens populations; shallow: the 300-lens seed-matched subsamples), each arm
calibrated on its own clean control and also scored at the absolute Delta chi^2 >= 20 threshold.
The U-Net is scored three ways: the deep-trained set at its deep threshold (a distribution-shift
measurement), the same set recalibrated on the shallow clean control, and a set RETRAINED on
30 000 shallow images (paper_config.UNET_SHALLOW).

    python scripts/evaluate_depth.py -> results/depth.json, paper/tables/depth.tex
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
from paper_config import A_JOINT_FULL, A_JOINT_SHALLOW, A_JOINT_SHALLOW600, A_ABS, wilson, load_scan, unet_seeds, UNET_SHALLOW  # noqa: E402

BINS = [(9.0, 9.5), (9.5, 10.0), (10.0, 10.5), (10.5, 11.0)]
ARMS = {"deep": (A_JOINT_FULL, {"no_subhalo": "no_subhalo", "c60": "test_fixed60", "c15": "test_fixed15", "mp": "multipole_m4_a3"}),
        "shallow": (A_JOINT_SHALLOW, {"no_subhalo": "no_subhalo", "c60": "test_fixed60", "c15": "test_fixed15", "mp": "multipole_m4_a3"}),
        # referee round 19: an intermediate depth, 600 s (a snapshot-length HST exposure), joint scan on 200 clean and c=60 lenses
        "mid": (A_JOINT_SHALLOW600, {"no_subhalo": "no_subhalo", "c60": "test_fixed60", "c15": "test_fixed15", "mp": "multipole_m4_a3"})}
DATA = {"deep": {"c60": "test_fixed60", "c15": "test_fixed15", "no_subhalo": "no_subhalo"},
        "shallow": {"c60": "test_shallow_fixed60", "c15": "test_shallow_fixed15", "no_subhalo": "test_shallow_no_subhalo"},
        "mid": {"c60": "test_shallow600_fixed60", "c15": "test_shallow600_fixed15", "no_subhalo": "test_shallow600_no_subhalo"}}


def arc_snr(pop, n=300):
    """Median over the first n lenses of the total arc signal-to-noise sqrt(sum((I_noiseless/sigma)^2)) and of the
    peak-pixel signal-to-noise, with sigma from lenstronomy's own per-pixel noise estimate on the noisy image."""
    from lenstronomy.SimulationAPI.sim_api import SimAPI
    d = ROOT / "data" / pop
    if not (d / "manifest.json").exists():
        return None
    m = json.loads((d / "manifest.json").read_text()); kb = m["kwargs_band"]; npx = m["image_shape"][0]
    sim = SimAPI(num_pix=npx, kwargs_single_band=kb, kwargs_model={"lens_model_list": ["EPL"], "source_light_model_list": ["SERSIC_ELLIPSE"]})
    noisy = np.load(d / "images_full_noisy.npy", mmap_mode="r"); clean = np.load(d / "images_full_noiseless.npy", mmap_mode="r")
    tot, pk = [], []
    for i in range(min(n, len(noisy))):
        r = np.asarray(clean[i]) / sim.estimate_noise(np.asarray(noisy[i]))
        tot.append(float(np.sqrt(np.sum(r ** 2)))); pk.append(float(r.max()))
    return {"exposure_s": kb.get("exposure_time"), "median_total_snr": float(np.median(tot)), "median_peak_snr": float(np.median(pk)), "n": len(tot)}


def family_a(arm):
    root, pops = ARMS[arm]
    R = {k: load_scan(root, v) for k, v in pops.items()}
    if not R["no_subhalo"]:
        return None
    rel = [r for r in R["no_subhalo"] if r["reliable_fit"]]
    thr = float(np.quantile([r["delta_chi2"] for r in rel], 0.90))
    ops = {"cal": thr, "abs": A_ABS}
    out = {"threshold_10pct": thr, "n_clean_reliable": len(rel),
           "clean_fpr_abs": float(np.mean([r["delta_chi2"] >= A_ABS for r in rel])),
           "excluded": {k: (sum(1 for r in v if not r["reliable_fit"]), len(v)) for k, v in R.items() if v},
           "truncated": {k: sum(1 for r in v if r.get("truncated")) for k, v in R.items() if v}}
    for key in ("c60", "c15"):
        if not R[key]:
            continue
        pos = [r for r in R[key] if r["reliable_fit"] and r["has_subhalo"]]
        for name, t in ops.items():
            d = {}
            for lo, hi in BINS:
                sel = [r for r in pos if lo <= r["log10_M200_true"] < hi]
                k = sum(1 for r in sel if r["delta_chi2"] >= t); p_, l_, h_ = wilson(k, len(sel))
                d[f"{lo}-{hi}"] = {"n": len(sel), "k": k, "p": p_, "lo": l_, "hi": h_}
            out[f"{key}_{name}"] = d
    if R.get("mp"):
        rel_mp = [r for r in R["mp"] if r["reliable_fit"]]
        out["mp_cal"] = float(np.mean([r["delta_chi2"] >= thr for r in rel_mp]))
        out["mp_abs"] = float(np.mean([r["delta_chi2"] >= A_ABS for r in rel_mp])); out["mp_n"] = len(rel_mp)
    if R.get("c60") and R.get("c15"):
        a = {r["index"]: r for r in R["c60"] if r["reliable_fit"] and r["has_subhalo"]}
        b = {r["index"]: r for r in R["c15"] if r["reliable_fit"]}
        common = sorted(set(a) & set(b))
        for name, t in ops.items():
            out[f"flip_{name}"] = {"n": len(common), "lost": sum(1 for i in common if a[i]["delta_chi2"] >= t and b[i]["delta_chi2"] < t),
                                   "gained": sum(1 for i in common if b[i]["delta_chi2"] >= t and a[i]["delta_chi2"] < t)}
    return out


def family_c(arm, seeds):
    """U-Net completeness in one arm for a seed set: at each seed's stored threshold ("naive"; for the
    deep-trained set on the shallow arm this measures a distribution shift) and at a threshold
    recalibrated on the arm's own clean control ("recal")."""
    dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    per_seed = []
    for run, ckdir in seeds.items():
        ck_p, res_p = ROOT / ckdir / "model_best.pt", ROOT / "results" / run / "results.json"
        if not ck_p.exists() or not res_p.exists():
            continue
        ck = torch.load(ck_p, map_location=dev)
        model = UNet(in_ch=1, base=ck.get("base", 16)).to(dev)
        model.load_state_dict(ck["model_state"]); model.eval()
        thr_stored = json.loads(res_p.read_text())["threshold_at_10pct_fpr"]
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
        neg = scores["no_subhalo"]; pos = scores["c60"]
        row = {"threshold_stored": thr_stored, "threshold_own": thr_own,
               "fpr_clean_at_stored_threshold": float((neg >= thr_stored).mean()),
               "auc_c60": float(np.mean([(p > neg).mean() + 0.5 * (p == neg).mean() for p in pos]))}
        for key in ("c60", "c15"):
            sc, ms = scores[key], masses[key]
            for name, thr in (("naive", thr_stored), ("recal", thr_own)):
                row[f"{key}_{name}"] = {f"{lo}-{hi}": float((sc[(ms >= lo) & (ms < hi)] >= thr).mean()) for lo, hi in BINS}
        per_seed.append(row)
    if not per_seed:
        return None
    out = {"n_seeds": len(per_seed), "seeds": list(seeds),
           "fpr_clean_at_stored_threshold": float(np.mean([r["fpr_clean_at_stored_threshold"] for r in per_seed])),
           "auc_c60": {"mean": float(np.mean([r["auc_c60"] for r in per_seed])), "std": float(np.std([r["auc_c60"] for r in per_seed]))}}
    for key in ("c60", "c15"):
        for name in ("naive", "recal"):
            out[f"{key}_{name}"] = {f"{lo}-{hi}": {
                "mean": float(np.mean([r[f"{key}_{name}"][f"{lo}-{hi}"] for r in per_seed])),
                "std": float(np.std([r[f"{key}_{name}"][f"{lo}-{hi}"] for r in per_seed]))} for lo, hi in BINS}
    return out


def main():
    res = {arm: {"A": family_a(arm), "C": family_c(arm, unet_seeds())} for arm in ("deep", "shallow")}
    res["shallow"]["C_retrained"] = family_c("shallow", UNET_SHALLOW)
    mid_ok = all((ROOT / "data" / p / "manifest.json").exists() for p in DATA["mid"].values())
    res["mid"] = {"A": family_a("mid"), "C": family_c("mid", unet_seeds()) if mid_ok else None} if mid_ok else None
    res["snr"] = {arm: arc_snr(DATA[arm]["c60"]) for arm in ("deep", "mid", "shallow")}
    (ROOT / "results/depth.json").write_text(json.dumps(res, indent=2))
    D, S = res["deep"], res["shallow"]
    def sci(x):
        e = int(np.floor(np.log10(x))); return f"${x/10**e:.1f}\\times10^{{{e}}}$"
    SNR = res["snr"]
    M = res.get("mid") or {}
    MA, MC = M.get("A"), M.get("C")
    def ma(f, default="--"):
        try:
            return f(MA) if MA else default
        except (KeyError, TypeError):
            return default
    def mc(f, default="--"):
        try:
            return f(MC) if MC else default
        except (KeyError, TypeError):
            return default
    def pc(x, nd=0):
        return f"{100*x:.{nd}f}\\%"
    def bins_row(label, get):
        return f"\\quad {label} & " + " & ".join(get(arm) for arm in ("D", "S")) + r" \\"
    A_ = {"D": D["A"], "S": S["A"]}
    def four(d, key, sub="p"):
        return " / ".join(f"{100*d[key][f'{lo}-{hi}'][sub]:.0f}" for lo, hi in BINS)
    def _low_bins_row():
        """the two lowest mass bins of the joint scan at its 10% calibration (results/joint_scan.json), which the
        summary figure's depth cell uses; the 600 s arm was scanned in the four upper bins only"""
        Jd = json.loads((ROOT / "results/joint_scan.json").read_text())
        g = lambda k: " / ".join(f"{100*Jd[k]['cal']['completeness_c60'][b]['p']:.0f}" for b in ("8.0-8.5", "8.5-9.0"))  # noqa: E731
        return f"\\quad compl.\\ $c{{=}}60$, 10\\% cal., 8--8.5 / 8.5--9 & {g('core')} & -- & {g('shallow')} \\\\"
    L = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
         r" & deep & 600 s & shallow \\", r"exposure time & 5400 s & 600 s & 135 s \\", "median arc S/N$^{b}$ & " + " & ".join(sci(SNR[a]["median_total_snr"]) if SNR.get(a) else "--" for a in ("deep", "mid", "shallow")) + r" \\",
         "median peak-pixel S/N & " + " & ".join(f"{SNR[a]['median_peak_snr']:.0f}" if SNR.get(a) else "--" for a in ("deep", "mid", "shallow")) + r" \\", r"\midrule",
         r"\multicolumn{4}{@{}l}{\emph{A, joint scan}} \\",
         f"\\quad threshold at 10\\% FPR & ${D['A']['threshold_10pct']:.1f}$ & {ma(lambda A: '$%.1f$' % A['threshold_10pct'])} & ${S['A']['threshold_10pct']:.1f}$ \\\\",
         f"\\quad clean FPR at $\\dchi\\ge{A_ABS:g}$ & {pc(D['A']['clean_fpr_abs'], 2)} & {ma(lambda A: pc(A['clean_fpr_abs'], 2))} & {pc(S['A']['clean_fpr_abs'], 2)} \\\\",
         f"\\quad macro fits rejected, $c{{=}}60$ & {D['A']['excluded']['c60'][0]}/{D['A']['excluded']['c60'][1]} & {ma(lambda A: f"{A['excluded']['c60'][0]}/{A['excluded']['c60'][1]}")} & {S['A']['excluded']['c60'][0]}/{S['A']['excluded']['c60'][1]} \\\\",
         f"\\quad compl.\\ $c{{=}}60$, 10\\% cal.$^{{a}}$ & {four(D['A'], 'c60_cal')} & {ma(lambda A: four(A, 'c60_cal'))} & {four(S['A'], 'c60_cal')} \\\\",
         _low_bins_row(),
         f"\\quad compl.\\ $c{{=}}60$, $\\dchi\\ge{A_ABS:g}$$^{{a}}$ & {four(D['A'], 'c60_abs')} & {ma(lambda A: four(A, 'c60_abs'))} & {four(S['A'], 'c60_abs')} \\\\",
         f"\\quad compl.\\ $c{{=}}15$, $\\dchi\\ge{A_ABS:g}$$^{{a}}$ & {four(D['A'], 'c15_abs')} & {ma(lambda A: four(A, 'c15_abs'))} & {four(S['A'], 'c15_abs')} \\\\",
         f"\\quad FPR multipole, cal.\\ / $\\ge{A_ABS:g}$ & {pc(D['A']['mp_cal'])} / {pc(D['A']['mp_abs'])} & {ma(lambda A: pc(A['mp_cal']) + ' / ' + pc(A['mp_abs']))} & {pc(S['A']['mp_cal'])} / {pc(S['A']['mp_abs'])} \\\\",
         r"\addlinespace[2pt]"]
    C = S["C"]
    L.append(f"\\multicolumn{{4}}{{@{{}}l}}{{\\emph{{C, U-Net trained deep ({C['n_seeds']} seeds)}}}} \\\\")
    L.append(f"\\quad clean FPR at deep threshold & 10\\% & {mc(lambda C_: pc(C_['fpr_clean_at_stored_threshold']))} & {pc(C['fpr_clean_at_stored_threshold'])} \\\\")
    L.append(f"\\quad AUC, $c{{=}}60$ vs.\\ clean & {D['C']['auc_c60']['mean']:.2f} & {mc(lambda C_: f"{C_['auc_c60']['mean']:.2f}")} & {C['auc_c60']['mean']:.2f} \\\\")
    L.append(f"\\quad compl.\\ $c{{=}}60$, recal.\\ 10\\%$^{{a}}$ & {four(D['C'], 'c60_recal', 'mean')} & {mc(lambda C_: four(C_, 'c60_recal', 'mean'))} & {four(C, 'c60_recal', 'mean')} \\\\")
    CR = S.get("C_retrained")
    if CR:
        L.append(r"\addlinespace[2pt]")
        L.append(f"\\multicolumn{{4}}{{@{{}}l}}{{\\emph{{C, U-Net retrained on 30\\,000 shallow images ({CR['n_seeds']} seeds)}}}} \\\\")
        L.append(f"\\quad AUC, $c{{=}}60$ vs.\\ clean & -- & -- & {CR['auc_c60']['mean']:.2f} \\\\")
        L.append(f"\\quad compl.\\ $c{{=}}60$$^{{a}}$ & -- & -- & {four(CR, 'c60_recal', 'mean')} \\\\")
        L.append(f"\\quad compl.\\ $c{{=}}15$$^{{a}}$ & -- & -- & {four(CR, 'c15_recal', 'mean')} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "paper/tables/depth.tex").write_text("\n".join(L) + "\n")
    for arm in ("deep", "shallow"):
        A, Cc = res[arm]["A"], res[arm]["C"]
        print(f"== {arm}: thr {A['threshold_10pct']:.1f} | clean FPR abs {100*A['clean_fpr_abs']:.2f}% | excluded {A['excluded']} | truncated {A['truncated']}")
        print("   A c60 cal ", " ".join(f"{100*A['c60_cal'][f'{lo}-{hi}']['p']:3.0f}" for lo, hi in BINS), "| abs", " ".join(f"{100*A['c60_abs'][f'{lo}-{hi}']['p']:3.0f}" for lo, hi in BINS),
              "| c15 cal", " ".join(f"{100*A['c15_cal'][f'{lo}-{hi}']['p']:3.0f}" for lo, hi in BINS), f"| mp {100*A['mp_cal']:.0f}/{100*A['mp_abs']:.0f}% | flips {A['flip_cal']} {A['flip_abs']}")
        if Cc:
            print(f"   C deep-trained: clean FPR at stored thr {100*Cc['fpr_clean_at_stored_threshold']:.0f}% AUC {Cc['auc_c60']['mean']:.3f}")
            print("   C c60 naive ", " ".join(f"{100*Cc['c60_naive'][f'{lo}-{hi}']['mean']:3.0f}" for lo, hi in BINS), "| recal", " ".join(f"{100*Cc['c60_recal'][f'{lo}-{hi}']['mean']:3.0f}" for lo, hi in BINS),
                  "| c15 recal", " ".join(f"{100*Cc['c15_recal'][f'{lo}-{hi}']['mean']:3.0f}" for lo, hi in BINS))
    CR = res["shallow"].get("C_retrained")
    if CR:
        print(f"   C retrained shallow: AUC {CR['auc_c60']['mean']:.3f} c60", " ".join(f"{100*CR['c60_recal'][f'{lo}-{hi}']['mean']:3.0f}" for lo, hi in BINS), "| c15", " ".join(f"{100*CR['c15_recal'][f'{lo}-{hi}']['mean']:3.0f}" for lo, hi in BINS))


if __name__ == "__main__":
    main()
