"""Family A, the joint (macro re-fitted) scan, on every population it was run on: the numbers the
revised paper quotes for the parametric family (2026-09-15).

Operating points. The joint statistic Delta chi^2 = chi^2_smooth(best basin) - min_cell chi^2_joint
is non-negative up to optimizer noise (the joint model nests the smooth one), so its clean
distribution is a proper null and needs no floor. Three thresholds are scored throughout:
  * `cal`  -- the 90th percentile of the clean control (the paper's common 10%-FPR axis),
  * `abs`  -- Delta chi^2 >= 20 (Delta ln L ~ 10, the Delta ln Z > 10 proxy of the parametric literature),
  * `abs100` -- Delta chi^2 >= 100.
Each stress arm (shallow, PSF-widened, lens light) is calibrated on its OWN clean control.

    python scripts/evaluate_joint_scan.py   -> results/joint_scan.json, paper/tables/joint_operating_points.tex,
                                               paper/tables/los.tex, paper/tables/gate_fpr.tex
"""
import json
from pathlib import Path

import numpy as np

from paper_config import (ROOT, BINS, LOC_ARCSEC, A_ABS, A_ABS_STRICT, wilson, load_scan,
                          A_JOINT_FULL, A_FROZEN_FULL, A_JOINT_LOS, A_JOINT_SHALLOW, A_JOINT_DECOY, A_JOINT_PSF, A_FROZEN_PSF,
                          A_JOINT_LL, A_JOINT_LL2, A_JOINT_TIDAL, A_JOINT_CM)

TRUTH = {}


def truths(pop):
    if pop not in TRUTH:
        TRUTH[pop] = [json.loads(l) for l in open(ROOT / "data" / pop / "truth.jsonl")]
    return TRUTH[pop]


def ops_from_clean(clean_recs):
    rel = [r for r in clean_recs if r["reliable_fit"]]
    d = np.array([r["delta_chi2"] for r in rel])
    thr = float(np.quantile(d, 0.90))
    return {"cal": thr, "abs": A_ABS, "abs100": A_ABS_STRICT}, rel


def hit(r, thr):
    return r["delta_chi2"] >= thr


def completeness(recs, thr, conservative=False, mass_key="log10_M200_true"):
    pool = [r for r in recs if r["has_subhalo"] and (conservative or r["reliable_fit"])]
    out = {}
    for lo, hi in BINS:
        sel = [r for r in pool if lo <= r[mass_key] < hi]
        k = sum(1 for r in sel if r["reliable_fit"] and hit(r, thr))
        p, l, h = wilson(k, len(sel))
        out[f"{lo}-{hi}"] = {"n": len(sel), "k": k, "p": p, "lo": l, "hi": h}
    return out


def fpr(recs, thr):
    rel = [r for r in recs if r["reliable_fit"]]
    k = sum(1 for r in rel if hit(r, thr)); p, l, h = wilson(k, len(rel))
    return {"n": len(rel), "k": k, "p": p, "lo": l, "hi": h}


def detections_c60(recs, thr, pop="test_fixed60"):
    tr = truths(pop)
    det = [r for r in recs if r["reliable_fit"] and r["has_subhalo"] and hit(r, thr)]
    if not det:
        return {"n": 0}
    off = np.array([np.hypot(r["best_x"] - tr[r["index"]]["subhalo"]["x"], r["best_y"] - tr[r["index"]]["subhalo"]["y"]) for r in det])
    merr = np.array([r["best_log10_m"] - tr[r["index"]]["subhalo"]["log10_M200"] for r in det])
    loc = off < LOC_ARCSEC
    return {"n": len(det), "n_localized": int(loc.sum()), "frac_localized": float(loc.mean()), "median_offset_arcsec": float(np.median(off)),
            "mass_error_all": {"median": float(np.median(merr)), "mad": float(np.median(np.abs(merr - np.median(merr)))), "n_low": int((merr < 0).sum()),
                               "n_within_0p25": int((np.abs(merr) < 0.25).sum()), "n_within_0p5": int((np.abs(merr) < 0.5).sum())},
            "mass_error_localized": ({"median": float(np.median(merr[loc])), "n_low": int((merr[loc] < 0).sum()), "n": int(loc.sum()),
                                      "n_within_0p25": int((np.abs(merr[loc]) < 0.25).sum()), "n_within_0p5": int((np.abs(merr[loc]) < 0.5).sum())} if loc.any() else None),
            "median_dchi2": float(np.median([r["delta_chi2"] for r in det]))}


def paired_flip(a_recs, b_recs, thr):
    a = {r["index"]: r for r in a_recs if r["reliable_fit"] and r["has_subhalo"]}
    b = {r["index"]: r for r in b_recs if r["reliable_fit"] and r["has_subhalo"]}
    common = sorted(set(a) & set(b))
    return {"n_pairs": len(common), "lost": sum(1 for i in common if hit(a[i], thr) and not hit(b[i], thr)),
            "gained": sum(1 for i in common if hit(b[i], thr) and not hit(a[i], thr))}


def run_stats(recs):
    rel = [r for r in recs if r["reliable_fit"]]
    out = {"n": len(recs), "n_reliable": len(rel), "n_unreliable": len(recs) - len(rel), "n_truncated": sum(1 for r in recs if r.get("truncated"))}
    sub = [r for r in recs if r["has_subhalo"]]
    if sub:
        out["n_unreliable_subhalo_only"] = sum(1 for r in sub if not r["reliable_fit"]); out["n_subhalo"] = len(sub)
    out["cost_s_median_fit_plus_scan"] = float(np.median([r["t_fit_s"] + r["t_scan_s"] for r in recs]))
    out["cost_s_median_with_basin"] = float(np.median([r["t_fit_s"] + r["t_scan_s"] + r.get("t_basin_s", 0) for r in recs]))
    if "basin_gain" in recs[0] and rel:
        g = np.array([r["basin_gain"] for r in rel])
        out["basin"] = {"frac_gain_gt_20": float((g > 20).mean()), "n_gain_gt_20": int((g > 20).sum()), "max_gain": float(g.max()),
                        "median_dchi2_uncorrected": float(np.median([r["delta_chi2_vs_unpolished_smooth"] for r in rel])),
                        "median_dchi2_corrected": float(np.median([r["delta_chi2"] for r in rel]))}
    return out


def core(root):
    """The Tier-0 populations: clean control, c=60/30/15, multipoles m=4 a=0.01/0.03 and m=3 a=0.03."""
    R = {p: load_scan(root, p) for p in ("no_subhalo", "test_fixed60", "test_fixed15", "test_fixed30", "multipole_m4_a1", "multipole_m4_a3", "multipole_m3_a3")}
    if not R["no_subhalo"]:
        return None
    ops, clean_rel = ops_from_clean(R["no_subhalo"])
    out = {"root": root, "ops": ops, "runs": {p: run_stats(v) for p, v in R.items() if v},
           "clean_dchi2_quantiles": {str(q): float(np.quantile([r["delta_chi2"] for r in clean_rel], q)) for q in (0.5, 0.9, 0.99)},
           "clean_max": float(max(r["delta_chi2"] for r in clean_rel)), "clean_frac_negative": float(np.mean([r["delta_chi2"] < 0 for r in clean_rel]))}
    for name, thr in ops.items():
        o = {"threshold": thr, "fpr_clean": fpr(R["no_subhalo"], thr)}
        for pop, key in (("test_fixed60", "c60"), ("test_fixed30", "c30"), ("test_fixed15", "c15")):
            if R[pop]:
                o[f"completeness_{key}"] = completeness(R[pop], thr); o[f"conservative_{key}"] = completeness(R[pop], thr, conservative=True)
        for pop in ("multipole_m4_a1", "multipole_m4_a3", "multipole_m3_a3"):
            if R[pop]:
                o[f"fpr_{pop}"] = fpr(R[pop], thr)
        if R["test_fixed60"]:
            o["detections_c60"] = detections_c60(R["test_fixed60"], thr)
            for pop, key in (("test_fixed15", "60_to_15"), ("test_fixed30", "60_to_30")):
                if R[pop]:
                    o[f"flip_{key}"] = paired_flip(R["test_fixed60"], R[pop], thr)
            if R["test_fixed30"] and R["test_fixed15"]:
                o["flip_30_to_15"] = paired_flip(R["test_fixed30"], R["test_fixed15"], thr)
        out[name] = o
    gates = []
    for gate in (1.2, 1.5, 2.0, 3.0, 10.0):
        cl = [r for r in R["no_subhalo"] if r["chi2_smooth_per_dof"] < gate]
        if len(cl) < 20:
            continue
        thr = float(np.quantile([r["delta_chi2"] for r in cl], 0.90))
        row = {"gate": gate, "n_clean_pass": len(cl), "frac_clean_pass": len(cl) / len(R["no_subhalo"]), "threshold": thr}
        for pop in ("multipole_m4_a1", "multipole_m4_a3"):
            if R[pop]:
                keep = [r for r in R[pop] if r["chi2_smooth_per_dof"] < gate]
                row[pop] = {"n_pass": len(keep), "frac_pass": len(keep) / len(R[pop]),
                            "fpr_cal": float(np.mean([r["delta_chi2"] >= thr for r in keep])) if keep else None,
                            "fpr_abs": float(np.mean([r["delta_chi2"] >= A_ABS for r in keep])) if keep else None}
        gates.append(row)
    out["gate"] = gates
    return out


def arm(root, pops):
    """A re-rendered arm (shallow, PSF-widened, lens light) calibrated on its own clean control.
    `pops` maps clean/c60/c15/mp -> population folder name under `root`; data population names may differ
    (handled by the caller for localization)."""
    R = {k: load_scan(root, v) for k, v in pops.items()}
    if not R.get("clean"):
        return None
    ops, _ = ops_from_clean(R["clean"])
    out = {"root": root, "ops": ops, "runs": {k: run_stats(v) for k, v in R.items() if v}}
    for name, thr in ops.items():
        o = {"threshold": thr, "fpr_clean": fpr(R["clean"], thr)}
        for key in ("c60", "c15"):
            if R.get(key):
                o[f"completeness_{key}"] = completeness(R[key], thr); o[f"conservative_{key}"] = completeness(R[key], thr, conservative=True)
        if R.get("mp"):
            o["fpr_multipole_m4_a3"] = fpr(R["mp"], thr)
        if R.get("c60") and R.get("c15"):
            o["flip_60_to_15"] = paired_flip(R["c60"], R["c15"], thr)
        out[name] = o
    return out


def add_detections(res_arm, root, popfolder, datapop):
    recs = load_scan(root, popfolder)
    if res_arm and recs:
        for k in ("cal", "abs", "abs100"):
            res_arm[k]["detections_c60"] = detections_c60(recs, res_arm[k]["threshold"], pop=datapop)


def los(root, clean_root):
    clean = load_scan(clean_root, "no_subhalo")
    if not clean:
        return None
    ops, _ = ops_from_clean(clean)
    out = {"root": root, "ops": ops, "fpr_clean": {k: fpr(clean, t) for k, t in ops.items()}, "arms": {}}
    for armname, pop in (("z025", "los_halo_z025"), ("z075", "los_halo_z075")):
        recs = load_scan(root, "los_" + armname)
        if not recs:
            continue
        tr = truths(pop); rel = [r for r in recs if r["reliable_fit"]]
        d = {"z_halo": tr[0]["los_halo"]["z_halo"], **run_stats(recs)}
        for name, thr in ops.items():
            o = {"flag_rate": fpr(recs, thr), "bins": {}}
            for lo, hi in [(9.0, 9.5), (9.5, 10.0), (10.0, 10.5), (10.5, 11.0)]:
                sel = [r for r in rel if lo <= tr[r["index"]]["los_halo"]["log10_M200"] < hi]
                k = sum(1 for r in sel if hit(r, thr)); p, l, h = wilson(k, len(sel))
                o["bins"][f"{lo}-{hi}"] = {"n": len(sel), "k": k, "p": p, "lo": l, "hi": h}
            det = [r for r in rel if hit(r, thr)]
            if det:
                off = np.array([np.hypot(r["best_x"] - tr[r["index"]]["los_halo"]["x"], r["best_y"] - tr[r["index"]]["los_halo"]["y"]) for r in det])
                merr = np.array([r["best_log10_m"] - tr[r["index"]]["los_halo"]["log10_M200"] for r in det])
                o["detections"] = {"n": len(det), "frac_localized": float((off < LOC_ARCSEC).mean()), "median_offset_arcsec": float(np.median(off)),
                                   "mass_error_median": float(np.median(merr)), "n_low": int((merr < 0).sum())}
            d[name] = o
        out["arms"][armname] = d
    return out


def decoys(root, clean_root):
    clean = load_scan(clean_root, "no_subhalo")
    if not clean:
        return None
    ops, _ = ops_from_clean(clean)
    out = {"root": root, "ops": ops, "populations": {}}
    for pop in ("decoy_gaussian_s42_a10", "decoy_gaussian_s43_a10", "decoy_dipole_s42_a10"):
        recs = load_scan(root, pop)
        if recs:
            out["populations"][pop] = {**run_stats(recs), **{name: fpr(recs, thr) for name, thr in ops.items()}}
    return out


def tidal(root, core_root):
    clean = load_scan(core_root, "no_subhalo")
    if not clean:
        return None
    ops, _ = ops_from_clean(clean)
    out = {"root": root, "ops": ops, "populations": {}}
    for pop in ("test_tidal_c15", "test_tidal_c30"):
        recs = load_scan(root, pop)
        if recs:
            out["populations"][pop] = {**run_stats(recs), **{name: completeness(recs, thr) for name, thr in ops.items()}}
    return out


# ------------------------------------------------------------------ tables
def pct(p, nd=0):
    return "--" if p is None or (isinstance(p, float) and np.isnan(p)) else f"{100*p:.{nd}f}\\%"


def table_operating_points(C):
    names = [("cal", "10\\% FPR"), ("abs", f"$\\dchi\\ge{A_ABS:g}$"), ("abs100", f"$\\dchi\\ge{A_ABS_STRICT:g}$")]
    L = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule", " & " + " & ".join(n for _, n in names) + r" \\", r"\midrule",
         "threshold on $\\dchi$ & " + " & ".join(f"${C[k]['threshold']:.1f}$" for k, _ in names) + r" \\",
         "FPR, clean & " + " & ".join(pct(C[k]["fpr_clean"]["p"], 2 if C[k]["fpr_clean"]["p"] < 0.01 else 1) for k, _ in names) + r" \\"]
    for pop, lab in (("multipole_m4_a1", r"FPR, multipole $a_4{=}0.01\,\thetaE$"), ("multipole_m4_a3", r"FPR, multipole $a_4{=}0.03\,\thetaE$"), ("multipole_m3_a3", r"FPR, multipole $a_3{=}0.03\,\thetaE$")):
        if f"fpr_{pop}" in C["cal"]:
            L.append(f"{lab} & " + " & ".join(pct(C[k][f"fpr_{pop}"]["p"]) for k, _ in names) + r" \\")
    L.append(r"\addlinespace[2pt]")
    for lo, hi in BINS[2:]:
        key = f"{lo}-{hi}"
        L.append(f"compl.\\ $c{{=}}60$, $10^{{{lo}}}$--$10^{{{hi}}}\\Msun$ & " + " & ".join(pct(C[k]["completeness_c60"][key]["p"]) for k, _ in names) + r" \\")
    if "completeness_c15" in C["cal"]:
        for lo, hi in BINS[2:]:
            key = f"{lo}-{hi}"
            L.append(f"compl.\\ $c{{=}}15$, $10^{{{lo}}}$--$10^{{{hi}}}\\Msun$ & " + " & ".join(pct(C[k]["completeness_c15"][key]["p"]) for k, _ in names) + r" \\")
        L.append("paired flip $c{=}60\\to15$, lost\\,:\\,gained & " + " & ".join(f"{C[k]['flip_60_to_15']['lost']}\\,:\\,{C[k]['flip_60_to_15']['gained']}" for k, _ in names) + r" \\")
    L.append(r"\addlinespace[2pt]")
    L.append("detections of $c{=}60$ subhalos & " + " & ".join(str(C[k]["detections_c60"]["n"]) for k, _ in names) + r" \\")
    L.append("localized ($\\le2$ px) & " + " & ".join(pct(C[k]["detections_c60"].get("frac_localized"), 1) for k, _ in names) + r" \\")
    L.append("mass error, all detections (dex) & " + " & ".join(f"${C[k]['detections_c60']['mass_error_all']['median']:+.2f}$" if C[k]["detections_c60"]["n"] else "--" for k, _ in names) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L) + "\n"


def table_los(S):
    a, b = S["arms"].get("z025"), S["arms"].get("z075")
    if not a or not b:
        return None
    L = [r"\begin{tabular}{@{}lcc@{}}", r"\toprule", r" & foreground & background \\", r"line-of-sight halo redshift & $z=0.25$ & $z=0.75$ \\", r"\midrule",
         f"flagged at $\\dchi\\ge{A_ABS:g}$ & {pct(a['abs']['flag_rate']['p'])} & {pct(b['abs']['flag_rate']['p'])} \\\\",
         f"flagged at the 10\\% threshold & {pct(a['cal']['flag_rate']['p'])} & {pct(b['cal']['flag_rate']['p'])} \\\\",
         f"macro fits rejected (of {a['n']}) & {a['n_unreliable']} & {b['n_unreliable']} \\\\"]
    for k in a["abs"]["bins"]:
        lo, hi = k.split("-")
        L.append(f"\\quad flagged at $\\dchi\\ge{A_ABS:g}$, $10^{{{lo}}}$--$10^{{{hi}}}\\Msun$ & {pct(a['abs']['bins'][k]['p'])} & {pct(b['abs']['bins'][k]['p'])} \\\\")
    L.append(r"\addlinespace[2pt]")
    da, db = a["abs"].get("detections"), b["abs"].get("detections")
    if da and db:
        L.append(f"localized ($\\le2$ px) & {pct(da['frac_localized'])} & {pct(db['frac_localized'])} \\\\")
        L.append(f"median offset from the halo & {da['median_offset_arcsec']:.2f}$''$ & {db['median_offset_arcsec']:.2f}$''$ \\\\")
        L.append(f"inferred minus true $\\log_{{10}}M$ (dex) & ${da['mass_error_median']:+.2f}$ & ${db['mass_error_median']:+.2f}$ \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L) + "\n"


def table_gate(C):
    L = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
         r"$\chi^2/{\rm dof}$ gate & \multicolumn{2}{c}{lenses passing} & \multicolumn{2}{c}{FPR, $a_4{=}0.03\,\thetaE$} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}", f" & clean & $a_4{{=}}0.03$ & 10\\% cal. & $\\dchi\\ge{A_ABS:g}$ \\\\", r"\midrule"]
    for r in C["gate"]:
        mp = r.get("multipole_m4_a3")
        if not mp:
            continue
        if mp["n_pass"] < 10:
            L.append(f"$<{r['gate']:g}$ & {100*r['frac_clean_pass']:.0f}\\% & {100*mp['frac_pass']:.1f}\\% & \\multicolumn{{2}}{{c}}{{{mp['n_pass']} lenses, not scored}} \\\\")
        else:
            L.append(f"$<{r['gate']:g}$ & {100*r['frac_clean_pass']:.0f}\\% & {100*mp['frac_pass']:.0f}\\% & {pct(mp['fpr_cal'])} & {pct(mp['fpr_abs'])} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L) + "\n"


def main():
    res = {"core": core(A_JOINT_FULL), "core_frozen": core(A_FROZEN_FULL),
           "los": los(A_JOINT_LOS, A_JOINT_FULL),
           "shallow": arm(A_JOINT_SHALLOW, {"clean": "no_subhalo", "c60": "test_fixed60", "c15": "test_fixed15", "mp": "multipole_m4_a3"}),
           "psf015": arm(A_JOINT_PSF, {"clean": "no_subhalo", "c60": "test_fixed60"}),
           "psf015_frozen": arm(A_FROZEN_PSF, {"clean": "no_subhalo", "c60": "test_fixed60", "c15": "test_fixed15", "mp": "multipole_m4_a3"}),
           "lenslight_single": arm(A_JOINT_LL, {"clean": "lenslight_no_subhalo", "c60": "lenslight_fixed60", "mp": "lenslight_multipole_m4_a3"}),
           "lenslight_double": arm(A_JOINT_LL2, {"clean": "lenslight_no_subhalo", "c60": "lenslight_fixed60", "mp": "lenslight_multipole_m4_a3"}),
           "decoys": decoys(A_JOINT_DECOY, A_JOINT_FULL), "tidal": tidal(A_JOINT_TIDAL, A_JOINT_FULL),
           "freepos": arm("results/baseline_a_joint_freepos", {"clean": "no_subhalo", "c60": "test_fixed60", "c15": "test_fixed15", "mp": "multipole_m4_a3"})}
    add_detections(res["freepos"], "results/baseline_a_joint_freepos", "test_fixed60", "test_fixed60")
    if res["freepos"]:
        # the free-position run's false alarms on the multipole population: are they still on the eight spokes?
        recs = load_scan("results/baseline_a_joint_freepos", "multipole_m4_a3") or []
        moved = [np.hypot(r["best_x"] - min(r["cells"], key=lambda c: c["chi2_joint"])["node_x"], r["best_y"] - min(r["cells"], key=lambda c: c["chi2_joint"])["node_y"]) for r in recs if r["reliable_fit"]]
        res["freepos"]["multipole_best_cell_moved_from_node_arcsec_median"] = float(np.median(moved)) if moved else None
    # referee round 19: the c-M template on the c-M-drawn population (same lenses as test_fixed60), mass error at the
    # absolute threshold, which needs no calibration; and the line-of-sight arms with the perturber position free
    cm = load_scan(A_JOINT_CM, "test_cdm")
    res["cm_template_on_cm_population"] = None
    if cm:
        res["cm_template_on_cm_population"] = {"root": A_JOINT_CM, **run_stats(cm),
                                               "abs": detections_c60(cm, A_ABS, pop="test_cdm"), "cal": detections_c60(cm, res["core"]["cal"]["threshold"], pop="test_cdm")}
        tr = truths("test_cdm"); cs = [tr[r["index"]]["subhalo"]["concentration"] for r in cm if r["has_subhalo"]]
        res["cm_template_on_cm_population"]["true_concentration_median"] = float(np.median(cs))
    res["los_freepos"] = los("results/baseline_a_joint_freepos", A_JOINT_FULL)
    if res["los_freepos"] and not res["los_freepos"]["arms"]:
        res["los_freepos"] = None
    # gate bookkeeping: how many lenses the basin/joint re-gate and the multi-start rescue recovered
    gate = {}
    for pop in ("no_subhalo", "test_fixed60", "test_fixed15", "test_fixed30", "multipole_m4_a1", "multipole_m4_a3", "multipole_m3_a3"):
        recs = load_scan(A_JOINT_FULL, pop) or []
        if recs and "reliable_fit_initial" in recs[0]:
            g = {"n": len(recs), "gated_initial": sum(1 for r in recs if not r["reliable_fit_initial"]), "gated_final": sum(1 for r in recs if not r["reliable_fit"]),
                 "rescued_basin": sum(1 for r in recs if r.get("gate_rescued_by") == "basin"), "rescued_joint": sum(1 for r in recs if r.get("gate_rescued_by") == "joint"),
                 "rescued_multistart": sum(1 for r in recs if r.get("rescued_record"))}
            sub = [r for r in recs if r["has_subhalo"] and not r["reliable_fit"]]
            if sub:
                g["excluded_subhalo_by_bin"] = {f"{lo}-{hi}": sum(1 for r in sub if lo <= r["log10_M200_true"] < hi) for lo, hi in BINS}
            gate[pop] = g
    res["gate"] = gate
    add_detections(res["shallow"], A_JOINT_SHALLOW, "test_fixed60", "test_shallow_fixed60")
    add_detections(res["psf015"], A_JOINT_PSF, "test_fixed60", "psf015_fixed60")
    add_detections(res["psf015_frozen"], A_FROZEN_PSF, "test_fixed60", "psf015_fixed60")
    add_detections(res["lenslight_single"], A_JOINT_LL, "lenslight_fixed60", "lenslight_fixed60")
    add_detections(res["lenslight_double"], A_JOINT_LL2, "lenslight_fixed60", "lenslight_fixed60")
    (ROOT / "results/joint_scan.json").write_text(json.dumps(res, indent=1))
    T = ROOT / "paper/tables"; T.mkdir(exist_ok=True)
    C = res["core"]
    if C and "completeness_c60" in C["cal"]:
        (T / "joint_operating_points.tex").write_text(table_operating_points(C))
        if C["gate"] and any("multipole_m4_a3" in g for g in C["gate"]):
            (T / "gate_fpr.tex").write_text(table_gate(C))
    if res["los"] and len(res["los"]["arms"]) == 2:
        t = table_los(res["los"])
        if t:
            (T / "los.tex").write_text(t)
    if C:
        print(f"== joint core ({A_JOINT_FULL}): runs", {p: (v['n'], v['n_unreliable'], v['n_truncated']) for p, v in C["runs"].items()})
        print("   clean quantiles", C["clean_dchi2_quantiles"], "max", round(C["clean_max"], 1), "frac<0", C["clean_frac_negative"])
        for k in ("cal", "abs", "abs100"):
            o = C[k]
            line = f"  {k:6s} thr {o['threshold']:6.1f}  clean FPR {100*o['fpr_clean']['p']:.2f}%"
            for pop in ("multipole_m4_a1", "multipole_m4_a3", "multipole_m3_a3"):
                if f"fpr_{pop}" in o:
                    line += f"  {pop[10:]} {100*o[f'fpr_{pop}']['p']:.1f}%"
            print(line)
            for key in ("c60", "c30", "c15"):
                if f"completeness_{key}" in o:
                    print(f"         {key} compl", " ".join(f"{100*v['p']:3.0f}" for v in o[f"completeness_{key}"].values()),
                          "| conservative", " ".join(f"{100*v['p']:3.0f}" for v in o[f"conservative_{key}"].values()))
            d = o.get("detections_c60", {})
            if d.get("n"):
                print(f"         det {d['n']}, localized {100*d['frac_localized']:.1f}% (median offset {d['median_offset_arcsec']:.2f}), mass err all {d['mass_error_all']['median']:+.2f} (low {d['mass_error_all']['n_low']}/{d['n']}, within 0.5 dex {d['mass_error_all']['n_within_0p5']}), localized {d['mass_error_localized']}")
            for fk in ("flip_60_to_15", "flip_60_to_30", "flip_30_to_15"):
                if fk in o:
                    print(f"         {fk}: {o[fk]}")
        print("   gates:", [(g["gate"], round(g["frac_clean_pass"], 2), g.get("multipole_m4_a3", {}).get("frac_pass"), g.get("multipole_m4_a3", {}).get("fpr_cal"), g.get("multipole_m4_a3", {}).get("fpr_abs")) for g in C["gate"]])
        print("   runs:", json.dumps({p: {kk: (round(vv, 1) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != "basin"} for p, v in C["runs"].items()}))
    for key in ("cm_template_on_cm_population", "los_freepos", "gate"):
        if res.get(key):
            print(f"== {key}:", json.dumps(res[key], indent=None)[:1500])
    for key in ("los", "shallow", "psf015", "psf015_frozen", "lenslight_single", "lenslight_double", "decoys", "tidal", "freepos"):
        if res[key]:
            print(f"== {key}: ops {{k: round(v,1) for k,v in res[key]['ops'].items()}}")
            print(json.dumps(res[key], indent=None)[:1800])
    print("wrote results/joint_scan.json")


if __name__ == "__main__":
    main()
