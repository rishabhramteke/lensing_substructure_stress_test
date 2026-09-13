"""Referee round 6, points 1 and 3: a scan statistic with a null floor, and confounder
false positives conditional on a goodness-of-fit gate.

Point 1 -- null floor. Family A's grid has 72 hypotheses, all with a subhalo, so its
statistic max_grid (chi2_smooth - chi2_cell) can be negative: the best of 72 perturbers
still makes the fit worse. Forcing a 10%-FPR calibration onto that distribution puts the
threshold at a negative Delta chi^2, so some "detections" are lenses the perturber hurt.
Adding a zero-mass hypothesis to the grid fixes this exactly: a zero-mass TNFW is the
smooth model, contributing Delta chi^2 = 0 identically, so the floored statistic is
    Delta chi^2_floor = max(Delta chi^2, 0),
computable from the stored scans without re-fitting. This script recalibrates on it.

Point 3 -- gate-conditional confounder FPR. Every published pipeline checks the smooth
fit before looking for substructure. We recompute the multipole false-positive rate over
only those lenses whose smooth fit passes a chi^2/dof gate, with the threshold recalibrated
on clean lenses passing the same gate.

    python scripts/evaluate_null_floor.py
-> results/null_floor.json, paper/tables/null_floor.tex, paper/tables/gate_fpr.tex
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BINS = [(8.0, 8.5), (8.5, 9.0), (9.0, 9.5), (9.5, 10.0), (10.0, 10.5), (10.5, 11.0)]
LOC = 0.16
GATES = [1.2, 1.5, 2.0, 3.0, 10.0]
truth60 = [json.loads(l) for l in open(ROOT / "data/test_fixed60/truth.jsonl")]


def load(root, pop):
    p = ROOT / root / pop / "scan_results.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else None


def wilson(k, n, z=1.0):
    if not n:
        return (np.nan,) * 3
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c, c - h, c + h


def completeness(recs, thr, strict):
    out = {}
    pos = [r for r in recs if r["reliable_fit"] and r["has_subhalo"]]
    for lo, hi in BINS:
        sel = [r for r in pos if lo <= r["log10_M200_true"] < hi]
        k = sum(1 for r in sel if (r["delta_chi2"] > thr if strict else r["delta_chi2"] >= thr))
        out[f"{lo}-{hi}"] = {"n": len(sel), "k": k, "p": (k / len(sel)) if sel else None}
    return out


def evaluate_root(root):
    """Both statistics on one set of runs: raw (10%-FPR calibration) and floored (Delta chi^2 > 0)."""
    R = {p: load(root, p) for p in ("no_subhalo", "test_fixed60", "test_fixed15", "multipole_m4_a1", "multipole_m4_a3")}
    if not R["no_subhalo"]:
        return None
    clean = [r for r in R["no_subhalo"] if r["reliable_fit"]]
    d_clean = np.array([r["delta_chi2"] for r in clean])
    out = {"n_clean": len(clean), "raw": {}, "floored": {}}
    thr_raw = float(np.quantile(d_clean, 0.90))
    out["raw"]["threshold"] = thr_raw
    out["raw"]["fpr_clean"] = float(np.mean(d_clean >= thr_raw))
    # floored: the zero-mass hypothesis makes the clean distribution degenerate at 0,
    # so a 10%-FPR calibration does not exist; the natural operating point is > 0.
    out["floored"]["frac_clean_positive"] = float(np.mean(d_clean > 0))
    out["floored"]["quantile90_floored"] = float(np.quantile(np.clip(d_clean, 0, None), 0.90))
    out["floored"]["threshold"] = 0.0
    out["floored"]["fpr_clean"] = float(np.mean(d_clean > 0))
    out["floored"]["max_clean"] = float(d_clean.max())
    for key, thr, strict in (("raw", thr_raw, False), ("floored", 0.0, True)):
        for pop in ("test_fixed60", "test_fixed15"):
            if R[pop]:
                out[key][f"completeness_{pop}"] = completeness(R[pop], thr, strict)
        for pop in ("multipole_m4_a1", "multipole_m4_a3"):
            if R[pop]:
                rel = [r for r in R[pop] if r["reliable_fit"]]
                k = sum(1 for r in rel if (r["delta_chi2"] > thr if strict else r["delta_chi2"] >= thr))
                out[key][f"fpr_{pop}"] = {"n": len(rel), "k": k, "p": k / len(rel)}
        # localization among detections on the c=60 population
        if R["test_fixed60"]:
            det = [r for r in R["test_fixed60"] if r["reliable_fit"] and r["has_subhalo"]
                   and (r["delta_chi2"] > thr if strict else r["delta_chi2"] >= thr)]
            off = np.array([np.hypot(r["best_x"] - truth60[r["index"]]["subhalo"]["x"],
                                     r["best_y"] - truth60[r["index"]]["subhalo"]["y"]) for r in det])
            out[key]["c60_detections"] = {"n": len(det), "n_localized": int((off < LOC).sum()),
                                          "frac_localized": float((off < LOC).mean()) if len(off) else None,
                                          "n_negative_dchi2": sum(1 for r in det if r["delta_chi2"] < 0)}
    # paired concentration flip on the same lens indices
    if R["test_fixed60"] and R["test_fixed15"]:
        a = {r["index"]: r for r in R["test_fixed60"] if r["reliable_fit"] and r["has_subhalo"]}
        b = {r["index"]: r for r in R["test_fixed15"] if r["reliable_fit"] and r["has_subhalo"]}
        common = sorted(set(a) & set(b))
        for key, thr, strict in (("raw", thr_raw, False), ("floored", 0.0, True)):
            f = (lambda r: r["delta_chi2"] > thr) if strict else (lambda r: r["delta_chi2"] >= thr)
            lost = sum(1 for i in common if f(a[i]) and not f(b[i]))
            gained = sum(1 for i in common if f(b[i]) and not f(a[i]))
            out[key]["paired_flip"] = {"n_pairs": len(common), "lost": lost, "gained": gained}
    return out


def gate_analysis(root, joint_root=None):
    """Multipole FPR restricted to lenses whose smooth fit passes a chi^2/dof gate,
    with the threshold recalibrated on clean lenses passing the same gate."""
    rows = []
    for gate in GATES:
        clean = [r for r in (load(root, "no_subhalo") or []) if r["chi2_smooth_per_dof"] < gate]
        if len(clean) < 20:
            continue
        d = np.array([r["delta_chi2"] for r in clean])
        thr = float(np.quantile(d, 0.90))
        row = {"gate": gate, "n_clean_pass": len(clean), "frac_clean_pass": len(clean) / len(load(root, "no_subhalo")),
               "threshold": thr, "floor_fpr_clean": float(np.mean(d > 0))}
        for pop in ("multipole_m4_a1", "multipole_m4_a3"):
            recs = load(root, pop)
            if not recs:
                continue
            keep = [r for r in recs if r["chi2_smooth_per_dof"] < gate]
            row[pop] = {"n_pass": len(keep), "frac_pass": len(keep) / len(recs),
                        "fpr_at_10pct": float(np.mean([r["delta_chi2"] >= thr for r in keep])) if keep else None,
                        "fpr_floored": float(np.mean([r["delta_chi2"] > 0 for r in keep])) if keep else None,
                        "fpr_at_20": float(np.mean([r["delta_chi2"] >= 20 for r in keep])) if keep else None}
        rows.append(row)
    if joint_root:
        for gate in GATES:
            clean = [r for r in (load(joint_root, "no_subhalo") or []) if r["chi2_smooth_per_dof"] < gate]
            mp = [r for r in (load(joint_root, "multipole_m4_a3") or []) if r["chi2_smooth_per_dof"] < gate]
            if len(clean) < 10 or not mp:
                continue
            thr = float(np.quantile([r["delta_chi2"] for r in clean], 0.90))
            rows.append({"gate": gate, "joint": True, "n_clean_pass": len(clean), "n_mp_pass": len(mp),
                         "frac_mp_pass": len(mp) / len(load(joint_root, "multipole_m4_a3")),
                         "threshold": thr,
                         "fpr_at_10pct": float(np.mean([r["delta_chi2"] >= thr for r in mp])),
                         "fpr_at_20": float(np.mean([r["delta_chi2"] >= 20 for r in mp]))})
    return rows


def main():
    res = {"subsample_300": evaluate_root("results/baseline_a"),
           "full_1000": evaluate_root("results/baseline_a_full"),
           "gate": {"subsample_300": gate_analysis("results/baseline_a", "results/baseline_a_joint_basin_c15"),
                    "full_1000": gate_analysis("results/baseline_a_full")}}
    (ROOT / "results/null_floor.json").write_text(json.dumps(res, indent=2))

    # ---- ONE operating-point table on ONE sample (the full populations), replacing the two that
    # compared different operating points on different samples and so quoted different clean rates
    R = {pop: load("results/baseline_a_full", pop) for pop in ("no_subhalo", "test_fixed60", "test_fixed15", "multipole_m4_a1", "multipole_m4_a3")}
    clean = [r for r in R["no_subhalo"] if r["reliable_fit"]]
    dcl = np.array([r["delta_chi2"] for r in clean])
    OPS = [("10\\% FPR", float(np.quantile(dcl, 0.90)), False),
           ("$\\dchi>0$", 0.0, True), ("$\\dchi>20$", 20.0, False), ("$\\dchi>100$", 100.0, False)]
    def hit(r, thr, strict):
        return r["delta_chi2"] > thr if strict else r["delta_chi2"] >= thr
    pos60 = [r for r in R["test_fixed60"] if r["reliable_fit"] and r["has_subhalo"]]
    ops = []
    for lab, thr, strict in OPS:
        det = [r for r in pos60 if hit(r, thr, strict)]
        off = np.array([np.hypot(r["best_x"] - truth60[r["index"]]["subhalo"]["x"],
                                 r["best_y"] - truth60[r["index"]]["subhalo"]["y"]) for r in det])
        merr = np.array([r["best_log10_m"] - truth60[r["index"]]["subhalo"]["log10_M200"] for r in det])
        loc = off < LOC
        comp = []
        for lo, hi in BINS[2:]:
            sel = [r for r in pos60 if lo <= r["log10_M200_true"] < hi]
            comp.append(100 * np.mean([hit(r, thr, strict) for r in sel]) if sel else np.nan)
        ops.append({"label": lab, "threshold": thr,
                    "fpr_clean": float(np.mean([hit(r, thr, strict) for r in clean])),
                    "fpr_mp_a1": float(np.mean([hit(r, thr, strict) for r in R["multipole_m4_a1"] if r["reliable_fit"]])),
                    "fpr_mp_a3": float(np.mean([hit(r, thr, strict) for r in R["multipole_m4_a3"] if r["reliable_fit"]])),
                    "n_detected": len(det), "n_positive": len(pos60),
                    "frac_negative_dchi2": float(np.mean([r["delta_chi2"] < 0 for r in det])) if det else 0.0,
                    "frac_localized": float(loc.mean()) if len(off) else None,
                    "median_mass_error_localized": float(np.median(merr[loc])) if loc.any() else None,
                    "completeness_bins": comp})
    res["operating_points"] = ops
    L = [r"\begin{tabular}{@{}l" + "c" * len(ops) + "@{}}", r"\toprule",
         " & " + " & ".join(o["label"] for o in ops) + r" \\", r"\midrule",
         "threshold on $\dchi$ & " + " & ".join(("none" if o["label"].startswith("10") and False else f"${o['threshold']:.1f}$") for o in ops) + r" \\",
         "FPR, clean & " + " & ".join(f"{100*o['fpr_clean']:.2g}\%" for o in ops) + r" \\",
         "of detections with $\dchi<0$ & " + " & ".join(f"{100*o['frac_negative_dchi2']:.0f}\%" for o in ops) + r" \\",
         r"\addlinespace[2pt]",
         r"FPR, multipole $a_m{=}0.01\,\thetaE$ & " + " & ".join(f"{100*o['fpr_mp_a1']:.0f}\%" for o in ops) + r" \\",
         r"FPR, multipole $a_m{=}0.03\,\thetaE$ & " + " & ".join(f"{100*o['fpr_mp_a3']:.0f}\%" for o in ops) + r" \\",
         r"\addlinespace[2pt]",
         "compl.\\ $c{=}60$, 9--9.5 & " + " & ".join(f"{o['completeness_bins'][0]:.0f}\%" for o in ops) + r" \\",
         "\\quad 9.5--10 / 10--10.5 & " + " & ".join(f"{o['completeness_bins'][1]:.0f} / {o['completeness_bins'][2]:.0f}\%" for o in ops) + r" \\",
         "localized ($\le2$ px) & " + " & ".join((f"{100*o['frac_localized']:.0f}\%" if o["frac_localized"] is not None else "--") for o in ops) + r" \\",
         "mass error, loc.\\ (dex) & " + " & ".join((f"${o['median_mass_error_localized']:+.1f}$" if o["median_mass_error_localized"] is not None else "--") for o in ops) + r" \\",
         r"\bottomrule", r"\end{tabular}"]
    (ROOT / "paper/tables/operating_points.tex").write_text("\n".join(L) + "\n")

    # ---- Table: raw vs floored statistic, full populations
    f = res["full_1000"]
    def pc(d, k):
        c = d[k]
        return " / ".join("--" if c[f"{lo}-{hi}"]["p"] is None else f"{100*c[f'{lo}-{hi}']['p']:.0f}" for lo, hi in BINS[2:])
    lines = [r"\begin{tabular}{@{}lcc@{}}", r"\toprule",
             r" & 72 hypotheses & $+$ zero-mass \\", r"\midrule",
             f"threshold at 10\\% FPR & ${f['raw']['threshold']:.1f}$ & none exists \\\\",
             f"clean lenses above threshold & 10\\% & {100*f['floored']['fpr_clean']:.1f}\\% \\\\",
             f"of which $\\dchi<0$ & {f['raw']['c60_detections']['n_negative_dchi2']}/{f['raw']['c60_detections']['n']} & 0 \\\\",
             r"\addlinespace[2pt]",
             f"compl.\\ $c{{=}}60$, 4 bins $>10^{{9}}\\Msun$ & {pc(f['raw'],'completeness_test_fixed60')}\\% & {pc(f['floored'],'completeness_test_fixed60')}\\% \\\\",
             f"compl.\\ $c{{=}}15$, same bins & {pc(f['raw'],'completeness_test_fixed15')}\\% & {pc(f['floored'],'completeness_test_fixed15')}\\% \\\\",
             f"paired flip, lost\\,:\\,gained & {f['raw']['paired_flip']['lost']}\\,:\\,{f['raw']['paired_flip']['gained']} & {f['floored']['paired_flip']['lost']}\\,:\\,{f['floored']['paired_flip']['gained']} \\\\",
             r"\addlinespace[2pt]",
             f"FPR, multipole $a_m{{=}}0.01\\,\\thetaE$ & {100*f['raw']['fpr_multipole_m4_a1']['p']:.0f}\\% & {100*f['floored']['fpr_multipole_m4_a1']['p']:.0f}\\% \\\\",
             f"FPR, multipole $a_m{{=}}0.03\\,\\thetaE$ & {100*f['raw']['fpr_multipole_m4_a3']['p']:.0f}\\% & {100*f['floored']['fpr_multipole_m4_a3']['p']:.0f}\\% \\\\",
             f"localized ($\\le2$ px) & {100*f['raw']['c60_detections']['frac_localized']:.0f}\\% & {100*f['floored']['c60_detections']['frac_localized']:.0f}\\% \\\\",
             r"\bottomrule", r"\end{tabular}"]
    (ROOT / "paper/tables/null_floor.tex").write_text("\n".join(lines) + "\n")

    # ---- Table: gate-conditional confounder FPR (full populations, frozen scan)
    g = [r for r in res["gate"]["full_1000"] if not r.get("joint")]
    lines = [r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
             r"$\chi^2/{\rm dof}$ gate & \multicolumn{2}{c}{lenses passing} & \multicolumn{3}{c}{FPR, $a_m{=}0.03\,\thetaE$} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}",
             r" & clean & $a_m{=}0.03$ & 10\% cal. & $\dchi{>}0$ & $\dchi{>}20$ \\", r"\midrule"]
    for r in g:
        mp = r.get("multipole_m4_a3")
        if not mp or mp["n_pass"] < 10:
            lines.append(f"$<{r['gate']:g}$ & {100*r['frac_clean_pass']:.0f}\\% & {100*mp['frac_pass']:.1f}\\% & \\multicolumn{{3}}{{c}}{{{mp['n_pass']} lenses, not scored}} \\\\" if mp else "")
            continue
        lines.append(f"$<{r['gate']:g}$ & {100*r['frac_clean_pass']:.0f}\\% & {100*mp['frac_pass']:.0f}\\% & "
                     f"{100*mp['fpr_at_10pct']:.0f}\\% & {100*mp['fpr_floored']:.0f}\\% & {100*mp['fpr_at_20']:.0f}\\% \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "paper/tables/gate_fpr.tex").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "gate"}, indent=1)[:2500])
    print("\n== gate rows (full):")
    for r in g:
        mp = r.get("multipole_m4_a3", {})
        print(f"  gate<{r['gate']}: clean pass {100*r['frac_clean_pass']:.0f}% thr {r['threshold']:.1f} | mp_a3 pass {mp.get('n_pass')} ({100*mp.get('frac_pass',0):.1f}%) fpr10 {mp.get('fpr_at_10pct')} fpr0 {mp.get('fpr_floored')} fpr20 {mp.get('fpr_at_20')}")
    print("== gate rows (joint, 300-subsample runs):")
    for r in res["gate"]["subsample_300"]:
        if r.get("joint"):
            print(f"  gate<{r['gate']}: clean {r['n_clean_pass']} mp {r['n_mp_pass']} ({100*r['frac_mp_pass']:.0f}%) thr {r['threshold']:.1f} fpr10 {r['fpr_at_10pct']:.2f} fpr20 {r['fpr_at_20']:.2f}")


if __name__ == "__main__":
    main()
