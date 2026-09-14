"""Emit the five tables that were still typed by hand in main.tex.

A number audit found that every one of the twelve generated tables was correct and that every
error in the paper sat in one of the tables written out by hand: a baseline false-positive rate
that matched no run, a bound quoted from one seed set instead of two, two cells left as stray
commas, one cell never filled in, two wrong decoy cells with a reversed pair, and a stale family
name. This script closes that failure mode by reading the same results files the prose is checked
against, so the tables cannot drift from the runs again.

The descriptive strings (row labels, the reproduction ledger's "specified in the paper" column)
stay here as literals -- they are prose, not measurements. Every number comes from a file.

    python scripts/make_paper_tables.py
-> paper/tables/{decoy,joint,lenslight,ledger,mass_mechanism}.tex
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/tables"


# --------------------------------------------------------------------------- helpers
def jload(rel):
    return json.loads((ROOT / rel).read_text())


def scans(rel):
    """All scan records for one population, plus the reliable-fit subset."""
    recs = [json.loads(l) for l in open(ROOT / rel / "scan_results.jsonl")]
    return recs, [r for r in recs if r["reliable_fit"]]


def stat(rel):
    _, rel_recs = scans(rel)
    return np.array([r["delta_chi2"] for r in rel_recs], float)


def rnd(x, nd=0):
    """Round half away from zero, the convention a reader checking a table expects.

    Python rounds halves to even, so a rate of exactly 32.5% prints as 32, which reads as an
    error against the 32.5 in the results file.
    """
    from decimal import Decimal, ROUND_HALF_UP
    q = Decimal(1).scaleb(-nd)
    return f"{Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP):.{nd}f}"


def pct(x, nd=0):
    """A percentage. Exact zero is "0" at any precision, so a row never mixes 0 with 0.00."""
    if x is None:
        return "--"
    return "0" if x == 0 else rnd(100 * x, nd)


def tabular(colspec, header, rows, rules=()):
    """One booktabs tabular. `rows` are already-formatted "a & b & c" strings."""
    out = [f"\\begin{{tabular}}{{{colspec}}}", r"\toprule"]
    out += [h + r" \\" for h in header]
    out.append(r"\midrule")
    for i, r in enumerate(rows):
        if i in rules:
            out.append(r"\midrule")
        out.append(r + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def write(name, text):
    (OUT / f"{name}.tex").write_text(text)
    print(f"  wrote paper/tables/{name}.tex")


# --------------------------------------------------------------------------- tab:decoy
def decoy():
    """False positives on non-physical flux decoys, at each family's own 10%-FPR threshold.

    Gaussian ran on two seeds, the dipole on one, so the Gaussian cells carry a pair. Family B's
    seed-43 arm was scored after its summary file was written, so it is recomputed here from the
    stored runs using the threshold that file records.
    """
    g42, g43 = jload("results/noise_decoy_control/results.json"), jload("results/noise_decoy_control_seed43/results.json")
    dip = jload("results/noise_decoy_control_dipole/results.json")
    fb = jload("results/noise_decoy_control_familyB/results.json")
    thr_b = fb["config"]["threshold_peak_abs_dkappa_lam1e5"]
    amps = ("3.0", "6.0", "10.0")

    def b_seed43(a):
        _, rel = scans(f"results/baseline_b_decoy/fitted/decoy_gaussian_s43_a{a}")
        return float(np.mean(np.array([r["delta_chi2"] for r in rel], float) > thr_b))

    rows = [
        "Gaussian & A scan & " + " & ".join(f"{pct(g42[a]['a_fpr'])}, {pct(g43[a]['a_fpr'])}\\%" for a in amps),
        r" & B linear $\delta\psi$\ & " + " & ".join(
            f"{pct(fb['gaussian'][a]['b_fpr'])}, {pct(b_seed43(a.split('.')[0]))}\\%" for a in amps),
        " & C U-Net & " + " & ".join(f"{pct(g42[a]['unet_fpr'])}, {pct(g43[a]['unet_fpr'])}\\%" for a in amps),
        "dipole & A scan & " + " & ".join(f"{pct(dip[a]['a_fpr'])}\\%" for a in amps),
        r" & B linear $\delta\psi$\ & " + " & ".join(f"{pct(fb['dipole'][a]['b_fpr'])}\\%" for a in amps),
        " & C U-Net & " + " & ".join(f"{pct(dip[a]['unet_fpr'])}\\%" for a in amps),
    ]
    write("decoy", tabular("@{}llccc@{}", [r"decoy & family & $3\sigma$ & $6\sigma$ & $10\sigma$"], rows))


# --------------------------------------------------------------------------- tab:joint
def joint():
    """The fit-then-scan shortcut removed: macro frozen versus re-fitted jointly per grid cell.

    The joint column is the macro-basin-controlled run (`baseline_a_joint_basin_c15`), as the
    caption states; the frozen column is the same lenses scored from `results/baseline_a`.
    """
    s = jload("results/baseline_a_joint_basin_c15/summary_vs_frozen.json")["c15"]
    F, J = s["frozen_same_lenses"], s["joint"]
    # Per-lens cost, median CPU seconds, read from each record's own timers rather than from
    # elapsed wall-clock: the two runs used different worker counts and the frozen run did not
    # record its own, so a manifest ratio would not be like for like. The basin control's
    # t_basin_s is excluded, being a separate diagnostic rather than part of the scan.
    def cpu_s(rel):
        recs = [json.loads(l) for l in open(ROOT / rel / "scan_results.jsonl")]
        return float(np.median([(r.get("t_fit_s") or 0) + (r.get("t_scan_s") or 0) for r in recs]))

    cost_f = cpu_s("results/baseline_a/no_subhalo")
    cost_j = cpu_s("results/baseline_a_joint_basin_c15/no_subhalo")
    secs = lambda t: f"{t:.1f}" if t < 10 else f"{t:.0f}"

    def two(fn):
        return f"{fn(F)} & {fn(J)}"

    rows = [
        f"cost per lens & {secs(cost_f)}~s & {secs(cost_j)}~s",
        r"threshold on $\dchi$ at 10\% FPR & " + two(lambda d: f"${d['threshold_10pct_fpr']:+.1f}$"),
        r"FPR, multipole $a_m{=}0.03\,\thetaE$, 10\% FPR & " + two(lambda d: pct(d["fpr_multipole_a3"]["at_10pct"]) + r"\%"),
        r"\quad at $\dchi>20$ & " + two(lambda d: pct(d["fpr_multipole_a3"]["at_20"]) + r"\%"),
        r"\quad at $\dchi>100$ & " + two(lambda d: pct(d["fpr_multipole_a3"]["at_100"]) + r"\%"),
        r"FPR, clean, at $\dchi>20$ / $>100$ & " + two(
            lambda d: f"{pct(d['fpr_clean']['at_20'])}\\% / {pct(d['fpr_clean']['at_100'])}\\%"),
        r"median $\dchi$, multipole lenses & " + two(lambda d: f"{d['multipole_median_dchi2']:.0f}"),
        f"detected, of subhalo-bearing ($c=60$) & " + two(
            lambda d: f"{d['c60']['n_detected']}/{d['c60']['n_positive_reliable']}"),
        r"of which localized ($\le2$ px) & " + two(lambda d: pct(d["c60"]["frac_localized_2px"]) + r"\%"),
        r"mass error of localized (dex) & " + two(
            lambda d: f"${d['c60']['mass_error_localized']['median']:+.1f}$ "
                      f"({d['c60']['mass_error_localized']['n_low']}/{d['c60']['mass_error_localized']['n']} low)"),
    ]
    write("joint", tabular("@{}lcc@{}", [r" & macro frozen & macro re-fitted"], rows, rules={7}))


# --------------------------------------------------------------------------- tab:lenslight
def lenslight():
    """Lens light added and subtracted two ways, against the lens-light-free Tier-0 run.

    The "none" column is the Tier-0 reference the experiment was built on: Family A from
    `results/baseline_a` (the $c=15$-template 300-lens scan -- identified by its rejected-fit
    count and threshold, which the experiment reuses), Family B from the $\\lambda=10^5$ arm of
    the Family-B baseline, the U-Net from the four-seed aggregate.
    """
    s1, s2 = jload("results/lens_light_experiment.json"), jload("results/lens_light_experiment_s8.json")

    # --- Family A reference, recomputed from the scan it comes from
    a_clean_recs, a_clean_rel = scans("results/baseline_a/no_subhalo")
    a_clean = np.array([r["delta_chi2"] for r in a_clean_rel], float)
    a_thr = float(np.quantile(a_clean, 0.90))
    a_mp = stat("results/baseline_a/multipole_m4_a3")
    _, a_pos = scans("results/baseline_a/test_fixed60")
    a_pos = [r for r in a_pos if r.get("has_subhalo")]

    def a_compl(lo, hi):
        sel = [r for r in a_pos if lo <= r["log10_M200_true"] < hi]
        return float(np.mean([r["delta_chi2"] > a_thr for r in sel]))

    # --- Family B and U-Net references
    b_ref = jload("results/baseline_b/summary_family_b.json")["fitted"]["by_lambda"]["1e5"]
    u_ref = jload("results/aggregate/summary.json")

    def col(d, path, nd=0):
        for k in path:
            if d is None:
                return "--"
            d = d.get(k) if isinstance(d, dict) else None
        return pct(d, nd)

    def A(key):
        return s1[key], s2.get(key)

    a1s, a2s = A("family_a")                       # single Sersic, two seed sets
    a1d, a2d = A("family_a_double_sersic")         # double Sersic, two seed sets
    b1s, b1d = s1["family_b_single_sersic"], s1["family_b_double_sersic"]
    u1, u2 = s1["unet_v0"], s2.get("unet_v0")

    def five(none, v1s, v2s, v1d, v2d):
        return " & ".join([none, v1s, v2s, v1d, v2d])

    def fpr2(d, key, nd20=0):
        if d is None:
            return "--"
        v20, v100 = d[key]["20"], d[key]["100"]
        s20 = pct(v20, 2) if 0 < v20 < 0.01 else pct(v20)
        return f"{s20} / {pct(v100)}\\%"

    rows = [
        "clean fits rejected (of 300) & " + five(
            str(len(a_clean_recs) - len(a_clean_rel)),
            str(a1s["n_unreliable"]["no_subhalo"]), str(a2s["n_unreliable"]["no_subhalo"]),
            str(a1d["n_unreliable"]["no_subhalo"]), str(a2d["n_unreliable"]["no_subhalo"])),
        r"scan threshold, 10\% FPR ($\dchi$) & " + five(
            f"${a_thr:+.1f}$",
            *[f"${d['threshold_10pct_fpr']:+.0f}$" if abs(d["threshold_10pct_fpr"]) >= 10
              else f"${d['threshold_10pct_fpr']:+.1f}$" for d in (a1s, a2s, a1d, a2d)]),
        r"scan FPR clean, $\dchi{>}20$ / ${>}100$ & " + five(
                f"{pct(float(np.mean(a_clean > 20)), 2)} / {pct(float(np.mean(a_clean > 100)))}\\%",
            *[fpr2(d, "fpr_clean_abs") for d in (a1s, a2s, a1d, a2d)]),
        r"scan FPR multipole, $\dchi{>}20$ / ${>}100$ & " + five(
            f"{pct(float(np.mean(a_mp > 20)))} / {pct(float(np.mean(a_mp > 100)))}\\%",
            *[f"{pct(d['fpr_multipole_a3']['20'])} / {pct(d['fpr_multipole_a3']['100'])}\\%"
              for d in (a1s, a2s, a1d, a2d)]),
        r"scan compl.\ $10^{9.5}$--$10^{10}\Msun$ & " + five(
            pct(a_compl(9.5, 10.0)) + r"\%",
            *[col(d, ("completeness_c60", "9.5-10.0", "completeness")) + r"\%" for d in (a1s, a2s, a1d, a2d)]),
        r"scan compl.\ $10^{10}$--$10^{10.5}\Msun$ & " + five(
            pct(a_compl(10.0, 10.5)) + r"\%",
            *[col(d, ("completeness_c60", "10.0-10.5", "completeness")) + r"\%" for d in (a1s, a2s, a1d, a2d)]),
        r"linear \dpsi\ threshold, $\max|\delta\kappa|$ & " + five(
            f"{b_ref['threshold']:.2f}", f"{b1s['threshold_10pct_fpr']:.2f}", "--",
            f"{b1d['threshold_10pct_fpr']:.2f}", "--"),
        r"linear \dpsi\ compl.\ $10^{10}$--$10^{10.5}\Msun$ & " + five(
            pct(b_ref["completeness_c60"]["10.0-10.5"]["completeness"]) + r"\%",
            col(b1s, ("completeness_c60", "10.0-10.5", "completeness")) + r"\%", "--",
            col(b1d, ("completeness_c60", "10.0-10.5", "completeness")) + r"\%", "--"),
        r"U-Net AUC ($c{=}60$ vs.\ clean) & " + five(
            f"{u_ref['auc']['mean']:.2f}",
            *[f"{d['subtracted_recalibrated']['auc_c60_vs_clean']:.2f}" for d in (u1, u2)],
            *[f"{d['double_sersic_subtracted']['auc_c60_vs_clean']:.2f}" for d in (u1, u2)]),
        r"U-Net FPR clean at Tier-0 thr.\ & " + five(
            pct(u_ref["confounder_fpr"]["no_subhalo (calibration set)"]["mean"]) + r"\%",
            *[pct(d["subtracted_at_original_threshold"]["fpr_clean"]) + r"\%" for d in (u1, u2)],
            *[pct(d["double_sersic_subtracted"]["fpr_clean_at_original_threshold"]) + r"\%" for d in (u1, u2)]),
        r"U-Net compl.\ $10^{10}$--$10^{10.5}\Msun$ & " + five(
            pct(u_ref["completeness_c60"]["10.0-10.5"]["mean"]) + r"\%",
            *[col(d, ("subtracted_recalibrated", "completeness_c60", "10.0-10.5", "completeness")) + r"\%"
              for d in (u1, u2)],
            *[col(d, ("double_sersic_subtracted", "completeness_c60_recalibrated", "10.0-10.5", "completeness")) + r"\%"
              for d in (u1, u2)]),
    ]
    header = [r" & none & \multicolumn{2}{c}{single} & \multicolumn{2}{c}{double}",
              r" & & S1 & S2 & S1 & S2"]
    write("lenslight", tabular("@{}lccccc@{}", header, rows))


# --------------------------------------------------------------------------- tab:ledger
def ledger():
    """What could and could not be reproduced from the published U-Net description.

    Only the completeness row is a measurement; it is the four-seed mean the paper reports
    everywhere else, not a single seed.
    """
    a8, a30 = jload("results/aggregate/summary.json"), jload("results/aggregate_30k/summary.json")
    c8, c30 = a8["completeness_c60"]["9.0-9.5"], a30["completeness_c60"]["9.0-9.5"]
    n8 = len(jload("results/aggregate/summary.json")["auc"]["values"])
    rows = [
        r"Architecture & ``U-Net image segmentation'' & 4-level, base 16",
        r"Training images & $5\times10^5$ & 8\,000 (30\,000 in the scale-up)",
        r"Loss & not specified & BCE, weight 150",
        r"Image score & max pixel probability & same",
        r"Success criterion & within 2 pixels & 2-px disc target",
        r"Completeness at $10^{9}$--$10^{9.5}\Msun$, $c=60$ & 33.6\% & "
        f"${100 * c8['mean']:.1f}\\pm{100 * c8['std']:.1f}$\\% (8k); "
        f"${100 * c30['mean']:.1f}\\pm{100 * c30['std']:.1f}$\\% (30k)",
        r"$c=15$ behaviour & ``random guessing'' & reproduced",
        r"Code released & no & yes (this work)",
    ]
    assert n8 == 4, f"ledger text says four seeds, aggregate has {n8}"
    write("ledger", tabular("@{}p{2.35cm}p{2.55cm}p{2.3cm}@{}",
                            [r"Item & Specified in the paper & This work"], rows))


# --------------------------------------------------------------------------- tab:mass_mechanism
def mass_mechanism():
    """Where Family A's mass bias comes from: four idealizations removed one at a time.

    Rows 1-2 are the scan as run, from the localization summaries. Rows 3-4 recompute the
    oracle-position estimates from the per-lens records with the same selection the summary uses
    (a fine-grid signal above 20 at the true position). Rows 5-6 come from the joint re-fit test.
    """
    loc15 = jload("results/localization/summary.json")["family_a"]
    loc60 = jload("results/localization/summary_c60.json")["family_a"]

    def loc_row(label, d):
        e = d["localized_mass_error_dex"]
        # every localized estimate is low and the least-biased is beyond 0.25 dex, so none is within
        assert e["n_underestimated"] == e["n"] and e["max"] < -0.25, "within-0.25 count no longer derivable"
        return f"{label} & ${e['median']:+.1f}$ & {e['n_underestimated']}/{e['n']} & 0"

    recs = [json.loads(l) for l in open(ROOT / "results/oracle_position_mass/records.jsonl")]
    sig = [x for x in recs if x["reliable_fit"] and max(x["dchi2_true_pos_fine"].values()) > 20]

    def oracle_row(label, key):
        e = np.array([float(max(x[key], key=lambda k: x[key][k])) - x["log10_M200_true"] for x in sig])
        return (f"{label} & ${np.median(e):+.2f}$ & {int((e < 0).sum())}/{len(e)} & "
                f"{int((np.abs(e) <= 0.25).sum())}")

    jr = jload("results/joint_refit_mass/summary.json")

    def jr_row(label, key, bold=False):
        d = jr[key]["signal"]
        med = f"{d['median_error_dex']:+.2f}"
        within = str(d["n_within_0p25dex"])
        if bold:
            med, within = rf"\mathbf{{{med}}}", rf"\textbf{{{within}}}"
        return (f"{label} & ${med}$ & {d['n_underestimated']}/{d['n']} & {within}")

    rows = [
        loc_row(r"scan, $c=15$, 72-pt grid (loc.\ finds)", loc15),
        loc_row(r"scan, $c=60$, 72-pt grid (loc.\ finds)", loc60),
        oracle_row("true pos., 3 masses, macro frozen", "dchi2_true_pos_coarse"),
        oracle_row("true pos., fine masses, macro frozen", "dchi2_true_pos_fine"),
        jr_row("true pos., free mass, macro frozen", "fixed_macro_mass_free"),
        jr_row(r"true pos., free mass, \textbf{joint re-fit}", "joint_refit", bold=True),
    ]
    header = [r" & med.\ err.\ & biased & within", r" & (dex) & low & 0.25 dex"]
    write("mass_mechanism", tabular("@{}lccc@{}", header, rows))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    decoy()
    joint()
    lenslight()
    ledger()
    mass_mechanism()
