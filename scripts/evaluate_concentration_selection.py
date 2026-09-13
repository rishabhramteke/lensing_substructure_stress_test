"""Referee round 8, point M2: is c=60 "unphysically dense", or is it what detection selects?

Two questions, both answerable from the matched concentration populations (identical lenses,
sources, subhalo masses and positions; only c differs).

1. Detection selection. Detection is concentration-selected, so the *detected* population is
   denser than the input one even if the input follows a physical c-M relation. We estimate
   P(detect | c, M) by interpolating in log c over the three matched populations, assume an
   input p(c | M) -- the Dutton & Maccio (2014) field relation with a Moline et al. (2017)
   tidal boost and 0.11 dex scatter -- weight by the CDM mass function, and compare the input
   and detected concentration distributions. This replaces an assertion with a measurement.

2. Tidal track. Everywhere else this paper holds tau = r_t/r_s fixed at 20 while varying c,
   which is the literature's convention but unphysical in a specific direction: at fixed M200
   a lower c gives a larger r_s, so fixed tau makes the truncation radius grow as well, and the
   low-c perturber is both more diffuse and more extended. `data/test_tidal_c*` instead scales
   tau with c (tau = 5 at c = 15), so the low-c subhalo is more severely stripped, as a real one
   near theta_E would be. The two are compared here.

    python scripts/evaluate_concentration_selection.py
-> results/concentration_selection.json, paper/tables/tidal.tex
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BINS = [(9.0, 9.5), (9.5, 10.0), (10.0, 10.5)]
ALPHA = -1.9                       # CDM subhalo mass function slope, as in reweight_by_mass_function.py
MOLINE_BOOST = 2.5                 # Moline et al. (2017): subhalos are ~2-3x denser than field halos
ARMS = {60: "results/baseline_a/test_fixed60", 30: "results/baseline_a/test_fixed30", 15: "results/baseline_a/test_fixed15"}
TIDAL = {15: "results/baseline_a_tidal/test_tidal_c15", 30: "results/baseline_a_tidal/test_tidal_c30"}


def wilson(k, n, z=1.0):
    if not n:
        return (np.nan,) * 3
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def load(root):
    p = ROOT / root / "scan_results.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else None


def threshold():
    cl = [r for r in load("results/baseline_a/no_subhalo") if r["reliable_fit"]]
    return float(np.quantile([r["delta_chi2"] for r in cl], 0.90)), float(np.mean([r["delta_chi2"] > 0 for r in cl]))


def completeness(root, thr, floor):
    recs = load(root)
    if recs is None:
        return None
    pos = [r for r in recs if r["reliable_fit"] and r["has_subhalo"]]
    hit = (lambda r: r["delta_chi2"] > 0) if floor else (lambda r: r["delta_chi2"] >= thr)
    out = {}
    for lo, hi in BINS:
        sel = [r for r in pos if lo <= r["log10_M200_true"] < hi]
        k = sum(1 for r in sel if hit(r)); p_, l_, h_ = wilson(k, len(sel))
        out[f"{lo}-{hi}"] = {"n": len(sel), "k": k, "p": p_, "lo": l_, "hi": h_}
    return out


def dutton_maccio(log10_m, z=0.5):
    a = 0.520 + (0.905 - 0.520) * np.exp(-0.617 * z ** 1.21)
    b = -0.101 + 0.026 * z
    return 10.0 ** (a + b * (log10_m - 12.0))


def main():
    thr, clean_floor = threshold()
    res = {"threshold_10pct": thr, "clean_fpr_floor": clean_floor, "completeness": {}, "tidal": {}}
    for c, root in ARMS.items():
        res["completeness"][c] = {"cal": completeness(root, thr, False), "floor": completeness(root, thr, True)}
    for c, root in TIDAL.items():
        res["tidal"][c] = {"cal": completeness(root, thr, False), "floor": completeness(root, thr, True)}

    # ---- 1. detection selection in concentration
    cs = np.array(sorted(ARMS))                       # 15, 30, 60
    logc = np.log10(cs)
    grid_c = np.logspace(np.log10(8), np.log10(80), 200)
    mass_mid = np.array([0.5 * (lo + hi) for lo, hi in BINS])
    w_mass = 10.0 ** ((ALPHA + 1) * (mass_mid - mass_mid[0]))   # CDM weight per 0.5-dex bin
    w_mass = w_mass / w_mass.sum()
    sel = {}
    for key in ("cal", "floor"):
        p_det = np.zeros((len(BINS), len(grid_c)))
        for i, (lo, hi) in enumerate(BINS):
            y = np.array([res["completeness"][c][key][f"{lo}-{hi}"]["p"] for c in cs])
            p_det[i] = np.clip(np.interp(np.log10(grid_c), logc, y), 0, 1)
        out = {}
        for boost, lab, sig in [(1.0, "field", 0.11), (MOLINE_BOOST, "tidally_boosted", 0.11),
                                (MOLINE_BOOST, "boosted_scatter0.2", 0.20), (MOLINE_BOOST, "boosted_scatter0.3", 0.30),
                                (MOLINE_BOOST, "boosted_scatter0.5", 0.50)]:
            num_in = np.zeros_like(grid_c); num_det = np.zeros_like(grid_c)
            for i, m in enumerate(mass_mid):
                mu = np.log10(boost * dutton_maccio(m))
                pc = np.exp(-0.5 * ((np.log10(grid_c) - mu) / sig) ** 2)
                pc = pc / np.trapezoid(pc, np.log10(grid_c))
                num_in += w_mass[i] * pc
                num_det += w_mass[i] * pc * p_det[i]
            det = num_det / np.trapezoid(num_det, np.log10(grid_c)) if np.trapezoid(num_det, np.log10(grid_c)) > 0 else num_det
            inp = num_in / np.trapezoid(num_in, np.log10(grid_c))
            mean_in = 10 ** np.trapezoid(np.log10(grid_c) * inp, np.log10(grid_c))
            mean_det = 10 ** np.trapezoid(np.log10(grid_c) * det, np.log10(grid_c))
            frac = float(np.trapezoid(num_det, np.log10(grid_c)) / np.trapezoid(num_in, np.log10(grid_c)))
            out[lab] = {"median_c_input": float(mean_in), "median_c_detected": float(mean_det),
                        "shift_dex": float(np.log10(mean_det / mean_in)), "overall_completeness": frac}
        sel[key] = out
    res["selection"] = sel
    (ROOT / "results/concentration_selection.json").write_text(json.dumps(res, indent=2))

    # ---- bound mass within r_t, so the reader can see whether "fixed M200" compares like with like
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "src"))
    from lenstronomy.Cosmo.lens_cosmo import LensCosmo
    from lenstronomy.LensModel.Profiles.tnfw import TNFW
    _LC, _P = LensCosmo(z_lens=0.5, z_source=1.0), TNFW()

    def _bound(sub):
        Rs, aRs = _LC.nfw_physical2angle(M=10 ** sub["log10_M200"], c=sub["concentration"])
        rho0 = _P.alpha2rho0(alpha_Rs=aRs, Rs=Rs); rt = sub["tau"] * Rs
        return float(_P.mass_3d(rt, Rs, rho0, rt) * _LC.sigma_crit_angle)

    res["bound_mass"] = {}
    for pop in ("test_fixed60", "test_fixed30", "test_fixed15", "test_tidal_c30", "test_tidal_c15"):
        f = ROOT / "data" / pop / "truth.jsonl"
        if not f.exists(): continue
        subs = [json.loads(l)["subhalo"] for l in open(f) if json.loads(l)["subhalo"]]
        mb = np.array([_bound(x) for x in subs]); m200 = np.array([10 ** x["log10_M200"] for x in subs])
        res["bound_mass"][pop] = {"c": subs[0]["concentration"], "tau": subs[0]["tau"],
                                  "median_log10_Mbound": float(np.median(np.log10(mb))),
                                  "median_Mbound_over_M200": float(np.median(mb / m200))}

    # ---- table: tidal track vs fixed tau
    lines = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
             r"completeness at $\dchi>0$ & $c{=}60$ & $c{=}30$ & $c{=}15$ \\", r"\midrule",
             r"\multicolumn{4}{@{}l}{$\tau=20$ fixed (this paper elsewhere, and the literature)} \\"]
    for lo, hi in BINS:
        k = f"{lo}-{hi}"
        lines.append(f"\\quad $10^{{{lo}}}$--$10^{{{hi}}}\\Msun$ & " +
                     " & ".join(f"{100*res['completeness'][c]['floor'][k]['p']:.0f}\\%" for c in (60, 30, 15)) + r" \\")
    lines.append(r"\addlinespace[2pt]")
    lines.append(r"\multicolumn{4}{@{}l}{$\tau$ scaled with $c$ (tidal track: $\tau=20,10,5$)} \\")
    for lo, hi in BINS:
        k = f"{lo}-{hi}"
        lines.append(f"\\quad $10^{{{lo}}}$--$10^{{{hi}}}\\Msun$ & " +
                     f"{100*res['completeness'][60]['floor'][k]['p']:.0f}\\% & " +
                     f"{100*res['tidal'][30]['floor'][k]['p']:.0f}\\% & " +
                     f"{100*res['tidal'][15]['floor'][k]['p']:.0f}\\%" + r" \\")
    B = res["bound_mass"]
    lines.append(r"\addlinespace[2pt]")
    lines.append(r"\multicolumn{4}{@{}l}{median bound mass within $r_{\rm t}$, as a fraction of $M_{200}$} \\")
    lines.append(r"\quad $\tau=20$ fixed & " + " & ".join(f"{B[p]['median_Mbound_over_M200']:.2f}" for p in ("test_fixed60", "test_fixed30", "test_fixed15")) + r" \\")
    lines.append(r"\quad tidal track & " + f"{B['test_fixed60']['median_Mbound_over_M200']:.2f} & {B['test_tidal_c30']['median_Mbound_over_M200']:.2f} & {B['test_tidal_c15']['median_Mbound_over_M200']:.2f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "paper/tables/tidal.tex").write_text("\n".join(lines) + "\n")

    print(f"threshold {thr:.1f} | clean floored FPR {100*clean_floor:.2f}%")
    for key in ("cal", "floor"):
        print(f"== {key}")
        for c in (60, 30, 15):
            print(f"   c={c:2d} tau=20  " + " ".join(f"{100*res['completeness'][c][key][f'{lo}-{hi}']['p']:3.0f}" for lo, hi in BINS))
        for c in (30, 15):
            print(f"   c={c:2d} tidal   " + " ".join(f"{100*res['tidal'][c][key][f'{lo}-{hi}']['p']:3.0f}" for lo, hi in BINS))
        for lab, d in res["selection"][key].items():
            print(f"   selection ({lab}): input median c {d['median_c_input']:.1f} -> detected {d['median_c_detected']:.1f} "
                  f"({d['shift_dex']:+.2f} dex), overall completeness {100*d['overall_completeness']:.1f}%")


if __name__ == "__main__":
    main()
