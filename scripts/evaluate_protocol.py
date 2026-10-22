"""The recommended protocol of Sect. 5.4 scored end to end, against the joint scan as run throughout, on the same lenses.

Protocol scan (scripts/launch_round32_protocol.sh): start from the data alone with seven restarts, lens centre free,
perturber position free, free m=3 and m=4 multipoles in the smooth and joint macro-models. Decision rules: Delta chi^2 >=
20; >= 48, the evidence-equivalent threshold; and >= 48 with the joint model's chi^2/dof < 1.1, the goodness-of-fit gate.
Reference: the joint scan of the paper (fixed centre, grid position, no multipole freedom, near-truth start) on the same
lenses at Delta chi^2 >= 20.

    python scripts/evaluate_protocol.py      -> results/protocol.json, paper/tables/protocol.tex
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "results/baseline_a_joint_protocol"
BINS = [(8.0, 8.5), (8.5, 9.0), (9.0, 9.5), (9.5, 10.0), (10.0, 10.5), (10.5, 11.0)]
REF = {"no_subhalo": "full", "test_fixed60": "full", "test_fixed15": "full", "multipole_m4_a3": "full", "multipole_m4_a1": "full",
       "multipole_m3_a3": "full", "ellmultipole_m4_a3": "ellmp", "decoy_gaussian_s42_a10": "decoy", "decoy_gaussian_s43_a10": "decoy",
       "decoy_dipole_s42_a10": "decoy", "realistic_no_subhalo": "realistic_full", "realistic_fixed60": "realistic_full",
       "realistic_multipole_m4_a3": "realistic_full",
       # forward-model variants (Sect. 4.8) and real sources (App. D): clean lenses only, the bare scan on the same lenses
       "misspec_t0_psf2_no_subhalo": "misspec", "misspec_t0_psf5_no_subhalo": "misspec", "misspec_t0_clump0p5_no_subhalo": "misspec",
       "misspec_t0_clump1_no_subhalo": "misspec", "misspec_t0_clump2_no_subhalo": "misspec", "misspec_t0_clump4_no_subhalo": "misspec",
       "tier1_no_subhalo": "tier1"}
REF_NAME = {"realistic_no_subhalo": "no_subhalo", "realistic_fixed60": "test_fixed60", "realistic_multipole_m4_a3": "multipole_m4_a3", "tier1_no_subhalo": "no_subhalo"}
DATA = {"tier1_no_subhalo": "tier1/no_subhalo"}
LOC = 0.16


def load(path):
    return {r["index"]: r for r in map(json.loads, open(path))} if path.exists() else {}


def joint_per_dof(r):
    """The best joint model's chi^2 per pixel: delta_chi2 = chi2_smooth_basin - chi2_joint, so n_pix follows exactly."""
    joint = min(c["chi2_joint"] for c in r["cells"])
    return joint * r["chi2_smooth_basin_per_dof"] / (joint + r["delta_chi2"])


def rules(r):
    d, j = r["delta_chi2"], joint_per_dof(r)
    return {"dchi2_20": d >= 20, "dchi2_48": d >= 48, "dchi2_48_gof": d >= 48 and j is not None and j < 1.1}


def summarize(pop):
    prot = load(P / pop / "scan_results.jsonl")
    if not prot:
        return None
    ref = load(ROOT / f"results/baseline_a_joint_{REF[pop]}" / REF_NAME.get(pop, pop) / "scan_results.jsonl")
    truths = [json.loads(l) for l in open(ROOT / "data" / DATA.get(pop, pop) / "truth.jsonl")]
    idx = sorted(prot)
    out = {"n": len(idx), "excluded": float(np.mean([not prot[i]["reliable_fit"] for i in idx])),
           "ref_excluded": float(np.mean([not ref[i]["reliable_fit"] for i in idx if i in ref])) if ref else None}
    keep = [i for i in idx if prot[i]["reliable_fit"]]
    keep_ref = [i for i in idx if i in ref and ref[i]["reliable_fit"]]
    sub = lambda ii: [i for i in ii if truths[i].get("subhalo")]  # noqa: E731
    for name in ("dchi2_20", "dchi2_48", "dchi2_48_gof"):
        out[name] = float(np.mean([rules(prot[i])[name] for i in keep])) if keep else None
    out["ref_dchi2_20"] = float(np.mean([ref[i]["delta_chi2"] >= 20 for i in keep_ref])) if keep_ref else None
    if any(truths[i].get("subhalo") for i in idx):
        m = lambda i: truths[i]["subhalo"]["log10_M200"]  # noqa: E731
        for name in ("dchi2_20", "dchi2_48", "dchi2_48_gof"):
            out[name + "_bins"] = [float(np.mean([rules(prot[i])[name] for i in sub(keep) if lo <= m(i) < hi])) if any(lo <= m(i) < hi for i in sub(keep)) else None for lo, hi in BINS]
        out["ref_dchi2_20_bins"] = [float(np.mean([ref[i]["delta_chi2"] >= 20 for i in sub(keep_ref) if lo <= m(i) < hi])) if any(lo <= m(i) < hi for i in sub(keep_ref)) else None for lo, hi in BINS]
        det = [i for i in sub(keep) if rules(prot[i])["dchi2_48"]]
        if det:
            off = [np.hypot(prot[i]["best_x"] - truths[i]["subhalo"]["x"], prot[i]["best_y"] - truths[i]["subhalo"]["y"]) for i in det]
            out["localized_dchi2_48"] = float(np.mean(np.array(off) <= LOC))
            out["mass_error_dchi2_48"] = float(np.median([prot[i]["best_log10_m"] - m(i) for i in det if prot[i].get("best_log10_m") is not None]))
    return out


def main():
    res = {pop: summarize(pop) for pop in REF}
    res = {k: v for k, v in res.items() if v}
    (ROOT / "results/protocol.json").write_text(json.dumps(res, indent=2))
    table(res)


def table(res=None):
    res = res or json.loads((ROOT / "results/protocol.json").read_text())
    pc = lambda v: "--" if v is None else (f"{100 * v:.1f}" if 0 < v < 0.095 else f"{100 * v:.0f}")  # noqa: E731
    rules_ = (("ref_dchi2_20", "reference"), ("dchi2_20", r"$\dchi\ge20$"), ("dchi2_48", r"$\dchi\ge48$"), ("dchi2_48_gof", r"$+$fit"))
    L = [r"\begin{tabular}[t]{@{}lrcccc@{}}", r"\toprule", r" & & reference & \multicolumn{3}{c}{protocol} \\ \cmidrule(lr){4-6}",
         r"population & $n$ & $\dchi\ge20$ & $\dchi\ge20$ & $\dchi\ge48$ & $+$fit \\", r"\midrule"]
    lab = {"no_subhalo": "clean", "multipole_m4_a3": r"$m{=}4$, $a_4/\thetaE=0.03$", "multipole_m4_a1": r"$m{=}4$, 0.01", "multipole_m3_a3": r"$m{=}3$, 0.03",
           "ellmultipole_m4_a3": r"elliptical $m{=}4$, 0.03", "decoy_gaussian_s42_a10": r"decoy, Gaussian (1)", "decoy_gaussian_s43_a10": r"decoy, Gaussian (2)",
           "decoy_dipole_s42_a10": "decoy, dipole", "realistic_no_subhalo": "realism: clean", "realistic_multipole_m4_a3": r"realism: $m{=}4$, 0.03",
           "misspec_t0_psf2_no_subhalo": r"clean, PSF 2\% too wide", "misspec_t0_psf5_no_subhalo": r"clean, PSF 5\%", "misspec_t0_clump0p5_no_subhalo": r"clean, clumps 0.5\%",
           "misspec_t0_clump1_no_subhalo": r"clean, clumps 1\%", "misspec_t0_clump2_no_subhalo": r"clean, clumps 2\%", "misspec_t0_clump4_no_subhalo": r"clean, clumps 4\%",
           "tier1_no_subhalo": r"clean, real source$^{a}$"}
    for pop, t in lab.items():
        if pop in res:
            r = res[pop]
            if pop == "misspec_t0_psf2_no_subhalo":
                L.append(r"\addlinespace\multicolumn{6}{@{}l}{\emph{forward-model variants, exact model otherwise}} \\")
            L.append(f"{t} & {r['n']} & " + " & ".join(pc(r[k]) for k, _ in rules_) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\hfill",
          r"\begin{tabular}[t]{@{}llcccccc@{}}", r"\toprule", r" & & \multicolumn{6}{c}{completeness (\%)} \\ \cmidrule(lr){3-8}",
          r"population & rule & $10^{8}$ & $10^{8.5}$ & $10^{9}$ & $10^{9.5}$ & $10^{10}$ & $10^{10.5}$ \\", r"\midrule"]
    for pop, t in (("test_fixed60", r"$c=60$"), ("test_fixed15", r"$c=15$"), ("realistic_fixed60", r"realism: $c=60$")):
        if pop not in res:
            continue
        for j, (k, kl) in enumerate(rules_):
            L.append(f"{t if j == 0 else ''} & {kl if j else 'reference'} & " + " & ".join(pc(v) for v in res[pop][k + "_bins"]) + r" \\")
        L.append(r"\addlinespace")
    L[-1] = r"\bottomrule"
    L.append(r"\end{tabular}")
    (ROOT / "paper/tables/protocol.tex").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    for pop in res:
        print(pop, "excluded", res[pop]["excluded"], "ref excl", res[pop]["ref_excluded"], "localized", res[pop].get("localized_dchi2_48"), "mass error", res[pop].get("mass_error_dchi2_48"))


if __name__ == "__main__":
    main()
