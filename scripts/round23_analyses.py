"""Analyses of existing runs requested in referee round 23 (no new fits):

  1. joint-scan completeness under a symmetric gate (smooth fit only; lenses only the joint model fits excluded)
  2. mass-function-weighted completeness above chance, (TPR - FPR) / (1 - FPR) per bin, at the 10% calibration
  3. every Family-A rate at Delta chi^2 >= 48, Delta ln Z > 10 at the median Occam penalty of the evidence test
  4. subhalo-versus-multipole AUC: c=60 subhalo lenses ranked against subhalo-free multipole lenses, per family
  5. the concentration-selection estimate of Appendix A with the U-Net's detection probability

    python scripts/round23_analyses.py        -> results/round23_analyses.json
"""
import json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from paper_config import B_FULL  # noqa: E402

A = ROOT / "results/baseline_a_joint_full"
BINS = [(8.0, 8.5), (8.5, 9.0), (9.0, 9.5), (9.5, 10.0), (10.0, 10.5), (10.5, 11.0)]
T48 = 48.0


def load(root, pop):
    return [json.loads(l) for l in open(root / pop / "scan_results.jsonl")]


def thr90(recs):
    return float(np.quantile([r["delta_chi2"] for r in recs if r["reliable_fit"]], 0.90))


def comp(recs, hit, keep):
    out = {}
    for lo, hi in BINS:
        S = [r for r in recs if keep(r) and r["has_subhalo"] and lo <= r["log10_M200_true"] < hi]
        out[f"{lo}-{hi}"] = {"n": len(S), "p": (sum(hit(r) for r in S) / len(S)) if S else None}
    return out


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    out = {}
    clean = load(A, "no_subhalo"); thr = thr90(clean)
    c60, c15 = load(A, "test_fixed60"), load(A, "test_fixed15")
    # 1. symmetric gate
    sym = lambda r: r["reliable_fit"] and r.get("gate_rescued_by") != "joint"  # noqa: E731
    std = lambda r: r["reliable_fit"]  # noqa: E731
    out["symmetric_gate"] = {k: {"standard": comp(R, lambda r: r["delta_chi2"] >= t, std), "symmetric": comp(R, lambda r: r["delta_chi2"] >= t, sym)}
                             for k, R, t in (("c60_cal", c60, thr), ("c60_abs", c60, 20.0), ("c15_cal", c15, thr))}
    # 3. Delta chi^2 >= 48
    rel = lambda R: [r for r in R if r["reliable_fit"]]  # noqa: E731
    d48 = {"clean_fpr": float(np.mean([r["delta_chi2"] >= T48 for r in rel(clean)])),
           "completeness_c60": comp(c60, lambda r: r["delta_chi2"] >= T48, std), "completeness_c15": comp(c15, lambda r: r["delta_chi2"] >= T48, std)}
    for pop in ("multipole_m4_a1", "multipole_m4_a3", "multipole_m3_a3"):
        d48[f"fpr_{pop}"] = float(np.mean([r["delta_chi2"] >= T48 for r in rel(load(A, pop))]))
    dec = ROOT / "results/baseline_a_joint_decoy"
    for pop in ("decoy_gaussian_s42_a10", "decoy_gaussian_s43_a10", "decoy_dipole_s42_a10"):
        if (dec / pop / "scan_results.jsonl").exists():
            d48[f"fpr_{pop}"] = float(np.mean([r["delta_chi2"] >= T48 for r in rel(load(dec, pop))]))
    ll = ROOT / "results/baseline_a_joint_lenslight2"
    d48["fpr_lenslight_double_clean"] = float(np.mean([r["delta_chi2"] >= T48 for r in rel(load(ll, "lenslight_no_subhalo"))]))
    ll1 = ROOT / "results/baseline_a_joint_lenslight"
    d48["fpr_lenslight_single_clean"] = float(np.mean([r["delta_chi2"] >= T48 for r in rel(load(ll1, "lenslight_no_subhalo"))]))
    out["dchi2_48"] = d48
    # 2. above-chance weighted completeness at the 10% calibration
    w = np.array([10.0 ** (-0.9 * 0.5 * k) for k in range(6)]); w /= w.sum()
    MF = {}
    cal = comp(c60, lambda r: r["delta_chi2"] >= thr, std)
    p = np.array([cal[f"{lo}-{hi}"]["p"] for lo, hi in BINS]); fpr = 0.10
    MF["A_c60"] = {"weighted": float(w @ p), "weighted_above_chance": float(w @ np.clip((p - fpr) / (1 - fpr), 0, None))}
    FB = json.loads((ROOT / "results/full_populations.json").read_text())["B"]
    pb = np.array([FB["completeness_c60"][f"{lo}-{hi}"]["completeness"] for lo, hi in BINS])
    MF["B_c60"] = {"weighted": float(w @ pb), "weighted_above_chance": float(w @ np.clip((pb - fpr) / (1 - fpr), 0, None))}
    U = json.loads((ROOT / "results/aggregate_30k/summary.json").read_text())
    pu = np.array([U["completeness_c60"][f"{lo}-{hi}"]["mean"] for lo, hi in BINS])
    MF["C_c60"] = {"weighted": float(w @ pu), "weighted_above_chance": float(w @ np.clip((pu - fpr) / (1 - fpr), 0, None))}
    MF["A_c60_abs"] = {"weighted": float(w @ np.array([comp(c60, lambda r: r["delta_chi2"] >= 20, std)[f"{lo}-{hi}"]["p"] for lo, hi in BINS]))}
    out["mass_function"] = MF
    # 4. subhalo vs multipole AUC
    AU = {}
    for fam, root in (("A", A), ("B", ROOT / B_FULL)):
        pos = [r["delta_chi2"] for r in load(root, "test_fixed60") if r["reliable_fit"] and r["has_subhalo"]]
        for pop in ("multipole_m4_a1", "multipole_m4_a3"):
            neg = [r["delta_chi2"] for r in load(root, pop) if r["reliable_fit"]]
            AU[f"{fam}_c60_vs_{pop}"] = auc(pos, neg)
    UP = ROOT / "results/unet_scores_cache.json"
    out["auc_subhalo_vs_multipole"] = AU
    # 5. Appendix A with the U-Net's P(detect | c, M)
    import evaluate_concentration_selection as ECS
    IC = json.loads((ROOT / "results/intermediate_concentration.json").read_text())["C"]["per_seed"]
    bins_u = [(9.0, 9.5), (9.5, 10.0), (10.0, 10.5)]
    compU = {}
    for c, pop in ((60, "test_fixed60"), (30, "test_fixed30"), (15, "test_fixed15")):
        compU[c] = {"cal": {f"{lo}-{hi}": {"n": IC[0]["populations"][pop]["bins"][f"{lo}-{hi}"]["n"],
                                            "p": float(np.mean([s["populations"][pop]["bins"][f"{lo}-{hi}"]["p"] for s in IC]))} for lo, hi in bins_u}}
    selU = ECS.selection(compU, bins_u, ("cal",))
    out["selection_unet"] = selU["cal"]
    (ROOT / "results/round23_analyses.json").write_text(json.dumps(out, indent=2))
    f = lambda c: " ".join("--" if v["p"] is None else f"{100*v['p']:.0f}" for v in c.values())  # noqa: E731
    for k, v in out["symmetric_gate"].items():
        print(f"sym gate {k}: standard {f(v['standard'])} | symmetric {f(v['symmetric'])}")
    print("dchi2>=48:", {k: (round(100 * v, 1) if isinstance(v, float) else f(v)) for k, v in d48.items()})
    print("mass function:", {k: {kk: round(100 * vv, 1) for kk, vv in v.items()} for k, v in MF.items()})
    print("AUC subhalo vs multipole:", {k: round(v, 3) for k, v in AU.items()})
    t = selU["cal"]["tidally_boosted"]; print(f"U-Net selection: input {t['median_c_input']:.1f} detected {t['median_c_detected']:.1f} shift {t['shift_dex']:+.3f} dex; boost x1 (field): {selU['cal']['field']['shift_dex']:+.3f}")


if __name__ == "__main__":
    main()
