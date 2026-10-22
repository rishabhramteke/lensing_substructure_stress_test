"""The pixelized-source parametric search (run_family_a_pix.py) on the stratified subset results/apix_subset.json, with
Family A's joint scan and gravitational imaging (two-stage) on the same lenses for comparison.

    python scripts/evaluate_apix.py      -> results/apix.json, paper/tables/apix.tex
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BINS = [(8.5, 9.0), (9.0, 9.5), (9.5, 10.0), (10.0, 10.5)]
POPS = ["no_subhalo", "test_fixed60", "multipole_m4_a3", "decoy_gaussian_s42_a10", "decoy_gaussian_s43_a10", "decoy_dipole_s42_a10"]
A_ROOT = {"no_subhalo": "baseline_a_joint_full", "test_fixed60": "baseline_a_joint_full", "multipole_m4_a3": "baseline_a_joint_full",
          "decoy_gaussian_s42_a10": "baseline_a_joint_decoy", "decoy_gaussian_s43_a10": "baseline_a_joint_decoy", "decoy_dipole_s42_a10": "baseline_a_joint_decoy"}


def load(p):
    out = {}
    if Path(p).exists():
        for l in open(p):
            try:
                r = json.loads(l); out[r["index"]] = r
            except json.JSONDecodeError:
                pass
    return out


def wilson(k, n, z=1.0):
    if not n:
        return None
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return {"n": n, "k": int(k), "p": p, "lo": max(0.0, c - h), "hi": min(1.0, c + h)}


def main():
    sub = json.loads((ROOT / "results/apix_subset.json").read_text())
    P = {p: load(ROOT / "results/baseline_apix" / p / "scan_results.jsonl") for p in POPS}
    A = {p: load(ROOT / "results" / A_ROOT[p] / p / "scan_results.jsonl") for p in POPS}
    G = {p: load(ROOT / "results/baseline_bgi" / p / "scan_results.jsonl") for p in POPS}
    GC = {p: load(ROOT / "results/baseline_bgi" / p / "confirmed.jsonl") for p in POPS}
    rel = {p: [P[p][i] for i in sub[p] if i in P[p] and P[p][i].get("reliable_fit")] for p in POPS}
    thr = float(np.quantile([r["delta_lnE"] for r in rel["no_subhalo"]], 0.90)) if rel["no_subhalo"] else None
    a_thr = float(np.quantile([r["delta_chi2"] for r in A["no_subhalo"].values() if r["reliable_fit"]], 0.90))

    def a_flag(p, i, rule):
        r = A[p].get(i)
        return bool(r and r["reliable_fit"] and r["delta_chi2"] >= (a_thr if rule == "cal" else 20.0))

    def g_flag(p, i):
        r = G[p].get(i); c = GC[p].get(i)
        return bool(r and r.get("reliable_fit") and r["delta_lnZ"] >= 10 and c and c["delta_chi2_confirm"] >= 20)
    rules = {"lne10": lambda p, r: r["delta_lnE"] >= 10, "cal": lambda p, r: thr is not None and r["delta_lnE"] >= thr}
    out = {"threshold_cal": thr, "n_done": {p: len(P[p]) for p in POPS}, "n_subset": {p: len(sub[p]) for p in POPS},
           "t_scan_median_min": float(np.median([r["t_scan_s"] for p in POPS for r in rel[p]]) / 60) if any(rel.values()) else None, "pops": {}}
    for p in POPS:
        R = rel[p]; d = {"n": len(R), "gated": sum(1 for i in sub[p] if i in P[p]) - len(R)}
        for k, f in rules.items():
            d[f"rate_{k}"] = wilson(sum(f(p, r) for r in R), len(R))
        d["rate_A_cal"] = wilson(sum(a_flag(p, r["index"], "cal") for r in R), len(R))
        d["rate_A_20"] = wilson(sum(a_flag(p, r["index"], "abs") for r in R), len(R))
        d["rate_GI_confirmed"] = wilson(sum(g_flag(p, r["index"]) for r in R), len(R))
        if p == "test_fixed60":
            for k, f in list(rules.items()) + [("A_20", None), ("GI_confirmed", None)]:
                comp = {}
                for lo, hi in BINS:
                    S = [r for r in R if lo <= r["log10_M200_true"] < hi]
                    if k == "A_20":
                        kk = sum(a_flag(p, r["index"], "abs") for r in S)
                    elif k == "GI_confirmed":
                        kk = sum(g_flag(p, r["index"]) for r in S)
                    else:
                        kk = sum(f(p, r) for r in S)
                    comp[f"{lo}-{hi}"] = wilson(kk, len(S))
                d[f"completeness_{k}"] = comp
            T = [json.loads(l) for l in open(ROOT / "data/test_fixed60/truth.jsonl")]
            det = [r for r in R if r["delta_lnE"] >= 10]
            dd = np.array([np.hypot(r["best_x"] - T[r["index"]]["subhalo"]["x"], r["best_y"] - T[r["index"]]["subhalo"]["y"]) for r in det])
            me = np.array([r["best_log10_m"] - r["log10_M200_true"] for r, x in zip(det, dd) if x < 0.16])
            d["localization"] = {"n": len(det), "frac_within_0p16": float(np.mean(dd < 0.16)) if len(dd) else None,
                                 "median_offset": float(np.median(dd)) if len(dd) else None,
                                 "mass_error_localized_median": float(np.median(me)) if len(me) else None, "n_localized": int(len(me))}
        out["pops"][p] = d
    (ROOT / "results/apix.json").write_text(json.dumps(out, indent=2))

    def pc(w):
        return "--" if w is None else f"{100 * w['p']:.0f}"
    Q = out["pops"]
    dec = lambda key: [Q[q][key] for q in POPS if q.startswith("decoy") and Q[q][key]]  # noqa: E731

    def drange(key):
        v = [round(100 * w["p"]) for w in dec(key)]
        return "--" if not v else (f"{min(v)}" if min(v) == max(v) else f"{min(v)}--{max(v)}")
    lines = [r"\begin{tabular}{@{}l" + "c" * len(BINS) + "ccc@{}}", r"\toprule",
             r" & \multicolumn{" + str(len(BINS)) + r"}{c}{$c=60$ completeness, $\log_{10}(M_{200}/\Msun)$} & \multicolumn{3}{c}{false-positive rate} \\",
             r"pipeline, operating point & " + " & ".join(f"{a:g}--{b:g}" for a, b in BINS) + r" & clean & mult. & decoys \\", r"\midrule"]
    for lab, key in (("pixelized-source search, $\\Delta\\ln E\\ge10$", "lne10"), ("pixelized-source search, 10\\% calibration", "cal"),
                     ("analytic-source scan, $\\dchi\\ge20$", "A_20"), ("gravitational imaging, confirmed", "GI_confirmed")):
        rk = {"lne10": "rate_lne10", "cal": "rate_cal", "A_20": "rate_A_20", "GI_confirmed": "rate_GI_confirmed"}[key]
        lines.append(f"{lab} & " + " & ".join(pc(Q["test_fixed60"][f"completeness_{key}"][f"{a}-{b}"]) for a, b in BINS)
                     + f" & {pc(Q['no_subhalo'][rk])} & {pc(Q['multipole_m4_a3'][rk])} & {drange(rk)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "paper/tables/apix.tex").write_text("\n".join(lines) + "\n")
    print("\n".join(lines)); print(json.dumps({k: v for k, v in out.items() if k != "pops"}, indent=1))
    print("localization", out["pops"]["test_fixed60"].get("localization"))


if __name__ == "__main__":
    main()
