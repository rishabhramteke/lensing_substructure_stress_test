"""Gravitational imaging (joint pixelized source + pixelized dpsi, evidence-set regularizations; run_family_b_gi.py) on
the seed-0 300-lens subsamples, alone and with the parametric confirmation of every candidate at Delta ln Z >= 10
(confirm_family_b.py --root results/baseline_bgi --min-stat 10).

    python scripts/evaluate_family_bgi.py      -> results/family_bgi.json, paper/tables/bgi.tex
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R0 = ROOT / "results/baseline_bgi"
BINS = [(8.0, 8.5), (8.5, 9.0), (9.0, 9.5), (9.5, 10.0), (10.0, 10.5), (10.5, 11.0)]
POPS = ["no_subhalo", "test_fixed60", "test_fixed15", "multipole_m4_a3", "decoy_gaussian_s42_a10", "decoy_gaussian_s43_a10", "decoy_dipole_s42_a10"]
LNZ, CONF = 10.0, 20.0


def load(p):
    out = {}
    if p.exists():
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
    S = {p: load(R0 / p / "scan_results.jsonl") for p in POPS}
    C = {p: load(R0 / p / "confirmed.jsonl") for p in POPS}
    rel = {p: [r for r in S[p].values() if r.get("reliable_fit")] for p in POPS}
    thr = float(np.quantile([r["delta_lnZ"] for r in rel["no_subhalo"]], 0.90))
    rules = {"cal": lambda p, r: r["delta_lnZ"] >= thr,
             "lnz10": lambda p, r: r["delta_lnZ"] >= LNZ,
             "confirmed": lambda p, r: r["delta_lnZ"] >= LNZ and r["index"] in C[p] and C[p][r["index"]]["delta_chi2_confirm"] >= CONF}
    out = {"threshold_cal": thr, "n_candidates_unconfirmed": {p: sum(1 for r in rel[p] if r["delta_lnZ"] >= LNZ and r["index"] not in C[p]) for p in POPS},
           "t_scan_median_s": float(np.median([r["t_scan_s"] for p in POPS for r in rel[p]])), "pops": {}}
    for p in POPS:
        d = {"n": len(rel[p]), "gated": len(S[p]) - len(rel[p])}
        for k, f in rules.items():
            d[f"rate_{k}"] = wilson(sum(f(p, r) for r in rel[p]), len(rel[p]))
            if p.startswith("test_"):
                d[f"completeness_{k}"] = {f"{a}-{b}": wilson(sum(f(p, r) for r in rel[p] if r["has_subhalo"] and a <= r["log10_M200_true"] < b),
                                                            sum(1 for r in rel[p] if r["has_subhalo"] and a <= r["log10_M200_true"] < b)) for a, b in BINS}
        out["pops"][p] = d
    # paired concentration flip on the lenses both arms retain, and localization of the stage-1 dkappa peak
    T = {p: [json.loads(l) for l in open(ROOT / "data" / p / "truth.jsonl")] for p in ("test_fixed60", "test_fixed15")}
    both = [i for i in S["test_fixed60"] if i in S["test_fixed15"] and S["test_fixed60"][i].get("reliable_fit") and S["test_fixed15"][i].get("reliable_fit")
            and S["test_fixed60"][i]["has_subhalo"]]
    for k, f in rules.items():
        a = np.array([f("test_fixed60", S["test_fixed60"][i]) for i in both]); b = np.array([f("test_fixed15", S["test_fixed15"][i]) for i in both])
        out[f"flip_{k}"] = [int((a & ~b).sum()), int((b & ~a).sum()), len(both)]
    loc = {}
    for p in ("test_fixed60", "test_fixed15"):
        det = [r for r in rel[p] if r["has_subhalo"] and r["delta_lnZ"] >= LNZ]
        dd = np.array([np.hypot(r["best_x"] - T[p][r["index"]]["subhalo"]["x"], r["best_y"] - T[p][r["index"]]["subhalo"]["y"]) for r in det])
        loc[p] = {"n": len(det), "frac_within_0p16": float(np.mean(dd < 0.16)) if len(dd) else None, "median": float(np.median(dd)) if len(dd) else None}
    out["localization_stage1"] = loc
    (ROOT / "results/family_bgi.json").write_text(json.dumps(out, indent=2))

    def pc(w):
        return "--" if w is None else f"{100 * w['p']:.0f}"
    lab = {"cal": "10\\% calibration", "lnz10": "$\\Delta\\ln Z\\ge10$", "confirmed": "$\\Delta\\ln Z\\ge10$, confirmed"}
    lines = [r"\begin{tabular}{@{}l" + "c" * len(BINS) + "ccc@{}}", r"\toprule",
             r" & \multicolumn{" + str(len(BINS)) + r"}{c}{$c=60$ completeness, $\log_{10}(M_{200}/\Msun)$} & \multicolumn{3}{c}{false-positive rate} \\",
             r"operating point & " + " & ".join(f"{a:g}--{b:g}" for a, b in BINS) + r" & clean & mult. & decoys \\", r"\midrule"]
    for k in rules:
        P = out["pops"]
        dec = [P[q][f"rate_{k}"]["p"] for q in POPS if q.startswith("decoy") and P[q][f"rate_{k}"]]
        lines.append(f"{lab[k]} & " + " & ".join(pc(P["test_fixed60"][f"completeness_{k}"][f"{a}-{b}"]) for a, b in BINS)
                     + f" & {100 * P['no_subhalo'][f'rate_{k}']['p']:.1f} & {pc(P['multipole_m4_a3'][f'rate_{k}'])} & "
                     + ((f"{100 * min(dec):.0f}--{100 * max(dec):.0f}" if round(100 * min(dec)) != round(100 * max(dec)) else f"{100 * min(dec):.0f}") if dec else "--") + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "paper/tables/bgi.tex").write_text("\n".join(lines) + "\n")
    print("\n".join(lines)); print(json.dumps({k: out[k] for k in out if k != "pops"}, indent=1))


if __name__ == "__main__":
    main()
