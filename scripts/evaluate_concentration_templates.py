"""The scan-template control for the concentration result (referee round 9, point M5a/M5d).

On the c=15 population the c=15 scan template is the *correct* one, so a like-for-like
comparison needs the same template run on both populations. This scores the three templates
(c=15, the Dutton & Maccio c-M relation, and c=60) on both the c=60 and c=15 test populations,
at both operating points -- the common 10% calibration and the null-floored dchi2>0 -- over all
four mass bins above 10^9 Msun, and reports the per-bin completeness loss.

The "4-13 points" quoted in the abstract and conclusions is the floored loss with the c=60
template held fixed; the "6-20 points" is the same comparison at the 10% calibration. Both are
generated here so that every number in the text traces to a row of the table.

    python scripts/evaluate_concentration_templates.py
-> results/concentration_templates.json, paper/tables/conc.tex
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BINS = [(9.0, 9.5), (9.5, 10.0), (10.0, 10.5), (10.5, 11.0)]
TEMPLATES = [("15", "results/baseline_a", "$c=15$"),
             ("cm", "results/baseline_a_ccm", "$c$--$M$"),
             ("60", "results/baseline_a_c60", "$c=60$")]


def wilson(k, n, z=1.0):
    if not n:
        return (np.nan,) * 3
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def load(root, pop):
    p = ROOT / root / pop / "scan_results.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else None


def main():
    res = {"bins": [f"{lo}-{hi}" for lo, hi in BINS], "templates": {}}
    for key, root, label in TEMPLATES:
        ns = load(root, "no_subhalo")
        if ns is None:
            print("missing", root); continue
        rel = [r for r in ns if r["reliable_fit"]]
        thr = float(np.quantile([r["delta_chi2"] for r in rel], 0.90))
        d = {"label": label, "root": root, "threshold_10pct": thr,
             "clean_fpr_floor": float(np.mean([r["delta_chi2"] > 0 for r in rel]))}
        for stat, hit in (("cal", lambda r: r["delta_chi2"] >= thr), ("floor", lambda r: r["delta_chi2"] > 0)):
            for pop in ("test_fixed60", "test_fixed15"):
                recs = load(root, pop)
                if recs is None: continue
                pos = [r for r in recs if r["reliable_fit"] and r["has_subhalo"]]
                b = {}
                for lo, hi in BINS:
                    sel = [r for r in pos if lo <= r["log10_M200_true"] < hi]
                    k = sum(1 for r in sel if hit(r)); p_, l_, h_ = wilson(k, len(sel))
                    b[f"{lo}-{hi}"] = {"n": len(sel), "k": k, "p": p_, "lo": l_, "hi": h_}
                d[f"{stat}_{pop}"] = b
            if f"{stat}_test_fixed60" in d and f"{stat}_test_fixed15" in d:
                d[f"{stat}_loss"] = {b: 100 * (d[f"{stat}_test_fixed60"][b]["p"] - d[f"{stat}_test_fixed15"][b]["p"]) for b in res["bins"]}
        res["templates"][key] = d
    (ROOT / "results/concentration_templates.json").write_text(json.dumps(res, indent=2))

    def row(key, stat, pop):
        d = res["templates"][key].get(f"{stat}_{pop}")
        return " & ".join(f"{100*d[b]['p']:.0f}\\%" for b in res["bins"]) if d else " & ".join(["--"] * 4)

    lines = [r"\begin{tabular}{@{}l" + "c" * 4 + "@{}}", r"\toprule",
             r"scan template & \multicolumn{4}{c}{$\log_{10}(M_{200}/\Msun)$ bin} \\",
             r"\cmidrule(l){2-5}",
             " & " + " & ".join(f"{lo}--{hi}" for lo, hi in BINS) + r" \\", r"\midrule"]
    for stat, slab in (("cal", r"at the common 10\% calibration"), ("floor", r"at the null floor, $\dchi>0$")):
        lines.append(r"\multicolumn{5}{@{}l}{\emph{" + slab + r"}} \\")
        for key, _, label in TEMPLATES:
            if key not in res["templates"]: continue
            lines.append(f"\\quad {label}, $c{{=}}60$ pop. & " + row(key, stat, "test_fixed60") + r" \\")
            lines.append(f"\\quad {label}, $c{{=}}15$ pop. & " + row(key, stat, "test_fixed15") + r" \\")
            L = res["templates"][key].get(f"{stat}_loss")
            if L:
                lines.append(f"\\quad\\quad loss (points) & " + " & ".join(f"{L[b]:.0f}" for b in res["bins"]) + r" \\")
        lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "paper/tables/conc.tex").write_text("\n".join(lines) + "\n")
    for key, _, label in TEMPLATES:
        if key not in res["templates"]: continue
        d = res["templates"][key]
        for stat in ("cal", "floor"):
            L = d.get(f"{stat}_loss")
            if L:
                v = [L[b] for b in res["bins"]]
                print(f"  {label:8s} {stat:5s} loss {' '.join(f'{x:5.1f}' for x in v)}  -> range {min(v):.0f}-{max(v):.0f}")


if __name__ == "__main__":
    main()
