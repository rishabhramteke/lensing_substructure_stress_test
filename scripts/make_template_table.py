"""The perturber-template control for the JOINT scan (referee round 24): the concentration loss (c=60 minus c=15
completeness, in points) on the same 300-lens subsample with the c=15 template (the full-population run restricted to
those lenses), the c=60 template and the Dutton & Maccio (2014) c-M relation, each calibrated on its own clean control.

    python scripts/make_template_table.py      -> paper/tables/tmpl_joint.tex
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BINS = [(8.0, 8.5), (8.5, 9.0), (9.0, 9.5), (9.5, 10.0), (10.0, 10.5)]


def L(root, pop):
    return [json.loads(l) for l in open(ROOT / "results" / root / pop / "scan_results.jsonl")]


def comp(R, t, idx):
    return [100 * np.mean([r["delta_chi2"] >= t for r in R if r["reliable_fit"] and r["has_subhalo"] and r["index"] in idx and a <= r["log10_M200_true"] < b]) for a, b in BINS]


def main():
    idx = {r["index"] for r in L("baseline_a_joint_tmpl60", "test_fixed60")}
    rows = []
    cidx = {r["index"] for r in L("baseline_a_joint_tmpl60", "no_subhalo")}   # the seed-1 300-lens clean subsample
    for lab, root in (("$c=15$ (as run)", "baseline_a_joint_full"), ("$c=60$", "baseline_a_joint_tmpl60"), ("$c$--$M$", "baseline_a_joint_tmplcm")):
        C = [r for r in L(root, "no_subhalo") if r["index"] in cidx]   # every template calibrated on the same clean lenses
        t = float(np.quantile([r["delta_chi2"] for r in C if r["reliable_fit"]], 0.90))
        out = []
        for thr in (t, 20.0):
            a = comp(L(root, "test_fixed60"), thr, idx); b = comp(L(root, "test_fixed15"), thr, idx)
            out.append([x - y for x, y in zip(a, b)])
        rows.append((lab, t, out))
    lines = [r"\begin{tabular}{@{}lc" + "c" * len(BINS) + "@{}}", r"\toprule",
             r"template & thr. & " + " & ".join(f"{a:g}--{b:g}" for a, b in BINS) + r" \\", r"\midrule",
             r"\multicolumn{" + str(2 + len(BINS)) + r"}{@{}l}{\emph{loss $c{=}60\to15$ at the 10\% calibration (points)}} \\"]
    for lab, t, out in rows:
        lines.append(f"\\quad {lab} & {t:.1f} & " + " & ".join(f"${x:+.0f}$" for x in out[0]) + r" \\")
    lines.append(r"\multicolumn{" + str(2 + len(BINS)) + r"}{@{}l}{\emph{loss at $\dchi\ge20$ (points)}} \\")
    for lab, t, out in rows:
        lines.append(f"\\quad {lab} & & " + " & ".join(f"${x:+.0f}$" for x in out[1]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "paper/tables/tmpl_joint.tex").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
