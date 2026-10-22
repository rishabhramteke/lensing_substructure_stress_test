"""Training objectives for the U-Net, each trained on the 30 000-image c=60 and c=15 populations (same seed, only the
concentration differs) and scored on the standard Tier-0 test populations (referee rounds 21-22, M2/M4): the 2-pixel
disc segmentation recipe of Tsang+2024, a 5-pixel disc with the class weight rescaled by area, and two image-level
objectives on the frame's maximum logit and on its log-sum-exp. AUC of the c=60 test population against the clean
control, mean and spread over the seeds that exist.

    python scripts/make_unet_objectives_table.py      -> paper/tables/unet_objectives.tex, results/unet_objectives.json
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ROWS = [("disc, $r=2$~px (the recipe)", ["detector_30k_seed0", "detector_30k_seed1", "detector_30k_seed2", "detector_30k_seed3"], ["detector_30k_c15_seed0", "detector_30k_c15_seed1"]),
        ("disc, $r=5$~px", ["detector_30k_c60wide_seed0"], ["detector_30k_c15wide_seed0", "detector_30k_c15wide_seed1"]),
        ("image level, max logit", ["detector_30k_c60img_seed0"], ["detector_30k_c15img_seed0", "detector_30k_c15img_seed1"]),
        ("image level, log-sum-exp", ["detector_30k_c60lse_seed0"], ["detector_30k_c15lse_seed0", "detector_30k_c15lse_seed1"])]


def aucs(runs):
    v = []
    for r in runs:
        p = ROOT / "results" / r / "results.json"
        if p.exists():
            v.append(json.loads(p.read_text())["roc_auc_fixed60_vs_no_subhalo"])
    return v


def main():
    out, lines = {}, [r"\begin{tabular}{lcc}", r"\toprule", r"objective & trained at $c{=}60$ & trained at $c{=}15$ \\", r"\midrule"]
    for lab, r60, r15 in ROWS:
        a, b = aucs(r60), aucs(r15)
        out[lab] = {"c60": a, "c15": b}
        f = lambda v: "--" if not v else (f"{v[0]:.3f}" if len(v) == 1 else (", ".join(f"{x:.3f}" for x in sorted(v)) if len(v) == 2 else f"${np.mean(v):.3f}\\pm{np.std(v, ddof=1):.3f}$ ({len(v)})"))  # noqa: E731
        lines.append(f"{lab} & {f(a)} & {f(b)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "paper/tables/unet_objectives.tex").write_text("\n".join(lines) + "\n")
    (ROOT / "results/unet_objectives.json").write_text(json.dumps(out, indent=2))
    print("\n".join(lines))


BUDGET = [("30\\,000 at $c=60$ (the recipe)", "c=60", ["detector_30k_seed0", "detector_30k_seed1", "detector_30k_seed2", "detector_30k_seed3"]),
          ("100\\,000 at $c=60$", "c=60", ["detector_30k_c60big_seed0"]),
          ("500\\,000 at $c=60$ (published scale)", "c=60", ["detector_30k_c60pub_seed0"]),
          ("100\\,000 at $c=60$, augmented, cosine, 40 epochs", "c=60", ["detector_100k_c60aug_seed0"]),
          ("100\\,000 at $c=60$ with lens multipoles", "c=60", ["detector_100k_mpsub_seed0"]),
          ("100\\,000 at $c=30$", "c=30", ["detector_100k_c30_seed0"]),
          ("30\\,000 at $c=60$, fine-tuned on $c=15$", "c=15", ["detector_30k_c15ft_seed0", "detector_30k_c15ft_seed1"]),
          ("30\\,000 at $c=15$, 50 epochs, cosine", "c=15", ["detector_30k_c15cos_seed0"]),
          ("100\\,000 at $c=15$", "c=15", ["detector_30k_c15big_seed0"]),
          ("100\\,000 at $c=60$, ConvNeXt encoder", "c=60", ["detector_100k_c60convnext_seed0"]),
          ("100\\,000 at $c=60$, neural ratio estimator", "c=60", ["nre_100k_c60_seed0"])]


def _std(d):
    """The ratio estimator's results.json (train_image_detector.py) in the U-Net evaluator's layout."""
    if "roc_auc_fixed60_vs_no_subhalo" in d:
        return d
    return {"roc_auc_fixed60_vs_no_subhalo": d["auc_test_fixed60"], "completeness_c60_by_mass_bin": d["completeness_test_fixed60"],
            "completeness_c15_by_mass_bin": d["completeness_test_fixed15"],
            "paired_concentration_flip": {"detected_at_c60_not_c15": d["paired_flip"]["c60_not_c15"], "detected_at_c15_not_c60": d["paired_flip"]["c15_not_c60"]}}


def budget_table():
    rows = [r"\begin{tabular}{lccccc}", r"\toprule",
            r"training set & AUC & \multicolumn{2}{c}{compl.\ at $c{=}60$} & compl.\ at $c{=}15$ & flip \\",
            r" & $c{=}60$ vs clean & $10^{9}$--$10^{9.5}$ & $10^{9.5}$--$10^{10}$ & $10^{9.5}$--$10^{10}$ & $60\to15$ \\", r"\midrule"]
    out = {}
    for lab, tr, runs in BUDGET:
        if "ConvNeXt" in lab:
            rows.append(r"\midrule")
        R = [_std(json.loads((ROOT / "results" / r / "results.json").read_text())) for r in runs if (ROOT / "results" / r / "results.json").exists()]
        if not R:
            rows.append(f"{lab} & \\multicolumn{{5}}{{c}}{{running}} \\\\"); continue
        auc = [d["roc_auc_fixed60_vs_no_subhalo"] for d in R]
        c60 = [d["completeness_c60_by_mass_bin"]["9.5-10.0"]["completeness"] for d in R]
        c60l = [d["completeness_c60_by_mass_bin"]["9.0-9.5"]["completeness"] for d in R]
        c15 = [d["completeness_c15_by_mass_bin"]["9.5-10.0"]["completeness"] for d in R]
        lost = np.mean([d["paired_concentration_flip"]["detected_at_c60_not_c15"] for d in R]); gained = np.mean([d["paired_concentration_flip"]["detected_at_c15_not_c60"] for d in R])
        out[lab] = {"auc": auc, "c60": c60, "c60_9_9p5": c60l, "c15": c15, "flip": [lost, gained]}
        fmt = lambda v, pct=False: (f"{100*np.mean(v):.0f}" if pct else f"{np.mean(v):.3f}") + (f" ({len(v)})" if len(v) > 1 else "")  # noqa: E731
        rows.append(f"{lab} & {fmt(auc)} & {fmt(c60l, True)}\\% & {fmt(c60, True)}\\% & {fmt(c15, True)}\\% & {lost:.0f}\\,:\\,{gained:.0f} \\\\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "paper/tables/unet_budget.tex").write_text("\n".join(rows) + "\n")
    (ROOT / "results/unet_budget.json").write_text(json.dumps(out, indent=2))
    print("\n".join(rows))


if __name__ == "__main__":
    budget_table()
    main()
