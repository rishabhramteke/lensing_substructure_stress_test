"""U-Net trained at low concentration (referee round 19, 2026-09-18): is the c=15 collapse a distribution shift
(curable by training at c=15) or a signal the pixel detector cannot learn at this budget?

Reads the per-seed aggregates of three training populations, all 30 000 images from the same seed (same lenses,
sources, subhalo masses and positions; only the concentration differs): fixed c=60 (the paper's primary set),
fixed c=15, and concentrations drawn from the Dutton & Maccio (2014) relation. Each is scored on the standard
Tier-0 test populations (c=60, c=15, clean, multipoles) by evaluate_detector.py / aggregate_unet_seeds.py.

    python scripts/evaluate_unet_lowc.py -> results/unet_lowc.json, paper/tables/unet_lowc.tex
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BINS = [(8.0, 8.5), (8.5, 9.0), (9.0, 9.5), (9.5, 10.0), (10.0, 10.5), (10.5, 11.0)]
SETS = [("c60", "$c{=}60$", "results/aggregate_30k/summary.json"),
        ("c15", "$c{=}15$", "results/aggregate_30k_cm/summary.json".replace("_cm", "_c15")),
        ("cm", "$c$--$M$", "results/aggregate_30k_cm/summary.json")]


def load(rel):
    p = ROOT / rel
    return json.loads(p.read_text()) if p.exists() else None


def main():
    S = {k: (lab, load(rel)) for k, lab, rel in SETS}
    out = {k: {"label": lab, "present": d is not None} for k, (lab, d) in S.items()}
    for k, (lab, d) in S.items():
        if not d:
            continue
        # best validation loss per seed from the training logs (the mechanism of a null result: a loss plateau)
        vals = []
        for run in d.get("seed_runs", []):
            ck = ROOT / "checkpoints" / run.replace("detector_", "unet_") / "training_log.json"
            if ck.exists():
                vals.append(json.loads(ck.read_text())["best_val_loss"])
        d["val_loss_best_mean"] = sum(vals) / len(vals) if vals else None
        out[k].update({"n_seeds": d["n_seeds"], "auc": d["auc"], "val_loss_best_mean": d["val_loss_best_mean"], "completeness_c60": d["completeness_c60"], "completeness_c15": d["completeness_c15"],
                       "confounder_fpr": d["confounder_fpr"], "paired_flip": d.get("paired_flip") or d.get("paired_concentration_flip")})
    (ROOT / "results/unet_lowc.json").write_text(json.dumps(out, indent=2))

    def ms(x, nd=0):
        v = x.get("values")
        if v and len(v) == 2:   # two seeds: the two values, not a standard deviation (referee round 23)
            lo_, hi_ = sorted(v)
            return f"${100*lo_:.{nd}f}$, ${100*hi_:.{nd}f}$"
        return f"${100*x['mean']:.{nd}f}\\pm{100*x['std']:.{nd}f}$"
    def row(label, f):
        return label + " & " + " & ".join((f(S[k][1]) if S[k][1] else "--") for k, _, _ in SETS) + " \\\\"
    def flip(d):
        pf = d.get("paired_flip") or d.get("paired_concentration_flip")
        if not pf:
            return "--"
        lost, gained = pf.get("c60_not_c15", pf.get("lost")), pf.get("c15_not_c60", pf.get("gained"))
        if isinstance(lost, dict):
            return f"{lost['mean']:.0f}\\,:\\,{gained['mean']:.0f}"
        return f"{lost}\\,:\\,{gained}"
    lines = ["\\begin{tabular}{@{}lccc@{}}", "\\toprule", "training set & " + " & ".join(lab for _, lab, _ in SETS) + " \\\\", "\\midrule",
             row("seeds", lambda d: str(d["n_seeds"])),
             row("AUC, $c{=}60$ vs.\\ clean", lambda d: (", ".join(f"${x:.3f}$" for x in sorted(d['auc']['values'])) if len(d['auc'].get('values', [])) == 2 else f"${d['auc']['mean']:.3f}\\pm{d['auc']['std']:.3f}$")),
             row("best validation loss", lambda d: f"{d['val_loss_best_mean']:.3f}" if d.get("val_loss_best_mean") is not None else "--"),
             "\\multicolumn{4}{@{}l}{\\emph{completeness at 10\\% FPR, tested at $c{=}60$}} \\\\"]
    for lo, hi in BINS[1:]:
        lines.append(row(f"\\quad $\\log_{{10}}M_{{200}}$: {lo}--{hi}", lambda d, k=f"{lo}-{hi}": ms(d["completeness_c60"][k])))
    lines.append("\\multicolumn{4}{@{}l}{\\emph{completeness at 10\\% FPR, tested at $c{=}15$}} \\\\")
    for lo, hi in BINS[1:]:
        lines.append(row(f"\\quad $\\log_{{10}}M_{{200}}$: {lo}--{hi}", lambda d, k=f"{lo}-{hi}": ms(d["completeness_c15"][k])))
    lines.append(row("paired flip $c{=}60\\to15$, lost\\,:\\,gained", flip))
    lines.append(row("FPR, multipole $a_4{=}0.03\\,\\thetaE$", lambda d: ms(d["confounder_fpr"]["multipole m=4, a=0.03*thetaE"])))
    lines += ["\\bottomrule", "\\end{tabular}"]
    (ROOT / "paper/tables/unet_lowc.tex").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
