"""The concentration-coincidence test at every aperture for every family, with a correction for the number of apertures
(referee round 21, M7). Reads the per-lens detections that completeness_vs_signal.py stores, recomputes the projected
TNFW mass within each aperture, and gives the unpaired pooled-chi^2 p and the paired permutation p (the one to read) at
each of seven apertures. Holm's step-down procedure over the seven apertures of a family marks which rejections survive
at family-wise alpha = 0.05.

    python scripts/evaluate_aperture_scan.py     -> results/aperture_scan.json, paper/tables/aperture_scan.tex
"""
import json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import completeness_vs_signal as CS  # noqa: E402

APERTURES = (0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4)
N_PERM = 4000


def holm(ps, alpha=0.05):
    order = np.argsort(ps); m = len(ps); rej = [False] * m
    for k, j in enumerate(order):
        if ps[j] <= alpha / (m - k):
            rej[j] = True
        else:
            break
    return rej


def main():
    res = json.loads((ROOT / "results/completeness_vs_signal.json").read_text())
    truths = {c: [json.loads(l) for l in open(ROOT / f"data/test_fixed{c}/truth.jsonl")] for c in (60, 15)}
    CS.BIN_WIDTH["log10_Mproj_ap"] = 0.5
    out = {}
    for fam in ("A", "B", "C"):
        F = res["families"][fam]
        rows = []
        for R in APERTURES:
            for c in (60, 15):
                F[f"c{c}"]["log10_Mproj_ap"] = [float(np.log10(CS.projected_mass_msun(truths[c][i]["subhalo"], R))) for i in F[f"c{c}"]["index"]]
            J = CS.joint_chi2(CS.compare({k: F[k] for k in ("c60", "c15")}, "log10_Mproj_ap"))
            P = CS.paired_permutation(res, fam, "log10_Mproj_ap", n_perm=N_PERM)
            rows.append({"aperture": R, "p_unpaired": J["p_value"] if J else None, "p_perm": P["p_permutation"] if P else None})
        rej = holm([r["p_perm"] for r in rows])
        for r, x in zip(rows, rej):
            r["rejected_holm"] = bool(x)
        out[fam] = rows
        print(fam, [(r["aperture"], f"{r['p_perm']:.4f}", r["rejected_holm"]) for r in rows])
    (ROOT / "results/aperture_scan.json").write_text(json.dumps(out, indent=2))

    def fmt(p):
        return r"$<$0.001" if p < 0.001 else f"{p:.3f}"
    lines = [r"\begin{tabular}{l" + "c" * len(APERTURES) + "}", r"\toprule",
             r"aperture & " + " & ".join(f"${R:g}''$" for R in APERTURES) + r" \\", r"\midrule"]
    names = {"A": "A, joint scan", "B": "B, free-form", "C": "C, U-Net"}
    for fam in "ABC":
        lines.append(names[fam] + " & " + " & ".join((r"\textbf{" + fmt(r["p_perm"]) + "}") if r["rejected_holm"] else fmt(r["p_perm"]) for r in out[fam]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "paper/tables/aperture_scan.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
