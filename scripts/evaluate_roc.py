"""Threshold-independent cross-family comparison: ROC curves and AUC (referee round 10, point 3).

The paper's common 10%-FPR calibration forces a negative threshold on Family A's signed statistic,
which then needs the null floor to repair, leaving two parallel bookkeeping systems. A ROC is
threshold-free and removes that: it compares the three families on ranking alone. We report AUC per
family and per population, plus completeness at 10% and at 1% false-positive rate, all on the lenses
Families A and B both retain so the samples match.

    python scripts/evaluate_roc.py
-> results/roc.json, paper/tables/roc.tex
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from detector.dataset import LensPatchDataset  # noqa: E402
from detector.unet import UNet  # noqa: E402

A_ROOT, B_ROOT = "results/baseline_a_full", "results/baseline_b_full/fitted"
SEEDS = {"detector_v0_best": "checkpoints/unet_v0", "detector_v1_seed1": "checkpoints/unet_v1_seed1",
         "detector_v2_seed2": "checkpoints/unet_v2_seed2", "detector_v3_seed3": "checkpoints/unet_v3_seed3"}
POPS = ("test_fixed60", "test_fixed15", "multipole_m4_a3")


def scans(root, pop):
    p = ROOT / root / pop / "scan_results.jsonl"
    return {r["index"]: r for r in (json.loads(l) for l in open(p))} if p.exists() else None


def auc_and_tpr(neg, pos, fpr_targets=(0.10, 0.01)):
    """AUC by the Mann-Whitney identity, plus TPR at fixed FPR (threshold from the negatives)."""
    neg, pos = np.asarray(neg, float), np.asarray(pos, float)
    if len(neg) < 5 or len(pos) < 5:
        return None
    allv = np.concatenate([neg, pos])
    r = np.argsort(np.argsort(allv)) + 1.0
    # average ranks for ties
    order = np.argsort(allv)
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = np.mean(r[order[i:j + 1]])
        i = j + 1
    rpos = r[len(neg):].sum()
    auc = (rpos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    out = {"auc": float(auc), "n_neg": int(len(neg)), "n_pos": int(len(pos))}
    for f in fpr_targets:
        thr = float(np.quantile(neg, 1 - f))
        out[f"tpr_at_fpr{f:g}"] = float(np.mean(pos > thr))
        out[f"threshold_at_fpr{f:g}"] = thr
    return out


def unet_scores(pop):
    dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    per_seed = []
    for run, ckdir in SEEDS.items():
        ck_p = ROOT / ckdir / "model_best.pt"
        if not ck_p.exists():
            continue
        ck = torch.load(ck_p, map_location=dev)
        m = UNet(in_ch=1, base=ck.get("base", 16)).to(dev)
        m.load_state_dict(ck["model_state"]); m.eval()
        ds = LensPatchDataset(ROOT / "data" / pop, scale=ck["scale"])
        sc = {}
        with torch.no_grad():
            for i in range(len(ds)):
                sc[i] = float(torch.sigmoid(m(ds[i][0].unsqueeze(0).to(dev))).max())
        per_seed.append(sc)
    return per_seed


def main():
    truths = {p: [json.loads(l) for l in open(ROOT / "data" / p / "truth.jsonl")] for p in ("no_subhalo",) + POPS}
    A = {p: scans(A_ROOT, p) for p in ("no_subhalo",) + POPS}
    B = {p: scans(B_ROOT, p) for p in ("no_subhalo",) + POPS}
    if not all(A.values()) or not all(B.values()):
        print("missing runs"); return
    keep = {}
    for p in ("no_subhalo",) + POPS:
        both = set(A[p]) & set(B[p])
        keep[p] = sorted(i for i in both if A[p][i]["reliable_fit"] and B[p][i]["reliable_fit"])
    res = {"n_common": {p: len(keep[p]) for p in keep}, "families": {}}

    neg_idx = keep["no_subhalo"]
    for fam, S in (("A", A), ("B", B)):
        res["families"][fam] = {}
        neg = [S["no_subhalo"][i]["delta_chi2"] for i in neg_idx]
        for pop in POPS:
            sel = [i for i in keep[pop] if (truths[pop][i].get("subhalo") is not None) or pop.startswith("multipole")]
            pos = [S[pop][i]["delta_chi2"] for i in sel if S[pop][i].get("has_subhalo") or pop.startswith("multipole")]
            r = auc_and_tpr(neg, pos)
            if r:
                res["families"][fam][pop] = r
    U = {p: unet_scores(p) for p in ("no_subhalo",) + POPS}
    res["families"]["C"] = {}
    for pop in POPS:
        rows = []
        for k, sc_neg in enumerate(U["no_subhalo"]):
            sc_pos = U[pop][k]
            neg = [sc_neg[i] for i in keep["no_subhalo"] if i in sc_neg]
            pos = [sc_pos[i] for i in keep[pop] if i in sc_pos and (truths[pop][i].get("subhalo") or pop.startswith("multipole"))]
            r = auc_and_tpr(neg, pos)
            if r:
                rows.append(r)
        if rows:
            res["families"]["C"][pop] = {k: float(np.mean([x[k] for x in rows])) for k in rows[0] if isinstance(rows[0][k], float)}
            res["families"]["C"][pop]["auc_std"] = float(np.std([x["auc"] for x in rows]))
            res["families"]["C"][pop]["n_seeds"] = len(rows)
    (ROOT / "results/roc.json").write_text(json.dumps(res, indent=2))

    names = {"A": "A scan", "B": r"B linear $\delta\psi$", "C": "C U-Net"}
    plab = {"test_fixed60": "$c{=}60$", "test_fixed15": "$c{=}15$", "multipole_m4_a3": "multipole"}
    lines = [r"\begin{tabular}{@{}llccc@{}}", r"\toprule",
             r"family & population & AUC & TPR at 10\% FPR & TPR at 1\% FPR \\", r"\midrule"]
    for fam in ("A", "B", "C"):
        first = True
        for pop in POPS:
            d = res["families"][fam].get(pop)
            if not d:
                continue
            lines.append(f"{names[fam] if first else ''} & {plab[pop]} & {d['auc']:.3f} & "
                         f"{100*d['tpr_at_fpr0.1']:.0f}\\% & {100*d['tpr_at_fpr0.01']:.0f}\\% \\\\")
            first = False
        lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "paper/tables/roc.tex").write_text("\n".join(lines) + "\n")
    for fam in ("A", "B", "C"):
        for pop in POPS:
            d = res["families"][fam].get(pop)
            if d:
                print(f"  {fam} {pop:18s} AUC {d['auc']:.3f}  TPR@10% {100*d['tpr_at_fpr0.1']:5.1f}%  TPR@1% {100*d['tpr_at_fpr0.01']:5.1f}%  (n {d.get('n_pos','?')})")


if __name__ == "__main__":
    main()
