"""Lens-light experiment (referee point 3, 2026-09-11): what does imperfect lens-light
subtraction do to Family A and to the U-Net?

Populations `data/lenslight_{no_subhalo,fixed60,multipole_m4_a3}` carry a two-component
lens light. Family A fits a SINGLE Sersic jointly with the mass model
(`results/baseline_a_lenslight/<pop>/`), and writes `lens_light_subtracted.npy` = data minus
that fitted light -- the image a practitioner's pipeline would hand to a pixel detector.

Family A:  reliability, 10%-FPR threshold, confounder FPR, completeness, absolute-threshold FPR
U-Net v0:  scored on (a) the raw lens-light images and (b) the subtracted images, at
           (i) its ORIGINAL Tier-0 threshold and (ii) a threshold recalibrated on the
           subtracted clean population.

    python scripts/evaluate_lens_light.py
"""
import json, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from detector.dataset import normalize  # noqa: E402
from detector.unet import UNet  # noqa: E402

BINS = [(8.0, 8.5), (8.5, 9.0), (9.0, 9.5), (9.5, 10.0), (10.0, 10.5), (10.5, 11.0)]
A_ROOT = ROOT / "results/baseline_a_lenslight"
SUFFIX = ""          # population suffix, e.g. "_s8" for the second lens-light seed set (set from the CLI)
CKPT = ROOT / "checkpoints/unet_v0/model_best.pt"
ORIG_THR = json.loads((ROOT / "results/detector_v0_best/results.json").read_text())["threshold_at_10pct_fpr"]


def truths(pop):
    return [json.loads(l) for l in open(ROOT / "data" / (pop + SUFFIX) / "truth.jsonl")]


def comp_by_bin(scores, trs, thr):
    out = {}
    for lo, hi in BINS:
        sel = [s for s, t in zip(scores, trs) if t.get("subhalo") and lo <= t["subhalo"]["log10_M200"] < hi]
        out[f"{lo}-{hi}"] = {"n": len(sel), "completeness": float(np.mean(np.array(sel) >= thr)) if sel else None}
    return out


def family_a(root=None):
    root = Path(root) if root is not None else A_ROOT
    def load(pop):
        p = root / (pop + SUFFIX) / "scan_results.jsonl"
        return [json.loads(l) for l in open(p)] if p.exists() else None
    ns, t60, mp = load("lenslight_no_subhalo"), load("lenslight_fixed60"), load("lenslight_multipole_m4_a3")
    if not ns:
        return None
    rel_ns = [r for r in ns if r["reliable_fit"]]
    thr = float(np.quantile([r["delta_chi2"] for r in rel_ns], 0.90))
    out = {"threshold_10pct_fpr": thr, "n_unreliable": {"no_subhalo": len(ns) - len(rel_ns)},
           "median_chi2_per_dof_clean": float(np.median([r["chi2_smooth_per_dof"] for r in ns])),
           "fpr_clean_abs": {"20": float(np.mean([r["delta_chi2"] >= 20 for r in rel_ns])), "100": float(np.mean([r["delta_chi2"] >= 100 for r in rel_ns]))}}
    if mp:
        rel = [r for r in mp if r["reliable_fit"]]
        out["n_unreliable"]["multipole_m4_a3"] = len(mp) - len(rel)
        out["fpr_multipole_a3"] = {"10pct": float(np.mean([r["delta_chi2"] >= thr for r in rel])), "20": float(np.mean([r["delta_chi2"] >= 20 for r in rel])), "100": float(np.mean([r["delta_chi2"] >= 100 for r in rel]))}
    # gate lifted: the chi2/N<10 rule would exclude most lens-light fits, so also score EVERY fit
    all_ns = [r["delta_chi2"] for r in ns]
    thr_all = float(np.quantile(all_ns, 0.90))
    out["gate_lifted"] = {"threshold_10pct_fpr": thr_all}
    if mp:
        out["gate_lifted"]["fpr_multipole_a3"] = {"10pct": float(np.mean([r["delta_chi2"] >= thr_all for r in mp])), "20": float(np.mean([r["delta_chi2"] >= 20 for r in mp])), "100": float(np.mean([r["delta_chi2"] >= 100 for r in mp]))}
        out["gate_lifted"]["fpr_clean_abs"] = {"20": float(np.mean([r["delta_chi2"] >= 20 for r in ns])), "100": float(np.mean([r["delta_chi2"] >= 100 for r in ns]))}
    if t60:
        pos = [r for r in t60 if r["has_subhalo"]]
        out["gate_lifted"]["completeness_c60"] = {f"{lo}-{hi}": {"n": len([r for r in pos if lo <= r["log10_M200_true"] < hi]),
                                                                  "completeness": (float(np.mean([r["delta_chi2"] >= thr_all for r in pos if lo <= r["log10_M200_true"] < hi])) if any(lo <= r["log10_M200_true"] < hi for r in pos) else None)} for lo, hi in BINS}
    if t60:
        rel = [r for r in t60 if r["reliable_fit"] and r["has_subhalo"]]
        out["n_unreliable"]["fixed60"] = sum(1 for r in t60 if not r["reliable_fit"])
        out["completeness_c60"] = {f"{lo}-{hi}": {"n": len([r for r in rel if lo <= r["log10_M200_true"] < hi]),
                                                    "completeness": (float(np.mean([r["delta_chi2"] >= thr for r in rel if lo <= r["log10_M200_true"] < hi])) if any(lo <= r["log10_M200_true"] < hi for r in rel) else None)} for lo, hi in BINS}
        out["completeness_c60_conservative_top2"] = {f"{lo}-{hi}": float(np.mean([(r["reliable_fit"] and r["delta_chi2"] >= thr) for r in t60 if r["has_subhalo"] and lo <= r["log10_M200_true"] < hi])) for lo, hi in BINS[-2:]}
    return out


def unet_scores(images):
    dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    ckpt = torch.load(CKPT, map_location=dev)
    model = UNet(in_ch=1, base=ckpt.get("base", 16)).to(dev); model.load_state_dict(ckpt["model_state"]); model.eval()
    x = torch.from_numpy(normalize(np.asarray(images, dtype=np.float32), ckpt["scale"])).unsqueeze(1)
    scores = []
    with torch.no_grad():
        for i in range(0, len(x), 64):
            prob = torch.sigmoid(model(x[i:i + 64].to(dev)))[:, 0]
            scores.append(prob.reshape(len(prob), -1).max(dim=1).values.cpu().numpy())
    return np.concatenate(scores)


def unet():
    out = {"original_threshold": ORIG_THR}
    # (a) raw lens-light images, full populations
    raw = {pop: unet_scores(np.load(ROOT / "data" / (pop + SUFFIX) / "images_full_noisy.npy")) for pop in ("lenslight_no_subhalo", "lenslight_fixed60", "lenslight_multipole_m4_a3")}
    tr = {pop: truths(pop) for pop in raw}
    out["raw_at_original_threshold"] = {"fpr_clean": float(np.mean(raw["lenslight_no_subhalo"] >= ORIG_THR)),
                                        "fpr_multipole_a3": float(np.mean(raw["lenslight_multipole_m4_a3"] >= ORIG_THR)),
                                        "completeness_c60": comp_by_bin(raw["lenslight_fixed60"], tr["lenslight_fixed60"], ORIG_THR)}
    thr_raw = float(np.quantile(raw["lenslight_no_subhalo"], 0.90))
    out["raw_recalibrated"] = {"threshold": thr_raw, "fpr_multipole_a3": float(np.mean(raw["lenslight_multipole_m4_a3"] >= thr_raw)),
                               "completeness_c60": comp_by_bin(raw["lenslight_fixed60"], tr["lenslight_fixed60"], thr_raw),
                               "auc_c60_vs_clean": auc(raw["lenslight_fixed60"], tr["lenslight_fixed60"], raw["lenslight_no_subhalo"])}
    # (b) subtracted images (Family A's fitted single-Sersic light removed), subsample of the same populations
    sub, subtr = {}, {}
    for pop in raw:
        p = A_ROOT / (pop + SUFFIX) / "lens_light_subtracted.npy"
        if p.exists():
            idx = np.load(A_ROOT / (pop + SUFFIX) / "subsample_indices.npy")
            sub[pop] = unet_scores(np.load(p)); subtr[pop] = [tr[pop][i] for i in idx]
    if "lenslight_no_subhalo" in sub:
        s = sub
        out["subtracted_at_original_threshold"] = {"fpr_clean": float(np.mean(s["lenslight_no_subhalo"] >= ORIG_THR))}
        thr_sub = float(np.quantile(s["lenslight_no_subhalo"], 0.90))
        out["subtracted_recalibrated"] = {"threshold": thr_sub, "n": {k: int(len(v)) for k, v in s.items()}}
        if "lenslight_multipole_m4_a3" in s:
            out["subtracted_at_original_threshold"]["fpr_multipole_a3"] = float(np.mean(s["lenslight_multipole_m4_a3"] >= ORIG_THR))
            out["subtracted_recalibrated"]["fpr_multipole_a3"] = float(np.mean(s["lenslight_multipole_m4_a3"] >= thr_sub))
        if "lenslight_fixed60" in s:
            out["subtracted_at_original_threshold"]["completeness_c60"] = comp_by_bin(s["lenslight_fixed60"], subtr["lenslight_fixed60"], ORIG_THR)
            out["subtracted_recalibrated"]["completeness_c60"] = comp_by_bin(s["lenslight_fixed60"], subtr["lenslight_fixed60"], thr_sub)
            out["subtracted_recalibrated"]["auc_c60_vs_clean"] = auc(s["lenslight_fixed60"], subtr["lenslight_fixed60"], s["lenslight_no_subhalo"])
    # (c) the same with the correctly specified double-Sersic subtraction, if that run exists
    A2 = ROOT / ("results/baseline_a_lenslight2" + SUFFIX)
    if (A2 / ("lenslight_no_subhalo" + SUFFIX) / "lens_light_subtracted.npy").exists():
        s2, t2 = {}, {}
        for pop in raw:
            p = A2 / (pop + SUFFIX) / "lens_light_subtracted.npy"
            if p.exists():
                idx = np.load(A2 / (pop + SUFFIX) / "subsample_indices.npy"); s2[pop] = unet_scores(np.load(p)); t2[pop] = [tr[pop][i] for i in idx]
        thr2 = float(np.quantile(s2["lenslight_no_subhalo"], 0.90))
        out["double_sersic_subtracted"] = {"fpr_clean_at_original_threshold": float(np.mean(s2["lenslight_no_subhalo"] >= ORIG_THR)),
                                           "recalibrated_threshold": thr2,
                                           "fpr_multipole_a3_recalibrated": float(np.mean(s2["lenslight_multipole_m4_a3"] >= thr2)) if "lenslight_multipole_m4_a3" in s2 else None,
                                           "completeness_c60_recalibrated": comp_by_bin(s2["lenslight_fixed60"], t2["lenslight_fixed60"], thr2) if "lenslight_fixed60" in s2 else None,
                                           "auc_c60_vs_clean": auc(s2["lenslight_fixed60"], t2["lenslight_fixed60"], s2["lenslight_no_subhalo"]) if "lenslight_fixed60" in s2 else None}
    return out


def auc(pos_scores, pos_truths, neg_scores):
    pos = np.array([s for s, t in zip(pos_scores, pos_truths) if t.get("subhalo")])
    neg = np.asarray(neg_scores)
    return float(np.mean([(p > neg).mean() + 0.5 * (p == neg).mean() for p in pos]))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-unet", action="store_true")
    ap.add_argument("--suffix", default="", help="population suffix for a second seed set, e.g. _s8 (roots get the same suffix)")
    args = ap.parse_args()
    global SUFFIX, A_ROOT
    SUFFIX = args.suffix
    A_ROOT = ROOT / ("results/baseline_a_lenslight" + SUFFIX)
    extra = {}
    for key, rel in (("family_a_double_sersic", "results/baseline_a_lenslight2" + SUFFIX), ("family_b_single_sersic", "results/baseline_b_lenslight" + SUFFIX + "/fitted"), ("family_b_double_sersic", "results/baseline_b_lenslight2" + SUFFIX + "/fitted")):
        if (ROOT / rel / ("lenslight_no_subhalo" + SUFFIX) / "scan_results.jsonl").exists():
            extra[key] = family_a(ROOT / rel)
    res = {"family_a": family_a(), "unet_v0": (None if args.skip_unet else unet()), **extra,
           "note": "True lens light is two-component (bulge n=3-5 + n=1 envelope); Family A fits ONE Sersic jointly with the mass model; 'subtracted' = data minus that fitted light. U-Net v0 was trained on Tier-0 images without lens light."}
    (ROOT / ("results/lens_light_experiment" + SUFFIX + ".json")).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
