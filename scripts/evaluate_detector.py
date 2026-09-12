"""Evaluate the trained U-Net across every RQ this stress test cares about.

    source ~/myenv/bin/activate
    python scripts/evaluate_detector.py --model checkpoints/unet_v0/model.pt --out results/detector_v0

Produces:
  - completeness_vs_mass.png   (RQ4: c=60 vs c=15, same lenses, paired)
  - roc_fixed60.png            (in-distribution ROC, for context)
  - confounder_fpr.png         (RQ2: FPR under zero-subhalo confounders)
  - results.json / results.md  (every number, plus the RQ6 reproducibility ledger)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from detector.dataset import LensPatchDataset
from detector.unet import UNet

MASS_BINS = [(8.0, 8.5), (8.5, 9.0), (9.0, 9.5), (9.5, 10.0), (10.0, 10.5), (10.5, 11.0)]


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def score_population(model, ds: LensPatchDataset, dev, batch_size=64) -> np.ndarray:
    """Image-level score = max sigmoid(logit) over the frame — Tsang+2024's
    exact evaluation convention: '"maximum pixel probability" defines image
    score; threshold determines classification.'"""
    scores = np.zeros(len(ds), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(ds), batch_size):
            idx = range(start, min(start + batch_size, len(ds)))
            imgs = torch.stack([ds[i][0] for i in idx]).to(dev)
            logits = model(imgs)
            probs = torch.sigmoid(logits)
            scores[list(idx)] = probs.amax(dim=(1, 2, 3)).cpu().numpy()
    return scores


def threshold_for_fpr(neg_scores: np.ndarray, target_fpr: float = 0.10) -> float:
    return float(np.quantile(neg_scores, 1 - target_fpr))


def roc_points(pos_scores, neg_scores):
    thresholds = np.unique(np.concatenate([pos_scores, neg_scores]))[::-1]
    tprs, fprs = [], []
    for t in thresholds:
        tprs.append((pos_scores >= t).mean())
        fprs.append((neg_scores >= t).mean())
    return np.array(fprs), np.array(tprs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=Path("data"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    dev = device()
    ckpt = torch.load(args.model, map_location=dev)
    model = UNet(in_ch=1, base=ckpt.get("base", 16)).to(dev)
    model.load_state_dict(ckpt["model_state"])
    scale = ckpt["scale"]
    print(f"loaded model, trained scale={scale:.4f}, device={dev}")

    def load(name):
        return LensPatchDataset(args.data_root / name, scale=scale)

    ds_test60 = load("test_fixed60")
    ds_test15 = load("test_fixed15")
    ds_no_sub = load("no_subhalo")
    ds_mp_a1 = load("multipole_m4_a1")
    ds_mp_a3 = load("multipole_m4_a3")

    print("scoring populations...")
    s_test60 = score_population(model, ds_test60, dev)
    s_test15 = score_population(model, ds_test15, dev)
    s_no_sub = score_population(model, ds_no_sub, dev)
    s_mp_a1 = score_population(model, ds_mp_a1, dev)
    s_mp_a3 = score_population(model, ds_mp_a3, dev)

    # ---- calibrate the operating threshold on the CLEAN negative pool, exactly
    # as Tsang+2024 report at "FPR = 10%" ----
    thr = threshold_for_fpr(s_no_sub, 0.10)
    fpr_at_thr_calib_set = float((s_no_sub >= thr).mean())

    def mass_of(truths, i):
        sub = truths[i].get("subhalo")
        return sub["log10_M200"] if sub else None

    def completeness_by_bin(ds, scores, thr):
        out = {}
        for lo, hi in MASS_BINS:
            sel = [i for i in range(len(ds)) if (m := mass_of(ds.truths, i)) is not None and lo <= m < hi]
            if not sel:
                out[f"{lo}-{hi}"] = {"n": 0, "completeness": None}
                continue
            comp = float((scores[sel] >= thr).mean())
            out[f"{lo}-{hi}"] = {"n": len(sel), "completeness": comp}
        return out

    comp60 = completeness_by_bin(ds_test60, s_test60, thr)
    comp15 = completeness_by_bin(ds_test15, s_test15, thr)

    # ---- RQ4, paired: for identical lens systems, does completeness collapse
    # from c=60 to c=15 in the same mass bin? (test_fixed60/15 share a seed,
    # so index i is the *same* lens/source/subhalo mass&position in both.) ----
    paired_sub_idx = [i for i in range(len(ds_test60)) if ds_test60.truths[i].get("subhalo")]
    paired_flip = None
    if paired_sub_idx:
        det60 = s_test60[paired_sub_idx] >= thr
        det15 = s_test15[paired_sub_idx] >= thr
        paired_flip = {
            "n_pairs": len(paired_sub_idx),
            "detected_at_c60": int(det60.sum()),
            "detected_at_c15": int(det15.sum()),
            "detected_at_c60_not_c15": int((det60 & ~det15).sum()),
            "detected_at_c15_not_c60": int((det15 & ~det60).sum()),
        }

    # ---- RQ2: false-positive rate under each confounder, at the SAME threshold ----
    fpr_confounders = {
        "no_subhalo (calibration set)": fpr_at_thr_calib_set,
        "multipole m=4, a=0.01*thetaE": float((s_mp_a1 >= thr).mean()),
        "multipole m=4, a=0.03*thetaE": float((s_mp_a3 >= thr).mean()),
    }

    # ---- plots ----
    fig, ax = plt.subplots(figsize=(6, 4.2))
    mids = [(lo + hi) / 2 for lo, hi in MASS_BINS]
    c60 = [comp60[f"{lo}-{hi}"]["completeness"] for lo, hi in MASS_BINS]
    c15 = [comp15[f"{lo}-{hi}"]["completeness"] for lo, hi in MASS_BINS]
    ax.plot(mids, c60, "o-", label="c = 60 (Tsang+2024 fiducial)", color="#1e64a8")
    ax.plot(mids, c15, "s--", label="c = 15 (their low-c ablation)", color="#c8177a")
    ax.axhline(0.10, color="gray", ls=":", lw=1, label="FPR operating point")
    ax.set_xlabel("log10(subhalo M200 / Msun)"); ax.set_ylabel(f"completeness @ FPR={0.10:.0%}")
    ax.set_title("RQ4 — completeness vs. concentration (paired lenses)")
    ax.legend(fontsize=8); ax.set_ylim(-0.02, 1.02); fig.tight_layout()
    fig.savefig(args.out / "completeness_vs_mass.png", dpi=140); plt.close(fig)

    pos60 = s_test60[[i for i in range(len(ds_test60)) if ds_test60.truths[i].get("subhalo")]]
    fprs, tprs = roc_points(pos60, s_no_sub)
    auc = float(np.trapezoid(tprs, fprs))  # fprs/tprs already ascending (thresholds swept high->low)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot(fprs, tprs, color="#1e64a8"); ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set_xlabel("false positive rate (no_subhalo)"); ax.set_ylabel("true positive rate (test_fixed60, all masses)")
    ax.set_title(f"ROC — in-distribution (c=60), AUC={auc:.3f}"); fig.tight_layout()
    fig.savefig(args.out / "roc_fixed60.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    names = list(fpr_confounders.keys()); vals = [fpr_confounders[k] for k in names]
    colors = ["#8e93a3", "#ffb86b", "#ff6fb0"]
    ax.bar(range(len(names)), vals, color=colors)
    ax.axhline(0.10, color="k", ls=":", lw=1, label="nominal 10% operating point")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("false positive rate"); ax.set_title("RQ2 — confounder-induced false positives\n(zero subhalos present)")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(args.out / "confounder_fpr.png", dpi=140); plt.close(fig)

    results = {
        "threshold_at_10pct_fpr": thr,
        "fpr_at_threshold_calibration_set": fpr_at_thr_calib_set,
        "completeness_c60_by_mass_bin": comp60,
        "completeness_c15_by_mass_bin": comp15,
        "paired_concentration_flip": paired_flip,
        "confounder_fpr": fpr_confounders,
        "roc_auc_fixed60_vs_no_subhalo": auc,
        "n_test_fixed60": len(ds_test60), "n_test_fixed15": len(ds_test15),
        "n_no_subhalo": len(ds_no_sub), "n_multipole_a1": len(ds_mp_a1), "n_multipole_a3": len(ds_mp_a3),
    }
    with open(args.out / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    lines = ["# Detector v0 — evaluation results\n", f"Threshold calibrated for 10% FPR on `no_subhalo` (n={len(ds_no_sub)}): **{thr:.4f}**  ·  ROC AUC (in-distribution, all masses) = **{auc:.3f}**\n",
             "\n## RQ4 — completeness vs. concentration (paired lenses, same mass & position)\n",
             "| mass bin (log10 Msun) | n | completeness c=60 | completeness c=15 |", "|---|---|---|---|"]
    for lo, hi in MASS_BINS:
        k = f"{lo}-{hi}"
        n = comp60[k]["n"]
        v60 = comp60[k]["completeness"]; v15 = comp15[k]["completeness"]
        s60 = f'{v60:.1%}' if v60 is not None else '—'
        s15 = f'{v15:.1%}' if v15 is not None else '—'
        lines.append(f"| {lo}-{hi} | {n} | {s60} | {s15} |")
    if paired_flip:
        lines += ["", f"**Paired flip test** ({paired_flip['n_pairs']} identical lenses, only concentration changed): "
                  f"detected at c=60: {paired_flip['detected_at_c60']}, at c=15: {paired_flip['detected_at_c15']}. "
                  f"{paired_flip['detected_at_c60_not_c15']} lenses flipped from detected->missed when concentration dropped; "
                  f"{paired_flip['detected_at_c15_not_c60']} flipped the other way."]
    lines += ["\n## RQ2 — confounder-induced false-positive rate (zero subhalos present)\n",
              "| condition | FPR |", "|---|---|"]
    for k, v in fpr_confounders.items():
        lines.append(f"| {k} | {v:.1%} |")
    (args.out / "results.md").write_text("\n".join(lines))

    print(json.dumps(results, indent=2))
    print(f"\nwrote plots + results.json + results.md to {args.out}/")


if __name__ == "__main__":
    main()
