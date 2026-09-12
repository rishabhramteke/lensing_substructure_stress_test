"""2D figure for the paper: does a 'detection' actually land on the true
subhalo position, or just cross a threshold? Motivated by a real finding
(2026-09-11): Family A's completeness numbers are inflated by a detection
threshold so loose that 18 of its 108 nominal test_fixed60 "correct finds"
have delta_chi2 < 0 and only 13.9% land within Ostdiek+2020's 2-pixel
localization criterion -- vs the U-Net's 56.7%. See first_results.md.

    source ~/myenv/bin/activate
    python scripts/plot_localization_accuracy.py                                            # Family A seed 0, U-Net v0
    python scripts/plot_localization_accuracy.py --a-root results/baseline_a_seed1 --tag seed200
    python scripts/plot_localization_accuracy.py --a-root results/baseline_a_c60 --tag c60        # oracle-concentration scan
    python scripts/plot_localization_accuracy.py --unet-ckpt checkpoints/unet_30k_seed0/model_best.pt \
        --unet-results results/detector_30k_seed0/results.json --tag unet30k

Family A's per-seed 10%-FPR threshold is read from <a-root>/summary/results.json when
present, otherwise recomputed exactly as evaluate_baseline_a.py does. The false-alarm
population (multipole_m4_a3) is optional -- concentration-variant runs may not have it.

Writes results/localization/localization_accuracy[_<tag>].{png,pdf} + summary[_<tag>].json
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

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "localization"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "src"))
from detector.dataset import LensPatchDataset  # noqa: E402
from detector.unet import UNet  # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
                     "font.size": 8, "axes.titlesize": 9, "legend.fontsize": 6.6, "pdf.fonttype": 42})

LOCALIZED_PX = 2.0
PIXEL_SCALE = 0.08
LOCALIZED_ARCSEC = LOCALIZED_PX * PIXEL_SCALE


def family_a_threshold(a_root: Path) -> float:
    summary = a_root / "summary" / "results.json"
    if summary.exists():
        return json.loads(summary.read_text())["threshold_delta_chi2_at_10pct_fpr"]
    recs = [json.loads(l) for l in open(a_root / "no_subhalo" / "scan_results.jsonl")]
    neg = np.array([r["delta_chi2"] for r in recs if r["reliable_fit"]])
    return float(np.quantile(neg, 0.90))


def family_a_data(a_root: Path, truths):
    thr = family_a_threshold(a_root)
    recs = [json.loads(l) for l in open(a_root / "test_fixed60" / "scan_results.jsonl")]
    tp = []
    for r in recs:
        if r["reliable_fit"] and r["has_subhalo"] and r["delta_chi2"] >= thr:
            sub = truths[r["index"]]["subhalo"]
            tp.append((sub["x"], sub["y"], r["best_x"], r["best_y"], r["delta_chi2"], r["best_log10_m"], sub["log10_M200"]))
    fp_path = a_root / "multipole_m4_a3" / "scan_results.jsonl"
    fp = []
    if fp_path.exists():
        fp = [(r["best_x"], r["best_y"]) for r in (json.loads(l) for l in open(fp_path))
              if r["reliable_fit"] and not r["has_subhalo"] and r["delta_chi2"] >= thr]
    return tp, fp, thr


def unet_detections(ckpt_path: Path, thr: float, population: str):
    """(true_x, true_y, pred_x, pred_y, score) for every detection on subhalo-bearing images,
    and (pred_x, pred_y) for every detection on subhalo-free images."""
    dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    ckpt = torch.load(ckpt_path, map_location=dev)
    model = UNet(in_ch=1, base=ckpt.get("base", 16)).to(dev)
    model.load_state_dict(ckpt["model_state"]); model.eval()
    ds = LensPatchDataset(ROOT / "data" / population, scale=ckpt["scale"])
    center = (ds.num_pix - 1) / 2.0
    tp, fp = [], []
    with torch.no_grad():
        for i in range(len(ds)):
            img, _, _, _ = ds[i]
            prob = torch.sigmoid(model(img.unsqueeze(0).to(dev)))[0, 0].cpu().numpy()
            score = float(prob.max())
            if score < thr:
                continue
            py, px = np.unravel_index(np.argmax(prob), prob.shape)
            gx, gy = (px - center) * ds.pixel_scale, (py - center) * ds.pixel_scale
            sub = ds.truths[i].get("subhalo")
            if sub:
                tp.append((sub["x"], sub["y"], gx, gy, score))
            else:
                fp.append((gx, gy))
    return tp, fp


def panel(ax, tp, fp, title, stat_line):
    theta = np.linspace(0, 2 * np.pi, 200)
    for r in (0.6, 1.3):
        ax.plot(r * np.cos(theta), r * np.sin(theta), color="#3a4368", lw=0.8, ls=":", zorder=0)
    for row in tp:
        tx, ty, gx, gy = row[0], row[1], row[2], row[3]
        d = np.hypot(gx - tx, gy - ty)
        color = "#2fbf71" if d < LOCALIZED_ARCSEC else "#e8973a"
        ax.plot([tx, gx], [ty, gy], color=color, lw=0.9, alpha=0.75, zorder=1)
        ax.plot(gx, gy, "o", color=color, ms=3.2, zorder=2)
    if tp:
        ax.plot([t[0] for t in tp], [t[1] for t in tp], "x", color="#8b93ad", ms=4, mew=1.0, zorder=1)
    if fp:
        ax.plot([f[0] for f in fp], [f[1] for f in fp], "*", color="#ff5470", ms=6.5, mew=0, alpha=0.85, zorder=3)
    ax.set_xlim(-1.8, 1.8); ax.set_ylim(-1.8, 1.8); ax.set_aspect("equal")
    ax.set_xlabel("x (arcsec)"); ax.set_ylabel("x (arcsec)".replace("x", "y"))
    ax.set_title(title, fontweight="bold")
    ax.text(0.02, 0.02, stat_line + (f" · {len(fp)} false alarms" if fp else ""), transform=ax.transAxes, fontsize=6.8, color="#444", va="bottom")


def shared_legend(fig):
    """One legend for all panels (referee: per-panel legends covered the data)."""
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color="#2fbf71", lw=1.6, label=f"correct find, localized (≤{LOCALIZED_PX:.0f} px of truth)"),
               Line2D([], [], color="#e8973a", lw=1.6, label="correct find, mislocalized (right call, wrong spot)"),
               Line2D([], [], marker="x", color="#8b93ad", ls="none", ms=5, mew=1.0, label="true subhalo position"),
               Line2D([], [], marker="*", color="#ff5470", ls="none", ms=7, mew=0, label="false alarm (no subhalo present; multipole population)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.02))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a-root", type=Path, default=ROOT / "results/baseline_a")
    p.add_argument("--unet-ckpt", type=Path, default=ROOT / "checkpoints/unet_v0/model_best.pt")
    p.add_argument("--unet-results", type=Path, default=ROOT / "results/detector_v0_best/results.json")
    p.add_argument("--tag", default="")
    p.add_argument("--b-root", type=Path, default=None,
                   help="optional Family B (potential-correction) root with the same scan_results.jsonl layout -> 3 panels")
    p.add_argument("--a-label", default="Family A (parametric scan)")
    args = p.parse_args()
    suffix = f"_{args.tag}" if args.tag else ""

    truths = [json.loads(l) for l in open(ROOT / "data/test_fixed60/truth.jsonl")]
    a_tp, a_fp, a_thr = family_a_data(args.a_root, truths)
    unet_thr = json.loads(args.unet_results.read_text())["threshold_at_10pct_fpr"]
    c_tp, _ = unet_detections(args.unet_ckpt, unet_thr, "test_fixed60")
    _, c_fp = unet_detections(args.unet_ckpt, unet_thr, "multipole_m4_a3")

    a_d = np.array([np.hypot(t[2] - t[0], t[3] - t[1]) for t in a_tp])
    c_d = np.array([np.hypot(t[2] - t[0], t[3] - t[1]) for t in c_tp])
    a_loc = a_d < LOCALIZED_ARCSEC
    a_merr = np.array([t[5] - t[6] for t in a_tp])

    b_tp = b_fp = None
    if args.b_root is not None:
        b_tp, b_fp, b_thr = family_a_data(args.b_root, truths)
        b_d = np.array([np.hypot(t[2] - t[0], t[3] - t[1]) for t in b_tp])
    ncol = 3 if b_tp is not None else 2
    fig, axes = plt.subplots(1, ncol, figsize=(7.09 if ncol == 2 else 10.2, 3.55))
    panel(axes[0], a_tp, a_fp, args.a_label,
          f"localized ≤{LOCALIZED_PX:.0f} px: {100*a_loc.mean():.0f}% of {len(a_tp)} correct finds · median offset {np.median(a_d):.2f}″")
    if b_tp is not None:
        panel(axes[1], b_tp, b_fp, "Family B (potential correction)",
              f"localized ≤{LOCALIZED_PX:.0f} px: {100*(b_d<LOCALIZED_ARCSEC).mean():.0f}% of {len(b_tp)} correct finds · median offset {np.median(b_d):.2f}″")
    panel(axes[-1], c_tp, c_fp, "Family C (U-Net)",
          f"localized ≤{LOCALIZED_PX:.0f} px: {100*(c_d<LOCALIZED_ARCSEC).mean():.0f}% of {len(c_tp)} correct finds · median offset {np.median(c_d):.2f}″")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    shared_legend(fig)
    fig.savefig(OUT / f"localization_accuracy{suffix}.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / f"localization_accuracy{suffix}.pdf", bbox_inches="tight")

    summary = {
        "family_a": {"a_root": str(args.a_root), "threshold_delta_chi2": a_thr, "n_correct_find": len(a_tp),
                     "frac_localized_2px": float(a_loc.mean()) if len(a_d) else None, "n_localized": int(a_loc.sum()),
                     "median_offset_arcsec": float(np.median(a_d)) if len(a_d) else None,
                     "n_negative_delta_chi2_among_correct": int(sum(1 for t in a_tp if t[4] < 0)),
                     "localized_mass_error_dex": {"n": int(a_loc.sum()),
                                                  "median": float(np.median(a_merr[a_loc])) if a_loc.any() else None,
                                                  "min": float(a_merr[a_loc].min()) if a_loc.any() else None,
                                                  "max": float(a_merr[a_loc].max()) if a_loc.any() else None,
                                                  "n_underestimated": int((a_merr[a_loc] < 0).sum()),
                                                  "n_within_0p5dex": int((np.abs(a_merr[a_loc]) < 0.5).sum())},
                     "n_false_alarms_shown": len(a_fp)},
        "family_c_unet": {"unet_ckpt": str(args.unet_ckpt), "n_correct_find": len(c_tp),
                          "frac_localized_2px": float((c_d < LOCALIZED_ARCSEC).mean()) if len(c_d) else None,
                          "median_offset_arcsec": float(np.median(c_d)) if len(c_d) else None, "n_false_alarms_shown": len(c_fp)},
    }
    (OUT / f"summary{suffix}.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {OUT}/localization_accuracy{suffix}.{{png,pdf}} + summary{suffix}.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
