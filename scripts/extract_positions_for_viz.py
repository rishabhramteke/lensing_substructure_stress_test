"""Extract WHERE each method thinks a subhalo is (not just whether it fired),
for the "Clump Atlas" spatial true-vs-detected visualization and the
localization-accuracy figure (`plot_localization_accuracy.py`). Neither
evaluate_detector.py nor evaluate_baseline_a.py needed this before -- both
only needed a scalar score/threshold for FPR & completeness. This adds the
position dimension, and a stricter "correct find" definition than a bare
score-threshold crossing: has_subhalo AND detected AND localized within
Ostdiek+2020's own 2-pixel criterion (see first_results.md's "Localization
accuracy" section for why this matters -- a naive score-only definition
credited Family A with 108 Tier-0 "correct finds" of which only 15 actually
land within 2 pixels of the truth).

    source ~/myenv/bin/activate
    python scripts/extract_positions_for_viz.py

Writes ONE file, data/sanity/explainer/positions_for_viz.json, with layers:
  ground_truth              [x, y, log10_mass]
  a_correct / c_correct     [guess_x, guess_y, log10_mass_or_null, true_x, true_y]
  a_false_alarm / c_false_alarm   [guess_x, guess_y, log10_mass_or_null]
  a_other / c_other         [true_x, true_y, log10_mass]  (missed or detected-but-mislocalized)
"a_"/"c_" = Family A (parametric scan) / Family C (U-Net). U-Net never
estimates mass, so its mass field is always null.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from detector.dataset import LensPatchDataset
from detector.unet import UNet

ROOT = Path(__file__).resolve().parents[1]
LOC_ARCSEC = 2.0 * 0.08  # Ostdiek+2020's 2-pixel localization criterion, HST/F160W 0.08"/px


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def r(v, n=3):
    return round(v, n) if v is not None else None


def unet_argmax_positions(model_path: Path, data_root: Path, population: str) -> list[dict]:
    """One record per image: model's own argmax pixel -> (x,y) in arcsec, plus the truth."""
    dev = device()
    ckpt = torch.load(model_path, map_location=dev)
    model = UNet(in_ch=1, base=ckpt.get("base", 16)).to(dev)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    ds = LensPatchDataset(data_root / population, scale=ckpt["scale"])
    center = (ds.num_pix - 1) / 2.0

    out = []
    with torch.no_grad():
        for i in range(len(ds)):
            img, _, _, _ = ds[i]
            prob = torch.sigmoid(model(img.unsqueeze(0).to(dev)))[0, 0].cpu().numpy()
            score = float(prob.max())
            py, px = np.unravel_index(np.argmax(prob), prob.shape)
            pred_x = (float(px) - center) * ds.pixel_scale
            pred_y = (float(py) - center) * ds.pixel_scale
            sub = ds.truths[i].get("subhalo")
            out.append({
                "true_x": sub["x"] if sub else None, "true_y": sub["y"] if sub else None,
                "true_log10_m": sub["log10_M200"] if sub else None,
                "pred_x": pred_x, "pred_y": pred_y, "score": score, "has_subhalo": sub is not None,
            })
    return out, json.loads((ROOT / "results/detector_v0_best/results.json").read_text())["threshold_at_10pct_fpr"]


def main():
    layers = {"ground_truth": [], "a_correct": [], "a_false_alarm": [], "a_other": [],
              "c_correct": [], "c_false_alarm": [], "c_other": []}

    truth60 = [json.loads(l) for l in open(ROOT / "data/test_fixed60/truth.jsonl")]
    for t in truth60:
        if t.get("subhalo"):
            layers["ground_truth"].append([r(t["subhalo"]["x"]), r(t["subhalo"]["y"]), r(t["subhalo"]["log10_M200"])])

    # ---- Family A: test_fixed60 (truth) + multipole_m4_a3 (false-alarm test), seed 0 ----
    a_thr = json.loads((ROOT / "results/baseline_a/summary/results.json").read_text())["threshold_delta_chi2_at_10pct_fpr"]
    for rec in [json.loads(l) for l in open(ROOT / "results/baseline_a/test_fixed60/scan_results.jsonl")]:
        if not rec["reliable_fit"]:
            continue
        detected = rec["delta_chi2"] >= a_thr
        if rec["has_subhalo"]:
            sub = truth60[rec["index"]]["subhalo"]
            d = float(np.hypot(rec["best_x"] - sub["x"], rec["best_y"] - sub["y"]))
            if detected and d < LOC_ARCSEC:
                layers["a_correct"].append([r(rec["best_x"]), r(rec["best_y"]), r(rec["best_log10_m"]), r(sub["x"]), r(sub["y"])])
            else:
                layers["a_other"].append([r(sub["x"]), r(sub["y"]), r(sub["log10_M200"])])
    for rec in [json.loads(l) for l in open(ROOT / "results/baseline_a/multipole_m4_a3/scan_results.jsonl")]:
        if rec["reliable_fit"] and not rec["has_subhalo"] and rec["delta_chi2"] >= a_thr:
            layers["a_false_alarm"].append([r(rec["best_x"]), r(rec["best_y"]), r(rec["best_log10_m"])])

    # ---- U-Net: same two populations, primary Tier-0 checkpoint ----
    unet_test60, unet_thr = unet_argmax_positions(ROOT / "checkpoints/unet_v0/model_best.pt", ROOT / "data", "test_fixed60")
    for p in unet_test60:
        if p["has_subhalo"]:
            detected = p["score"] >= unet_thr
            d = float(np.hypot(p["pred_x"] - p["true_x"], p["pred_y"] - p["true_y"])) if detected else None
            if detected and d < LOC_ARCSEC:
                layers["c_correct"].append([r(p["pred_x"]), r(p["pred_y"]), None, r(p["true_x"]), r(p["true_y"])])
            else:
                layers["c_other"].append([r(p["true_x"]), r(p["true_y"]), r(p["true_log10_m"])])
    unet_mp_a3, _ = unet_argmax_positions(ROOT / "checkpoints/unet_v0/model_best.pt", ROOT / "data", "multipole_m4_a3")
    for p in unet_mp_a3:
        if not p["has_subhalo"] and p["score"] >= unet_thr:
            layers["c_false_alarm"].append([r(p["pred_x"]), r(p["pred_y"])])

    out_path = ROOT / "data/sanity/explainer/positions_for_viz.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(layers, separators=(",", ":")))
    print(f"wrote {out_path}")
    for k, v in layers.items():
        print(f"  {k:16s} n={len(v)}")


if __name__ == "__main__":
    main()
