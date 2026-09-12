"""Evaluate the Family-D NPE posterior across the same 5 populations and
metrics as scripts/evaluate_detector.py / evaluate_baseline_a.py.

    source ~/myenv/bin/activate
    python scripts/evaluate_baseline_d.py --model checkpoints/npe_v0/posterior.pkl --out results/baseline_d/summary --n 300

Detection statistic: P(theta > DETECT_THRESH | x), the posterior mass placed
above a value close to the FLOOR=6.0 "no subhalo" sentinel (see
src/baseline_d/__init__.py) -- estimated by sampling.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from detector.dataset import load_population, normalize

MASS_BINS = [(8.0, 8.5), (8.5, 9.0), (9.0, 9.5), (9.5, 10.0), (10.0, 10.5), (10.5, 11.0)]
DETECT_THRESH = 7.5  # posterior mass above this = "detected"; floor is 6.0, prior support starts at 8.0 for real subhalos
N_POSTERIOR_SAMPLES = 300


def score_population(posterior, x_t, n_samples=N_POSTERIOR_SAMPLES):
    scores = np.zeros(len(x_t), dtype=np.float32)
    for i in range(len(x_t)):
        s = posterior.sample((n_samples,), x=x_t[i], show_progress_bars=False)
        scores[i] = float((s > DETECT_THRESH).float().mean())
    return scores


def load_subset(name, data_root, scale, n, seed):
    images, truths, pixel_scale, num_pix, manifest = load_population(data_root / name)
    rng = np.random.default_rng(seed)
    n = min(n, len(truths))
    idx = np.sort(rng.choice(len(truths), size=n, replace=False)) if n < len(truths) else np.arange(len(truths))
    x = normalize(images[idx], scale).reshape(len(idx), -1)
    return torch.from_numpy(x).float(), [truths[i] for i in idx], idx


def completeness_by_bin(scores, truths, thr):
    out = {}
    for lo, hi in MASS_BINS:
        sel = [i for i, t in enumerate(truths) if t.get("subhalo") and lo <= t["subhalo"]["log10_M200"] < hi]
        out[f"{lo}-{hi}"] = {"n": len(sel), "completeness": float(np.mean(scores[sel] >= thr)) if sel else None}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    with open(args.model, "rb") as f:
        posterior = pickle.load(f)
    log = json.loads((args.model.parent / "training_log.json").read_text())
    scale = log["scale"]
    print(f"loaded posterior, training scale={scale:.4f}")

    x60, t60, idx60 = load_subset("test_fixed60", args.data_root, scale, args.n, args.seed)
    x15, t15, idx15 = load_subset("test_fixed15", args.data_root, scale, args.n, args.seed)
    xns, tns, _ = load_subset("no_subhalo", args.data_root, scale, args.n, args.seed + 1)
    xa1, ta1, _ = load_subset("multipole_m4_a1", args.data_root, scale, args.n, args.seed + 2)
    xa3, ta3, _ = load_subset("multipole_m4_a3", args.data_root, scale, args.n, args.seed + 3)

    print("scoring populations (this samples the posterior per image, may take a while)...")
    s60 = score_population(posterior, x60)
    s15 = score_population(posterior, x15)
    sns = score_population(posterior, xns)
    sa1 = score_population(posterior, xa1)
    sa3 = score_population(posterior, xa3)

    thr = float(np.quantile(sns, 0.90))
    fpr_calib = float((sns >= thr).mean())

    comp60 = completeness_by_bin(s60, t60, thr)
    comp15 = completeness_by_bin(s15, t15, thr)

    # paired flip: idx60/idx15 are the same subsample indices (same seed) -> same underlying lenses
    common = [i for i in range(len(idx60)) if idx60[i] == idx15[i] and t60[i].get("subhalo")]
    det60 = s60[common] >= thr
    det15 = s15[common] >= thr
    paired_flip = {"n_pairs": len(common), "detected_at_c60": int(det60.sum()), "detected_at_c15": int(det15.sum()),
                   "detected_at_c60_not_c15": int((det60 & ~det15).sum()), "detected_at_c15_not_c60": int((det15 & ~det60).sum())}

    results = {
        "threshold": thr, "fpr_at_threshold_calibration_set": fpr_calib,
        "completeness_c60_by_mass_bin": comp60, "completeness_c15_by_mass_bin": comp15,
        "paired_concentration_flip": paired_flip,
        "confounder_fpr": {"no_subhalo (calibration set)": fpr_calib,
                           "multipole m=4, a=0.01*thetaE": float((sa1 >= thr).mean()),
                           "multipole m=4, a=0.03*thetaE": float((sa3 >= thr).mean())},
        "n": args.n,
    }
    (args.out / "results.json").write_text(json.dumps(results, indent=2))

    mids = [(lo + hi) / 2 for lo, hi in MASS_BINS]
    c60 = [comp60[f"{lo}-{hi}"]["completeness"] for lo, hi in MASS_BINS]
    c15 = [comp15[f"{lo}-{hi}"]["completeness"] for lo, hi in MASS_BINS]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(mids, c60, "o-", label="c = 60 (Tsang+2024 fiducial)", color="#1e64a8")
    ax.plot(mids, c15, "s--", label="c = 15 (their low-c ablation)", color="#c8177a")
    ax.axhline(0.10, color="gray", ls=":", lw=1, label="FPR operating point")
    ax.set_xlabel("log10(subhalo M200 / Msun)"); ax.set_ylabel("completeness @ FPR=10%")
    ax.set_title("Family D (NPE) — completeness vs. concentration")
    ax.legend(fontsize=8); ax.set_ylim(-0.02, 1.02); fig.tight_layout()
    fig.savefig(args.out / "completeness_vs_mass.png", dpi=140); plt.close(fig)

    print(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}/results.json + completeness_vs_mass.png")


if __name__ == "__main__":
    main()
