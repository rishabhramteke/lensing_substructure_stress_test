"""Control experiment: does either detector recognize a *real subhalo's*
lensing signature, or does it just react to any localized bright anomaly,
physical or not?

Motivation (2026-09-11, user question): both families' failure modes found
so far are consistent with "reacts to local anomalies in general" rather
than "recognizes the specific dipole/quadrupole shape a real NFW perturber
produces". This tests it directly: inject a **non-physical** flux pattern (no
lensing physics at all -- just added pixel flux) into otherwise clean
`no_subhalo` images, at the same annulus real subhalos are placed in, and see
whether either detector "detects" it anyway.

Two decoy shapes (the self-review asked for an asymmetric one, since a real
subhalo's imprint is dipole-like and a symmetric blob is the easy case for the
U-Net):
  gaussian -- one positive ~1-px Gaussian bump
  dipole   -- a positive and a negative lobe of equal amplitude, 0.9 px apart
              along a random direction (the shape a subhalo residual has, but
              with no gravitational deflection behind it)

    source ~/myenv/bin/activate
    python scripts/noise_decoy_control.py --n 100 --shape gaussian --seed 42
    python scripts/noise_decoy_control.py --n 100 --shape gaussian --seed 43 --tag seed43
    python scripts/noise_decoy_control.py --n 100 --shape dipole   --seed 42 --tag dipole
    python scripts/noise_decoy_control.py --unet-ckpt checkpoints/unet_30k_seed0/model_best.pt \
        --unet-results results/detector_30k_seed0/results.json --tag unet30k

Writes results/noise_decoy_control[_<tag>]/results.json + a figure.
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
from detector.dataset import load_population, normalize
from detector.unet import UNet
from baseline_a.fit import fit_smooth, scan_subhalo
from lenstronomy.SimulationAPI.sim_api import SimAPI

ROOT = Path(__file__).resolve().parents[1]


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def inject_decoy(image, pixel_scale, num_pix, theta_E, noise_std, amp_sigma, rng, shape="gaussian"):
    """Pure-pixel decoy -- no lens physics. Same placement annulus as real subhalos
    (0.6-1.3 theta_E); amplitude in multiples of this image's own noise_std."""
    r = rng.uniform(0.6, 1.3) * theta_E
    ang = rng.uniform(0, 2 * np.pi)
    x, y = r * np.cos(ang), r * np.sin(ang)
    center = (num_pix - 1) / 2.0
    px, py = center + x / pixel_scale, center + y / pixel_scale
    yy, xx = np.mgrid[0:num_pix, 0:num_pix]
    s = 1.0  # ~1 native pixel

    def g(cx, cy):
        return np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s ** 2)))

    if shape == "gaussian":
        bump = g(px, py)
    elif shape == "dipole":
        d = rng.uniform(0, 2 * np.pi)
        dx, dy = 0.9 * np.cos(d), 0.9 * np.sin(d)
        bump = g(px + dx, py + dy) - g(px - dx, py - dy)
    else:
        raise ValueError(shape)
    return image + amp_sigma * noise_std * bump, {"x": x, "y": y, "amp_sigma": amp_sigma, "shape": shape}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--amps", type=float, nargs="+", default=[3.0, 6.0, 10.0])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shape", choices=["gaussian", "dipole"], default="gaussian")
    p.add_argument("--tag", default="")
    p.add_argument("--unet-ckpt", type=Path, default=ROOT / "checkpoints/unet_v0/model_best.pt")
    p.add_argument("--unet-results", type=Path, default=ROOT / "results/detector_v0_best/results.json")
    p.add_argument("--a-summary", type=Path, default=ROOT / "results/baseline_a/summary/results.json")
    args = p.parse_args()

    images, truths, pixel_scale, num_pix, manifest = load_population(ROOT / "data/no_subhalo")
    kwargs_band = manifest["kwargs_band"]
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(truths), size=args.n, replace=False)

    dev = device()
    ckpt = torch.load(args.unet_ckpt, map_location=dev)
    model = UNet(in_ch=1, base=ckpt.get("base", 16)).to(dev)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    unet_scale = ckpt["scale"]
    unet_thr = json.loads(args.unet_results.read_text())["threshold_at_10pct_fpr"]
    a_thr = json.loads(args.a_summary.read_text())["threshold_delta_chi2_at_10pct_fpr"]

    results = {amp: {"unet_fpr": None, "a_fpr": None, "unet_scores": [], "a_deltas": []} for amp in args.amps}
    rng_decoy = np.random.default_rng(args.seed + 1)
    rng_fit = np.random.default_rng(args.seed + 1000)

    for amp in args.amps:
        unet_flags, a_flags = [], []
        for k, i in enumerate(idx):
            truth = truths[i]
            theta_E = truth["lens_macro"]["theta_E"]
            sim = SimAPI(num_pix=num_pix, kwargs_single_band=kwargs_band,
                         kwargs_model={"lens_model_list": ["EPL", "SHEAR"], "source_light_model_list": ["SERSIC_ELLIPSE"]})
            noise_std = float(np.median(sim.estimate_noise(images[i])))
            decoy_img, _ = inject_decoy(images[i], pixel_scale, num_pix, theta_E, noise_std, amp, rng_decoy, args.shape)

            x_t = torch.from_numpy(normalize(decoy_img[None], unet_scale)).unsqueeze(0).to(dev)
            with torch.no_grad():
                score = float(torch.sigmoid(model(x_t)).max())
            unet_flags.append(score >= unet_thr)
            results[amp]["unet_scores"].append(score)

            vec, chi2_smooth, image_model, ns = fit_smooth(decoy_img, truth, kwargs_band, num_pix, rng=rng_fit, maxiter=60)
            if (chi2_smooth / (num_pix * num_pix)) < 10.0:
                best = scan_subhalo(decoy_img, ns, vec, kwargs_band, num_pix, theta_E=theta_E, chi2_smooth=chi2_smooth, concentration=15.0)
                a_flags.append(best["delta_chi2"] >= a_thr)
                results[amp]["a_deltas"].append(best["delta_chi2"])
            if (k + 1) % 25 == 0:
                print(f"  shape={args.shape} amp={amp}  {k+1}/{args.n}")

        results[amp]["unet_fpr"] = float(np.mean(unet_flags))
        results[amp]["a_fpr"] = float(np.mean(a_flags)) if a_flags else None
        results[amp]["n_a_reliable"] = len(a_flags)
        print(f"shape={args.shape} amp={amp}sigma  U-Net FPR={results[amp]['unet_fpr']:.1%}  Family-A FPR={results[amp]['a_fpr']}")

    out = ROOT / ("results/noise_decoy_control" + (f"_{args.tag}" if args.tag else ""))
    out.mkdir(parents=True, exist_ok=True)
    summary = {str(amp): {k: v for k, v in results[amp].items() if k not in ("unet_scores", "a_deltas")} for amp in args.amps}
    summary["config"] = {"shape": args.shape, "seed": args.seed, "n": args.n, "unet_ckpt": str(args.unet_ckpt)}
    summary["baseline_fpr_clean_no_subhalo"] = {"unet": 0.10, "family_a": 0.10}
    (out / "results.json").write_text(json.dumps(summary, indent=2))

    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    ax.plot(args.amps, [results[a]["unet_fpr"] for a in args.amps], "o-", color="#1e64a8", label="U-Net (Family C)")
    ax.plot(args.amps, [results[a]["a_fpr"] for a in args.amps], "s--", color="#c8177a", label="Family A (parametric scan)")
    ax.axhline(0.10, color="gray", ls=":", label="baseline FPR on clean images (10%)")
    ax.set_xlabel("decoy amplitude (multiples of local noise σ)")
    ax.set_ylabel("fraction flagged as \"dark matter detected\"")
    ax.set_title(f"Non-physical {args.shape} decoy (no lensing signature)\ninjected into clean no-subhalo images")
    ax.legend(fontsize=9); ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(out / "noise_decoy_control.png", dpi=140)
    print(f"\nwrote {out}/results.json + noise_decoy_control.png")


if __name__ == "__main__":
    main()
