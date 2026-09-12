"""Generate a Tier-0 dataset: images + hidden truth + a reproducibility manifest.

    source ~/myenv/bin/activate
    python scripts/generate_tier0.py --config tsang_fixed60 --n 200 --out data/tier0_fixed60 --seed 0
    python scripts/generate_tier0.py --config tsang_fixed15 --n 200 --out data/tier0_fixed15 --seed 0
    python scripts/generate_tier0.py --config no_subhalo     --n 200 --out data/tier0_control --seed 1
    python scripts/generate_tier0.py --config multipole_m4_a3 --n 200 --out data/tier2_multipole --seed 2

Every array is index-aligned with `truth.jsonl` (line i <-> image i). `manifest.json`
records the exact config and package versions, per problem_statement.md §4.4
("every config, seed and evaluation script released").
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lensing.config import (
    SimConfig, tier0_tsang, tier0_no_subhalo, tier1_cosmos, tier1_cosmos_no_subhalo,
    tier1_cosmos_multipole_confounder, tier2_multipole_confounder,
    tier2_lens_light, tier2_lens_light_no_subhalo, tier2_lens_light_multipole,
)
from lensing.simulate import LensRenderer, sample_truth

CONFIGS = {
    "tsang_fixed60": lambda: tier0_tsang("fixed60"),
    "tsang_fixed15": lambda: tier0_tsang("fixed15"),
    "tsang_cdm": lambda: tier0_tsang("cdm"),
    "no_subhalo": tier0_no_subhalo,
    "multipole_m4_a1": lambda: tier2_multipole_confounder(am_over_thetaE=0.01, m=4),
    "multipole_m4_a3": lambda: tier2_multipole_confounder(am_over_thetaE=0.03, m=4),
    "multipole_m3_a3": lambda: tier2_multipole_confounder(am_over_thetaE=0.03, m=3),
    "lenslight_fixed60": lambda: tier2_lens_light("fixed60"),
    "lenslight_no_subhalo": tier2_lens_light_no_subhalo,
    "lenslight_multipole_m4_a3": lambda: tier2_lens_light_multipole(am_over_thetaE=0.03, m=4),
    "cosmos_fixed60": lambda: tier1_cosmos("fixed60"),
    "cosmos_fixed15": lambda: tier1_cosmos("fixed15"),
    "cosmos_cdm": lambda: tier1_cosmos("cdm"),
    "cosmos_no_subhalo": tier1_cosmos_no_subhalo,
    "cosmos_multipole_m4_a1": lambda: tier1_cosmos_multipole_confounder(am_over_thetaE=0.01, m=4),
    "cosmos_multipole_m4_a3": lambda: tier1_cosmos_multipole_confounder(am_over_thetaE=0.03, m=4),
}


def _asdict(cfg: SimConfig) -> dict:
    return dataclasses.asdict(cfg)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", choices=sorted(CONFIGS), required=True)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--noiseless-only", action="store_true", help="skip the noisy variant (faster; for control populations you only need the ablation)")
    args = p.parse_args()

    cfg = CONFIGS[args.config]()
    args.out.mkdir(parents=True, exist_ok=True)
    renderer = LensRenderer(cfg)
    rng = np.random.default_rng(args.seed)

    H = W = cfg.instrument.num_pix
    full_noisy = np.zeros((args.n, H, W), dtype=np.float32) if not args.noiseless_only else None
    full_noiseless = np.zeros((args.n, H, W), dtype=np.float32)
    control_noiseless = np.zeros((args.n, H, W), dtype=np.float32)  # no_subhalo if a subhalo is present, else no_multipole, else = smooth

    t0 = time.time()
    truths = []
    for i in range(args.n):
        truth = sample_truth(cfg, rng)
        out = renderer.render(truth, noiseless_only=args.noiseless_only, seed=int(rng.integers(0, 2**31 - 1)))
        control_key = "no_subhalo" if truth.get("subhalo") else ("no_multipole" if truth.get("multipole") else "smooth")
        full_noiseless[i] = out["noiseless"]["full"]
        control_noiseless[i] = out["noiseless"][control_key]
        if not args.noiseless_only:
            full_noisy[i] = out["noisy"]["full"]
        truth["_control_variant"] = control_key
        truth["_index"] = i
        truths.append(truth)
        if (i + 1) % 50 == 0 or i + 1 == args.n:
            rate = (i + 1) / (time.time() - t0)
            print(f"  {i+1}/{args.n}  ({rate:.1f} images/s)")

    np.save(args.out / "images_full_noiseless.npy", full_noiseless)
    np.save(args.out / "images_control_noiseless.npy", control_noiseless)
    if not args.noiseless_only:
        np.save(args.out / "images_full_noisy.npy", full_noisy)
    with open(args.out / "truth.jsonl", "w") as f:
        for t in truths:
            f.write(json.dumps(t) + "\n")

    manifest = {
        "config_name": args.config,
        "n": args.n,
        "seed": args.seed,
        "noiseless_only": args.noiseless_only,
        "image_shape": [H, W],
        "sim_config": _asdict(cfg),
        "kwargs_band": renderer.kwargs_band,
        "python": sys.version,
        "platform": platform.platform(),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "packages": {},
    }
    for pkg in ("lenstronomy", "numpy", "scipy", "astropy"):
        try:
            manifest["packages"][pkg] = __import__(pkg).__version__
        except Exception:
            manifest["packages"][pkg] = "unknown"
    with open(args.out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\nwrote {args.n} images to {args.out}/  ({elapsed:.1f}s total, {args.n/elapsed:.1f} images/s)")


if __name__ == "__main__":
    main()
