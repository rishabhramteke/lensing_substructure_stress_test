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
    SimConfig, tier0_tsang, tier0_no_subhalo, tier0_population, tier0_euclid, tier1_cosmos, tier1_cosmos_no_subhalo,
    tier1_cosmos_multipole_confounder, tier2_multipole_confounder,
    tier2_lens_light, tier2_lens_light_no_subhalo, tier2_lens_light_multipole, tier0_shallow, tier0_los_halo, tier0_tidal_track, tier0_psf, tier0_realistic, tier0_clumpy_source, tier2_elliptical_multipole, tier0_inverse_crime, tier0_multipole_subhalo, tier0_supersampled,
)
from lensing.simulate import LensRenderer, sample_truth

def _depth(cfg, exposure_time, tag):
    """A misspecification arm at a survey-like depth (referee round 26, M2): the same lenses at `exposure_time`."""
    cfg.instrument.exposure_time = exposure_time
    cfg.tier = cfg.tier + f"_{int(exposure_time)}s"
    return cfg


def _psf_only(cfg):
    """The physical half of the inverse-crime control: a 10% wider PSF than the fitters assume, rendered at the
    fitters' own 1x sampling (referee round 26, M8)."""
    cfg.instrument.render_supersampling = 1
    cfg.tier = cfg.tier.replace("invcrime", "psferr")
    return cfg


CONFIGS = {
    "tsang_fixed60": lambda: tier0_tsang("fixed60"),
    "tsang_fixed30": lambda: tier0_tsang("fixed30"),
    "tsang_fixed15": lambda: tier0_tsang("fixed15"),
    "tsang_cdm": lambda: tier0_tsang("cdm"),
    "no_subhalo": tier0_no_subhalo,
    "tidal_c15": lambda: tier0_tidal_track(15.0),
    "tidal_c30": lambda: tier0_tidal_track(30.0),
    "los_halo_z025": lambda: tier0_los_halo(0.25),
    "los_halo_z075": lambda: tier0_los_halo(0.75),
    "shallow_fixed60": lambda: tier0_shallow("fixed60"),
    "shallow_fixed15": lambda: tier0_shallow("fixed15"),
    "shallow_no_subhalo": lambda: tier0_shallow(no_subhalo=True),
    "shallow_multipole_m4_a3": lambda: tier0_shallow(no_subhalo=True, am_over_thetaE=0.03, m=4),
    # intermediate depth (referee round 19): 600 s, a snapshot-length HST exposure, one third of the preset signal-to-noise
    "shallow600_fixed60": lambda: tier0_shallow("fixed60", exposure_time=600.0),
    "shallow600_fixed15": lambda: tier0_shallow("fixed15", exposure_time=600.0),
    "shallow600_no_subhalo": lambda: tier0_shallow(no_subhalo=True, exposure_time=600.0),
    "shallow600_multipole_m4_a3": lambda: tier0_shallow(no_subhalo=True, exposure_time=600.0, am_over_thetaE=0.03, m=4),
    "psf015_fixed60": lambda: tier0_psf(0.15, "fixed60"),
    "psf015_fixed15": lambda: tier0_psf(0.15, "fixed15"),
    "psf015_no_subhalo": lambda: tier0_psf(0.15, no_subhalo=True),
    "psf015_multipole_m4_a3": lambda: tier0_psf(0.15, no_subhalo=True, am_over_thetaE=0.03, m=4),
    # combined realism arm (referee round 21): 0.15" PSF + 600 s + two-component lens light
    "realistic_fixed60": lambda: tier0_realistic("fixed60"),
    "realistic_fixed15": lambda: tier0_realistic("fixed15"),
    "realistic_no_subhalo": lambda: tier0_realistic(no_subhalo=True),
    "realistic_multipole_m4_a3": lambda: tier0_realistic(no_subhalo=True, am_over_thetaE=0.03, m=4),
    # inverse-crime control and multipole-plus-subhalo lenses (referee round 23)
    "invcrime_fixed60": lambda: tier0_inverse_crime("fixed60"),
    "invcrime_no_subhalo": lambda: tier0_inverse_crime(no_subhalo=True),
    "invcrime_multipole_m4_a3": lambda: tier0_inverse_crime(no_subhalo=True, am_over_thetaE=0.03, m=4),
    "super_fixed60": lambda: tier0_supersampled("fixed60"),
    "super_no_subhalo": lambda: tier0_supersampled(no_subhalo=True),
    "super_multipole_m4_a3": lambda: tier0_supersampled(no_subhalo=True, am_over_thetaE=0.03, m=4),
    "mpsub_fixed60": lambda: tier0_multipole_subhalo("fixed60"),
    "mpsub_no_subhalo": lambda: tier0_multipole_subhalo(no_subhalo=True),
    # source misspecification and out-of-family multipole (referee round 22)
    "clumpy600_fixed60": lambda: _depth(tier0_clumpy_source("fixed60"), 600.0, "clumpy600"),
    "clumpy600_no_subhalo": lambda: _depth(tier0_clumpy_source(no_subhalo=True), 600.0, "clumpy600"),
    "clumpy600_multipole_m4_a3": lambda: _depth(tier0_clumpy_source(no_subhalo=True, am_over_thetaE=0.03, m=4), 600.0, "clumpy600"),
    "super600_fixed60": lambda: _depth(tier0_supersampled("fixed60"), 600.0, "super600"),
    "super600_no_subhalo": lambda: _depth(tier0_supersampled(no_subhalo=True), 600.0, "super600"),
    "super600_multipole_m4_a3": lambda: _depth(tier0_supersampled(no_subhalo=True, am_over_thetaE=0.03, m=4), 600.0, "super600"),
    "psferr_fixed60": lambda: _psf_only(tier0_inverse_crime("fixed60")),
    "psferr_no_subhalo": lambda: _psf_only(tier0_inverse_crime(no_subhalo=True)),
    "psferr_multipole_m4_a3": lambda: _psf_only(tier0_inverse_crime(no_subhalo=True, am_over_thetaE=0.03, m=4)),
    "clumpy_fixed60": lambda: tier0_clumpy_source("fixed60"),
    "clumpy_no_subhalo": lambda: tier0_clumpy_source(no_subhalo=True),
    "clumpy_multipole_m4_a3": lambda: tier0_clumpy_source(no_subhalo=True, am_over_thetaE=0.03, m=4),
    "ellmultipole_m4_a3": lambda: tier2_elliptical_multipole(am_over_thetaE=0.03, m=4),
    "multipole_m4_a1": lambda: tier2_multipole_confounder(am_over_thetaE=0.01, m=4),
    "multipole_m4_a3": lambda: tier2_multipole_confounder(am_over_thetaE=0.03, m=4),
    "multipole_m3_a3": lambda: tier2_multipole_confounder(am_over_thetaE=0.03, m=3),
    "lenslight_fixed60": lambda: tier2_lens_light("fixed60"),
    "lenslight_no_subhalo": tier2_lens_light_no_subhalo,
    "lenslight_multipole_m4_a3": lambda: tier2_lens_light_multipole(am_over_thetaE=0.03, m=4),
    "euclid_no_subhalo": lambda: tier0_euclid(no_subhalo=True, source_mag_offset=2.5),
    "euclid_fixed60": lambda: tier0_euclid("fixed60", source_mag_offset=2.5),
    "euclid_fixed15": lambda: tier0_euclid("fixed15", source_mag_offset=2.5),
    "euclid_multipole_m4_a3": lambda: tier0_euclid(no_subhalo=True, am_over_thetaE=0.03, m=4, source_mag_offset=2.5),
    "pop_no_subhalo": lambda: tier0_population(no_subhalo=True),
    "pop_fixed60": lambda: tier0_population("fixed60"),
    "pop_fixed15": lambda: tier0_population("fixed15"),
    "cosmos_fixed60": lambda: tier1_cosmos("fixed60"),
    "cosmos_fixed15": lambda: tier1_cosmos("fixed15"),
    "cosmos_cdm": lambda: tier1_cosmos("cdm"),
    "cosmos_no_subhalo": tier1_cosmos_no_subhalo,
    "cosmos_multipole_m4_a1": lambda: tier1_cosmos_multipole_confounder(am_over_thetaE=0.01, m=4),
    "cosmos_multipole_m4_a3": lambda: tier1_cosmos_multipole_confounder(am_over_thetaE=0.03, m=4),
    # real sources at 600 s, for the published pipeline's pixelized source (same seeds as the Tier-1 populations)
    "cosmos600_no_subhalo": lambda: _depth(tier1_cosmos_no_subhalo(), 600.0, "cosmos600"),
    "cosmos600_fixed60": lambda: _depth(tier1_cosmos("fixed60"), 600.0, "cosmos600"),
    # the measured multipole amplitude (~1% of the convergence) at Euclid-like depth, for the survey-purity estimate
    "euclid_multipole_m4_a1": lambda: tier0_euclid(no_subhalo=True, am_over_thetaE=0.01, m=4, source_mag_offset=2.5),
}


def _misspec(base: str, kind: str, amp: float):
    """Misspecification-tolerance grid (referee round 29, M6): a clean population at one of three operating points
    (Tier 0, 600 s, Euclid-like) with either the PSF wider than the fitters assume by `amp` (fractional, rendered at the
    fitters' own sampling) or two source clumps of `amp` of the source flux each."""
    cfg = tier0_euclid(no_subhalo=True, source_mag_offset=2.5) if base == "euclid" else tier0_tsang("fixed60")
    if base != "euclid":
        cfg.subhalo.enabled = False
        cfg.subhalo.presence_prob = 0.0
    if base == "600":
        cfg.instrument.exposure_time = 600.0
    if kind == "psf":
        cfg.instrument.render_supersampling = 1
        cfg.instrument.render_psf_scale = 1.0 + amp
    else:
        cfg.source.n_clumps = 2
        cfg.source.clump_flux_frac_range = (amp, amp)
    cfg.tier = f"misspec_{base}_{kind}{amp:g}_no_subhalo"
    return cfg


for _b in ("t0", "600", "euclid"):
    for _a in (0.02, 0.05, 0.10):
        CONFIGS[f"misspec_{_b}_psf{round(100*_a)}_no_subhalo"] = (lambda b=_b, a=_a: _misspec(b, "psf", a))
    for _a in (0.005, 0.01, 0.02, 0.04):
        CONFIGS[f"misspec_{_b}_clump{('%g' % (100*_a)).replace('.', 'p')}_no_subhalo"] = (lambda b=_b, a=_a: _misspec(b, "clump", a))


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
        control_key = ("no_los" if truth.get("los_halo") else
                       ("no_subhalo" if truth.get("subhalo") else ("no_multipole" if truth.get("multipole") else "smooth")))
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
