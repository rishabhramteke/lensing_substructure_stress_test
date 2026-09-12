"""Run the Family-A fast parametric scan over a population.

    source ~/myenv/bin/activate
    python scripts/run_baseline_a.py --population test_fixed60 --n 300 --out results/baseline_a/test_fixed60 --seed 0
    python scripts/run_baseline_a.py --population test_fixed15 --n 300 --out results/baseline_a/test_fixed15 --seed 0   # paired with the above
    python scripts/run_baseline_a.py --population no_subhalo --n 300 --out results/baseline_a/no_subhalo --seed 1
    python scripts/run_baseline_a.py --population multipole_m4_a1 --n 300 --out results/baseline_a/multipole_m4_a1 --seed 2
    python scripts/run_baseline_a.py --population multipole_m4_a3 --n 300 --out results/baseline_a/multipole_m4_a3 --seed 3

Writes one JSON line per image (delta_chi2, chi2_smooth, best-fit subhalo
position/mass, wall time, whether the fit is flagged unreliable) plus a
manifest. See src/baseline_a/__init__.py for the two stated idealizations
(near-truth initialization; fixed-concentration grid scan).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from baseline_a.fit import fit_smooth, scan_subhalo, lens_light_image, set_multipole_orders

# A smooth-only fit at chi2/dof above this is flagged unreliable (optimizer
# didn't converge) rather than silently trusted -- see first_results.md.
# Tier-0 default: 10.0, calibrated for a correctly-specified Sersic source.
# On Tier-1 (COSMOS source) this default is wrong for a different reason: the
# Sersic proxy is a genuinely misspecified model there (see
# src/baseline_a/__init__.py), so chi2/dof is large even for a fully-converged
# fit (verified directly: scipy status=2/success=True at chi2/dof~20-150 on
# Tier-1 images, not an optimizer failure) -- pass a much larger
# --chi2-dof-unreliable for Tier-1 runs so this flag keeps catching genuine
# non-convergence without excluding the entire population by construction.
CHI2_DOF_UNRELIABLE_DEFAULT = 10.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--population", required=True, help="name of a folder under data/")
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0, help="which images to sample (indices), not the fit itself")
    p.add_argument("--lens-light-components", type=int, default=1, choices=[1, 2], help="Sersic components fitted for lens light when the population has it (1 = usual single Sersic, 2 = correctly specified)")
    p.add_argument("--macro-multipole", action="store_true", help="include multipole terms (a_m, phi_m free per order) in the smooth macro-model, as post-Lange+2024 pipelines do")
    p.add_argument("--macro-multipole-orders", default="4", help="comma-separated orders carried by --macro-multipole (default 4; '3,4' tests the cost of the extra freedom)")
    p.add_argument("--blind", action="store_true", help="initialize the smooth fit from the data alone (no truth), see fit._blind_init_vec")
    p.add_argument("--maxiter", type=int, default=60, help="least-squares max_nfev multiplier (60 = the value used throughout)")
    p.add_argument("--concentration", type=str, default="15",
                   help="concentration assumed by the scan: a number (fixed c), or 'cm' for the Dutton & Maccio 2014 "
                        "concentration-mass relation evaluated at each trial mass (zero scatter)")
    p.add_argument("--chi2-dof-unreliable", type=float, default=CHI2_DOF_UNRELIABLE_DEFAULT,
                   help="flag a smooth fit unreliable above this chi2/dof (default: Tier-0 value; pass a much larger number for Tier-1, see module comment)")
    args = p.parse_args()
    CHI2_DOF_UNRELIABLE = args.chi2_dof_unreliable
    mp_orders = tuple(int(m) for m in args.macro_multipole_orders.split(",")) if args.macro_multipole else False
    if mp_orders:
        set_multipole_orders(mp_orders)
    if args.concentration.lower() in ("cm", "dm14", "dutton"):
        from lensing.concentration import concentration_dutton_maccio14
        conc = lambda logm: float(concentration_dutton_maccio14(np.array(logm), rng=None, z=0.5, scatter_dex=0.0))
        conc_label = "dutton_maccio14_zero_scatter"
    else:
        conc = float(args.concentration)
        conc_label = conc

    pop_dir = args.data_root / args.population
    manifest = json.loads((pop_dir / "manifest.json").read_text())
    kwargs_band = manifest["kwargs_band"]
    num_pix = manifest["image_shape"][0]
    truths = [json.loads(l) for l in open(pop_dir / "truth.jsonl")]
    noisy_path = pop_dir / "images_full_noisy.npy"
    images = np.load(noisy_path) if noisy_path.exists() else np.load(pop_dir / "images_full_noiseless.npy")

    rng_sel = np.random.default_rng(args.seed)
    n = min(args.n, len(truths))
    idx = np.sort(rng_sel.choice(len(truths), size=n, replace=False)) if n < len(truths) else np.arange(len(truths))

    args.out.mkdir(parents=True, exist_ok=True)
    rng_fit = np.random.default_rng(args.seed + 1000)
    t0 = time.time()
    n_unreliable = 0
    has_ll = any(tr.get("lens_light") for tr in truths)
    subtracted = np.zeros((len(idx), num_pix, num_pix), dtype=np.float32) if has_ll else None   # data minus the FITTED single-Sersic lens light
    with open(args.out / "scan_results.jsonl", "w") as f:
        for k, i in enumerate(idx):
            truth = truths[i]
            theta_E = truth["lens_macro"]["theta_E"]
            t1 = time.time()
            vec, chi2_smooth, image_model, noise_std = fit_smooth(images[i], truth, kwargs_band, num_pix, rng=rng_fit, maxiter=args.maxiter, blind=args.blind, lens_light_components=args.lens_light_components, macro_multipole=mp_orders)
            t_fit = time.time() - t1
            if has_ll:
                subtracted[k] = images[i] - lens_light_image(vec, kwargs_band, num_pix)
            reliable = (chi2_smooth / (num_pix * num_pix)) < CHI2_DOF_UNRELIABLE
            if not reliable:
                n_unreliable += 1
            t2 = time.time()
            best = scan_subhalo(images[i], noise_std, vec, kwargs_band, num_pix, theta_E=theta_E,
                                 chi2_smooth=chi2_smooth, concentration=conc)
            t_scan = time.time() - t2
            rec = {
                "index": int(i), "has_subhalo": truth.get("subhalo") is not None,
                "log10_M200_true": (truth["subhalo"] or {}).get("log10_M200"),
                "concentration_true": (truth["subhalo"] or {}).get("concentration"),
                "has_multipole": truth.get("multipole") is not None,
                "chi2_smooth_per_dof": chi2_smooth / (num_pix * num_pix),
                "reliable_fit": reliable,
                "delta_chi2": best["delta_chi2"], "best_x": best["x"], "best_y": best["y"], "best_log10_m": best["log10_m"],
                "t_fit_s": t_fit, "t_scan_s": t_scan,
            }
            f.write(json.dumps(rec) + "\n")
            if (k + 1) % 25 == 0 or k + 1 == n:
                elapsed = time.time() - t0
                print(f"  {k+1}/{n}  ({elapsed/(k+1):.2f}s/image avg, {elapsed:.0f}s elapsed)")

    manifest_out = {
        "population": args.population, "n": n, "seed": args.seed, "concentration_assumed": conc_label,
        "blind_initialization": args.blind, "lens_light_components": args.lens_light_components, "macro_multipole": args.macro_multipole, "macro_multipole_orders": list(mp_orders) if mp_orders else None, "maxiter": args.maxiter, "chi2_dof_unreliable_threshold": CHI2_DOF_UNRELIABLE,
        "n_unreliable_fits": n_unreliable, "elapsed_s": time.time() - t0,
        "source_manifest": manifest,
    }
    if has_ll:
        np.save(args.out / "lens_light_subtracted.npy", subtracted)
        np.save(args.out / "subsample_indices.npy", np.asarray(idx))
    (args.out / "manifest.json").write_text(json.dumps(manifest_out, indent=2, default=str))
    print(f"\nwrote {n} scans to {args.out}/  ({n_unreliable} unreliable fits, {time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
