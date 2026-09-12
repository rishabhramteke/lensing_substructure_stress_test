"""Joint-refit mass test for Family A (added 2026-09-11).

`oracle_position_mass_test.py` showed that the scan's mass underestimate survives the
oracle position, the matched concentration AND a fine mass grid (121/121 low, median
-1.0 dex). The one idealization left is the fit-then-scan shortcut itself: the smooth
macro-model + source are fitted once, WITH the subhalo present in the data, and then held
fixed while a perturber is added. If the smooth model has absorbed part of the subhalo's
deflection (source position, shear, ellipticity all can), adding the full true perturber
on top double-counts it, and the best additional mass is smaller than the truth.

This script tests that directly. For every subhalo-bearing lens, subhalo at the TRUE
position and TRUE c=60:
  (i)  macro+source held fixed at the smooth fit, only log10 M free (continuous)
  (ii) macro+source+log10 M all re-fitted jointly, initialised from the smooth fit
Published pipelines (Nightingale+2024, Despali+2022) do (ii) at every grid cell; our
scan -- and any fast fit-then-scan -- does (i).

    python scripts/joint_refit_mass_test.py --n 300 --seed 0 --out results/joint_refit_mass
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lenstronomy.Cosmo.lens_cosmo import LensCosmo  # noqa: E402
from baseline_a.fit import fit_smooth, _sim, _unpack, BOUNDS, SCALE  # noqa: E402

CHI2_DOF_UNRELIABLE = 10.0
LC = LensCosmo(z_lens=0.5, z_source=1.0)


def sub_kwargs(log10_m, c, x, y, tau=20.0):
    Rs, alpha_Rs = LC.nfw_physical2angle(M=10 ** log10_m, c=c)
    return {"Rs": Rs, "alpha_Rs": alpha_Rs, "r_trunc": tau * Rs, "center_x": x, "center_y": y}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--population", default="test_fixed60")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--concentration", type=float, default=60.0)
    p.add_argument("--out", type=Path, default=ROOT / "results/joint_refit_mass")
    args = p.parse_args()

    pop_dir = ROOT / "data" / args.population
    manifest = json.loads((pop_dir / "manifest.json").read_text())
    kwargs_band, num_pix = manifest["kwargs_band"], manifest["image_shape"][0]
    truths = [json.loads(l) for l in open(pop_dir / "truth.jsonl")]
    images = np.load(pop_dir / "images_full_noisy.npy")
    rng_sel = np.random.default_rng(args.seed)
    idx = np.sort(rng_sel.choice(len(truths), size=min(args.n, len(truths)), replace=False))
    idx = [i for i in idx if truths[i].get("subhalo") is not None]
    rng_fit = np.random.default_rng(args.seed + 1000)

    sim = _sim(kwargs_band, num_pix, ["EPL", "SHEAR", "TNFW"])
    image_model = sim.image_model_class(kwargs_numerics={"supersampling_factor": 1})
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); recs = []
    for k, i in enumerate(idx):
        truth = truths[i]; sub = truth["subhalo"]; data = images[i]
        vec, chi2_smooth, _, noise_std = fit_smooth(data, truth, kwargs_band, num_pix, rng=rng_fit, maxiter=60)
        reliable = (chi2_smooth / num_pix ** 2) < CHI2_DOF_UNRELIABLE

        def resid_full(v):
            kl, ks = _unpack(v[:13])
            model = image_model.image(kwargs_lens=kl + [sub_kwargs(v[13], args.concentration, sub["x"], sub["y"])], kwargs_source=ks)
            return ((data - model) / noise_std).ravel()

        # (i) macro fixed, mass free
        r1 = least_squares(lambda m: resid_full(np.concatenate([vec, m])), [9.5], bounds=([7.5], [11.5]), x_scale=[1.0], method="trf", max_nfev=200)
        # (ii) joint refit
        lo = [b[0] for b in BOUNDS] + [7.5]; hi = [b[1] for b in BOUNDS] + [11.5]
        x0 = np.concatenate([vec, [9.5]])
        r2 = least_squares(resid_full, x0, bounds=(lo, hi), x_scale=SCALE + [1.0], method="trf", max_nfev=60 * 14)
        recs.append({"index": int(i), "reliable_fit": reliable, "log10_M200_true": sub["log10_M200"], "chi2_smooth": chi2_smooth,
                     "fixed_macro": {"log10_m": float(r1.x[0]), "chi2": float(np.sum(r1.fun ** 2))},
                     "joint": {"log10_m": float(r2.x[13]), "chi2": float(np.sum(r2.fun ** 2)), "nfev": int(r2.nfev),
                               "macro_shift": {n: float(r2.x[j] - vec[j]) for j, n in enumerate(["theta_E", "gamma", "e1", "e2", "g1", "g2", "src_x", "src_y", "R", "n", "e1s", "e2s", "amp"])}}})
        if (k + 1) % 25 == 0:
            print(f"  {k+1}/{len(idx)}  ({time.time()-t0:.0f}s)", flush=True)

    with open(args.out / "records.jsonl", "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")

    rel = [r for r in recs if r["reliable_fit"]]
    tm = np.array([r["log10_M200_true"] for r in rel])
    sig = np.array([(r["chi2_smooth"] - r["joint"]["chi2"]) > 20 for r in rel])   # lenses where the joint fit finds >20 improvement

    def stats(est, mask):
        e = est[mask] - tm[mask]
        return {"n": int(mask.sum()), "median_error_dex": float(np.median(e)) if mask.any() else None,
                "mean_abs_error_dex": float(np.mean(np.abs(e))) if mask.any() else None,
                "n_underestimated": int((e < 0).sum()), "n_within_0p5dex": int((np.abs(e) < 0.5).sum()),
                "n_within_0p25dex": int((np.abs(e) < 0.25).sum())}
    m1 = np.array([r["fixed_macro"]["log10_m"] for r in rel]); m2 = np.array([r["joint"]["log10_m"] for r in rel])
    summary = {"population": args.population, "n_subhalo_lenses": len(recs), "n_reliable": len(rel), "concentration": args.concentration,
               "n_with_joint_dchi2_gt_20": int(sig.sum()),
               "fixed_macro_mass_free": {"signal": stats(m1, sig), "all_reliable": stats(m1, np.ones(len(rel), bool))},
               "joint_refit": {"signal": stats(m2, sig), "all_reliable": stats(m2, np.ones(len(rel), bool))},
               "median_joint_dchi2_minus_fixed_dchi2": float(np.median([(r["fixed_macro"]["chi2"] - r["joint"]["chi2"]) for r in rel])),
               "median_abs_macro_shift_signal": {n: float(np.median([abs(r["joint"]["macro_shift"][n]) for r, s in zip(rel, sig) if s])) for n in rel[0]["joint"]["macro_shift"]},
               "note": "Subhalo at TRUE position and TRUE c for every lens. (i) reproduces the fit-then-scan shortcut with a continuous mass; (ii) is what published pipelines do at each grid cell."}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
