"""Family B population runs -- pixelized linear delta-psi residual detector ("gravitational
imaging", Koopmans 2005; Vegetti & Koopmans 2009) with PyAutoLens's own
`autolens.potential_correction` (linear dpsi inversion), on the SAME 300-image
subsamples Family A was run on, writing Family A's file format so
`evaluate_baseline_a.py`, `evaluate_baseline_a_fixed_threshold.py` and
`plot_localization_accuracy.py --a-root results/baseline_b/<variant>` run verbatim.

    source ~/myenv/bin/activate
    python scripts/run_family_b.py --population test_fixed60 --variant fitted --seed 0 --n 300 \
        --out results/baseline_b/fitted/test_fixed60

API pattern, idealizations and the failure regime are documented in
scripts/family_b_prototype.py and results/family_b/NOTES.md; this script does
not re-derive them.

Two macro-model variants:
  fitted (primary)  the smooth EPL+shear+Sersic model from Family A's
                    `fit_smooth` (near-truth init, least squares, maxiter=60),
                    rendered noiselessly by lenstronomy on the same grid; the
                    source gradient for the inversion is built from the FITTED
                    Sersic parameters. Same idealization level as Family A, so
                    the two are directly comparable.
  oracle            lenstronomy's own noiseless no-subhalo render of the truth
                    (`images_control_noiseless.npy`: no-subhalo twin for the
                    subhalo populations, no-multipole twin for the confounder
                    populations, the image itself for `no_subhalo`) as the macro
                    model, truth source gradient. The inversion's ceiling; an
                    idealization one step beyond Family A's.

Statistic: score = max|dkappa| over the dpsi mesh at a FIXED CurvatureMask
coefficient lam=1e5 (also stored at 1e4 and 1e6 for a sensitivity band; fixed by
rule, not per-lens evidence -- see NOTES.md). Written as `peak_abs_dkappa` AND as
`delta_chi2` (documented alias so Family A's evaluators work; the 10%-FPR
calibration on `no_subhalo`'s 90th percentile is statistic-agnostic).
Position = dkappa peak on the mesh (world frame shared with the truth).
Mass proxy = Sigma_crit * sum(dkappa) * dpix^2 in a 2-mesh-pixel aperture at
the peak; `best_log10_m` = log10 of that, floored at 6.0 when the aperture mass
is <= 0 (the raw signed value is kept in `aperture_mass_proxy_Msun`; the floor
keeps the mass-error evaluators numeric -- documented deviation from "null").
`reliable_fit`: fitted variant uses Family A's rule (chi2_smooth/N_pix < 10);
oracle is always reliable unless the inversion raised.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import logging
logging.disable(logging.WARNING)

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import autolens as al  # noqa: E402
import autoarray as aa  # noqa: E402
from autolens.potential_correction import FitDpsiImaging, DpsiPixelization, RegularDpsiMesh, AnalyticSrcFactory  # noqa: E402
from autolens.potential_correction import util as pc_util  # noqa: E402
from lenstronomy.SimulationAPI.sim_api import SimAPI  # noqa: E402
from lenstronomy.Cosmo.lens_cosmo import LensCosmo  # noqa: E402
from lenstronomy.Util import param_util  # noqa: E402
from baseline_a.fit import fit_smooth, _unpack, _unpack3, lens_light_image  # noqa: E402

CHI2_DOF_UNRELIABLE = 10.0
MASS_FLOOR_LOG10 = 6.0


# ------------------------------------------------------------------ helpers (prototype pattern)
def to_pal(arr):
    return np.asarray(arr)[::-1, :].copy()


def noise_map_for(image, kwargs_band, num_pix):
    sim = SimAPI(num_pix=num_pix, kwargs_single_band=kwargs_band,
                 kwargs_model={"lens_model_list": ["EPL", "SHEAR"], "source_light_model_list": ["SERSIC_ELLIPSE"]})
    return np.asarray(sim.estimate_noise(image))


def build_imaging(data_l, noise_l, kwargs_band):
    ps = kwargs_band["pixel_scale"]
    psf = al.Convolver.from_gaussian(shape_native=(11, 11), pixel_scales=ps, sigma=kwargs_band["seeing"] / 2.3548, normalize=True)
    return al.Imaging(data=al.Array2D.no_mask(values=to_pal(data_l), pixel_scales=ps),
                      noise_map=al.Array2D.no_mask(values=to_pal(noise_l), pixel_scales=ps), psf=psf)


def arc_mask(data_l, noise_l, ps, snr_threshold=3.0, ext_size=5):
    m = pc_util.arc_mask_from(to_pal(data_l) / to_pal(noise_l), threshold=snr_threshold, ignore_size=25, ext_size=ext_size)
    return al.Mask2D(mask=np.asarray(m, dtype=bool), pixel_scales=ps)


def galaxies_from(params, flux_scale=1.0):
    """PyAutoLens lens+source from a lens_macro/source/cosmology dict (truth or fitted)."""
    lm, s, cos = params["lens_macro"], params["source"], params["cosmology"]
    lens = al.Galaxy(redshift=cos["z_lens"],
                     mass=al.mp.PowerLaw(centre=(0.0, 0.0), ell_comps=al.convert.ell_comps_from(axis_ratio=lm["q"], angle=lm["phi_deg"]),
                                         einstein_radius=lm["theta_E"], slope=lm["gamma"]),
                     shear=al.mp.ExternalShear(gamma_1=lm["gamma1"], gamma_2=lm["gamma2"]))
    source = al.Galaxy(redshift=cos["z_source"],
                       bulge=al.lp.Sersic(centre=(s["y"], s["x"]), ell_comps=al.convert.ell_comps_from(axis_ratio=s["q"], angle=s["phi_deg"]),
                                          intensity=s["amp"] * flux_scale, effective_radius=s["R_sersic"], sersic_index=s["n_sersic"]))
    return lens, source


def fit_flux_scale(mi, params, target_slim):
    lens, source = galaxies_from(params, 1.0)
    model = np.asarray(al.FitImaging(dataset=mi, tracer=al.Tracer(galaxies=[lens, source])).model_data.slim)
    w = 1.0 / np.asarray(mi.noise_map.slim) ** 2
    return float(np.sum(w * target_slim * model) / max(np.sum(w * model * model), 1e-30))


def params_from_fitted_vec(vec, cosmology):
    """Family A's 13-vector -> the dict layout galaxies_from() reads."""
    kwargs_lens, kwargs_source = _unpack(vec)
    kl, ks = kwargs_lens[0], kwargs_source[0]
    phi_l, q_l = param_util.ellipticity2phi_q(kl["e1"], kl["e2"])
    phi_s, q_s = param_util.ellipticity2phi_q(ks["e1"], ks["e2"])
    return {"lens_macro": {"theta_E": kl["theta_E"], "gamma": kl["gamma"], "q": float(q_l), "phi_deg": float(np.degrees(phi_l)),
                           "gamma1": kwargs_lens[1]["gamma1"], "gamma2": kwargs_lens[1]["gamma2"]},
            "source": {"x": ks["center_x"], "y": ks["center_y"], "q": float(q_s), "phi_deg": float(np.degrees(phi_s)),
                       "amp": ks["amp"], "R_sersic": ks["R_sersic"], "n_sersic": ks["n_sersic"]},
            "cosmology": cosmology}, kwargs_lens, kwargs_source


def dpsi_inversion(mi, residual, params, flux_scale, reg_coeff, factor, preloads=None):
    lens, source = galaxies_from(params, flux_scale)
    grid = np.asarray(mi.grid.slim)
    src_pos = grid - np.asarray(lens.deflections_yx_2d_from(grid=mi.grid.slim))
    src_grad = AnalyticSrcFactory(source).eval_grad(src_pos[:, 1], src_pos[:, 0])
    pix = DpsiPixelization(mesh=RegularDpsiMesh(factor=factor), regularization=aa.reg.CurvatureMask(coefficient=reg_coeff))
    fit = FitDpsiImaging(masked_imaging=mi, image_residual=residual, source_gradient=src_grad, dpsi_pixelization=pix, preloads=preloads)
    dpsi = np.asarray(fit.solve_dpsi())
    pair = fit.pair_dpsi_data_obj
    return {"dkappa": 0.5 * np.asarray(pair.hamiltonian_dpsi @ dpsi), "pair": pair, "log_evidence": float(fit.log_evidence),
            "preloads": {"psf_mat": fit.psf_mat}}


def peak_stats(res, lc):
    pair, dk = res["pair"], res["dkappa"]
    i = int(np.argmax(np.abs(dk)))
    px, py = float(pair.xgrid_dpsi_1d[i]), float(pair.ygrid_dpsi_1d[i])
    r = np.hypot(pair.xgrid_dpsi_1d - px, pair.ygrid_dpsi_1d - py)
    m_ap = float(lc.sigma_crit_angle * np.sum(dk[r <= 2.0 * pair.dpix_dpsi]) * pair.dpix_dpsi ** 2)
    mad = 1.4826 * np.median(np.abs(dk - np.median(dk))) + 1e-30
    return {"peak_abs_dkappa": float(np.abs(dk[i])), "peak_signed_dkappa": float(dk[i]), "peak_significance_mad": float(np.abs(dk[i]) / mad),
            "peak_x": px, "peak_y": py, "aperture_mass_proxy_Msun": m_ap,
            "best_log10_m": float(np.log10(m_ap)) if m_ap > 10 ** MASS_FLOOR_LOG10 else MASS_FLOOR_LOG10, "n_dpsi": int(dk.size)}


# ------------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--population", required=True)
    p.add_argument("--data-root", type=Path, default=ROOT / "data")
    p.add_argument("--variant", choices=["fitted", "oracle"], default="fitted")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--seed", type=int, default=0, help="subsample selection seed (same rule as run_baseline_a.py)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--lams", type=float, nargs="+", default=[1e4, 1e5, 1e6])
    p.add_argument("--lens-light-components", type=int, default=1, choices=[1, 2], help="Sersic components fitted for lens light when the population has it")
    p.add_argument("--lam-primary", type=float, default=1e5)
    p.add_argument("--factor", type=int, default=2)
    args = p.parse_args()

    pop_dir = args.data_root / args.population
    manifest = json.loads((pop_dir / "manifest.json").read_text())
    kwargs_band, num_pix, ps = manifest["kwargs_band"], manifest["image_shape"][0], manifest["kwargs_band"]["pixel_scale"]
    truths = [json.loads(l) for l in open(pop_dir / "truth.jsonl")]
    noisy_path = pop_dir / "images_full_noisy.npy"
    images = np.load(noisy_path) if noisy_path.exists() else np.load(pop_dir / "images_full_noiseless.npy")
    control = np.load(pop_dir / "images_control_noiseless.npy")

    rng_sel = np.random.default_rng(args.seed)
    n = min(args.n, len(truths))
    idx = np.sort(rng_sel.choice(len(truths), size=n, replace=False)) if n < len(truths) else np.arange(len(truths))
    rng_fit = np.random.default_rng(args.seed + 1000)

    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_unreliable = n_err = 0
    with open(args.out / "scan_results.jsonl", "w") as f:
        for k, i in enumerate(idx):
            truth = truths[i]
            lc = LensCosmo(z_lens=truth["cosmology"]["z_lens"], z_source=truth["cosmology"]["z_source"])
            data = images[i]
            rec = {"index": int(i), "has_subhalo": truth.get("subhalo") is not None,
                   "log10_M200_true": (truth["subhalo"] or {}).get("log10_M200"),
                   "concentration_true": (truth["subhalo"] or {}).get("concentration"),
                   "has_multipole": truth.get("multipole") is not None, "variant": args.variant}
            try:
                t1 = time.time()
                noise = noise_map_for(data, kwargs_band, num_pix)
                if args.variant == "fitted":
                    vec, chi2_smooth, image_model, _ = fit_smooth(data, truth, kwargs_band, num_pix, rng=rng_fit, maxiter=60, lens_light_components=args.lens_light_components)
                    params, kwargs_lens, kwargs_source = params_from_fitted_vec(vec, truth["cosmology"])
                    _, _, kwargs_ll = _unpack3(vec)   # fitted lens light (None without lens light), added 2026-09-12
                    macro = np.asarray(image_model.image(kwargs_lens=kwargs_lens, kwargs_source=kwargs_source, kwargs_lens_light=kwargs_ll))
                    ll_img = lens_light_image(vec, kwargs_band, num_pix) if kwargs_ll else 0.0
                    chi2_per_dof = chi2_smooth / (num_pix * num_pix)
                    reliable = chi2_per_dof < CHI2_DOF_UNRELIABLE
                else:
                    params, macro = truth, control[i]
                    chi2_per_dof = float(np.mean(((data - macro) / noise) ** 2))
                    reliable = True
                    ll_img = 0.0   # the oracle macro render already contains the true lens light
                t_fit = time.time() - t1

                t2 = time.time()
                mask = arc_mask(data - ll_img, noise, ps)   # arc mask on the lens-light-subtracted image
                mask2d = np.asarray(mask)
                mi = build_imaging(data, noise, kwargs_band).apply_mask(mask=mask)
                macro_slim = to_pal(macro)[~mask2d]
                flux_scale = fit_flux_scale(mi, params, macro_slim)
                residual = np.asarray(mi.data.slim) - macro_slim
                preloads, per_lam = None, {}
                for lam in args.lams:
                    res = dpsi_inversion(mi, residual, params, flux_scale, lam, args.factor, preloads)
                    preloads = res["preloads"]
                    per_lam[lam] = (res, peak_stats(res, lc))
                t_inv = time.time() - t2

                res_p, st_p = per_lam[args.lam_primary]
                rec.update({
                    "chi2_smooth_per_dof": float(chi2_per_dof), "reliable_fit": bool(reliable), "flux_scale": flux_scale,
                    "n_unmasked": int((~mask2d).sum()), "n_dpsi": st_p["n_dpsi"],
                    "peak_abs_dkappa": st_p["peak_abs_dkappa"], "delta_chi2": st_p["peak_abs_dkappa"],  # alias for Family A evaluators
                    "peak_signed_dkappa": st_p["peak_signed_dkappa"], "peak_significance_mad": st_p["peak_significance_mad"],
                    "log_evidence_primary": res_p["log_evidence"],
                    "best_x": st_p["peak_x"], "best_y": st_p["peak_y"], "best_log10_m": st_p["best_log10_m"],
                    "aperture_mass_proxy_Msun": st_p["aperture_mass_proxy_Msun"],
                    "t_fit_s": t_fit, "t_inv_s": t_inv, "t_scan_s": t_inv,  # t_scan_s alias for evaluate_baseline_a.py's timing
                })
                for lam, (_, st) in per_lam.items():
                    rec[f"peak_abs_dkappa_lam{lam:.0e}".replace("+0", "").replace("e0", "e")] = st["peak_abs_dkappa"]
                if not reliable:
                    n_unreliable += 1
            except Exception as e:  # noqa: BLE001 -- one bad lens must not kill a 300-lens run
                n_err += 1
                rec.update({"error": f"{type(e).__name__}: {e}", "chi2_smooth_per_dof": float("nan"), "reliable_fit": False,
                            "delta_chi2": 0.0, "peak_abs_dkappa": 0.0, "best_x": None, "best_y": None, "best_log10_m": MASS_FLOOR_LOG10,
                            "t_fit_s": 0.0, "t_inv_s": 0.0, "t_scan_s": 0.0})
            f.write(json.dumps(rec, default=float) + "\n")
            f.flush()
            if (k + 1) % 25 == 0 or k + 1 == n:
                el = time.time() - t0
                print(f"  {args.population}/{args.variant}  {k+1}/{n}  ({el/(k+1):.2f}s/lens avg, {el:.0f}s elapsed, {n_unreliable} unreliable, {n_err} errors)", flush=True)

    manifest_out = {"family": "B (linear delta-psi residual detector, PyAutoLens autolens.potential_correction, linear dpsi inversion)",
                    "population": args.population, "variant": args.variant, "n": n, "seed": args.seed,
                    "lams": args.lams, "lam_primary": args.lam_primary, "mesh_factor": args.factor,
                    "statistic": "max|dkappa| at lam_primary, written also as delta_chi2 (alias)",
                    "mass_proxy": "Sigma_crit*sum(dkappa)*dpix^2 in 2-mesh-pixel aperture at peak; best_log10_m floored at 6.0 when <= 0",
                    "chi2_dof_unreliable_threshold": CHI2_DOF_UNRELIABLE, "n_unreliable_fits": n_unreliable, "n_errors": n_err,
                    "elapsed_s": time.time() - t0, "source_manifest": manifest}
    (args.out / "manifest.json").write_text(json.dumps(manifest_out, indent=2, default=str))
    print(f"\nwrote {n} records to {args.out}/  ({n_unreliable} unreliable, {n_err} errors, {time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
