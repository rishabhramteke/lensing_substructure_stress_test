"""Family B prototype -- pixelized linear delta-psi residual detector ("gravitational imaging")
on ONE of our Tier-0 lenses at a time, using PyAutoLens's own implementation of
the method (`autolens.potential_correction`, the port of Cao et al. 2025's
package; the technique is Koopmans 2005 / Vegetti & Koopmans 2009). This is the
published family's real code, not a re-implementation.

    source ~/myenv/bin/activate
    python scripts/family_b_prototype.py --idx 104          # the lens quoted in the analysis (log10 M=10.67)
    python scripts/family_b_prototype.py --idx 332          # moderate mass (log10 M=10.28), linear regime

Scope: single-lens feasibility. Establishes (a) the API pattern, (b) whether a
localized convergence correction dkappa appears at the true subhalo and NOT on
the subhalo-free twin rendered with the identical noise draw, (c) a candidate
detection statistic compatible with our common 10%-FPR metric, (d) the
regularization-strength dependence (Galan+2022's own caveat), (e) wall time
per lens. No population runs here.

What is run, exactly:
  1. Render the lens (from data/test_fixed60 truth) twice with lenstronomy via
     src/lensing/simulate.LensRenderer -- with and without the subhalo, sharing one
     noise realization -- and take the per-pixel noise map from SimAPI, as the
     rest of this project does. Arrays are row-flipped once into PyAutoLens's
     native orientation (the convention difference established in
     validate_pyautolens.py).
  2. Smooth macro model. Default `--smooth oracle`: lenstronomy's OWN noiseless
     no-subhalo image is the macro model, so the residual fed to the inversion is
     exactly (subhalo signature + noise) on the subhalo image and exactly (noise)
     on the twin. This is the "perfect macro model" idealization -- one step
     beyond Family A's truth-initialized fit -- chosen so the prototype measures
     the inversion and not the cross-code convention mismatch: with
     `--smooth pyautolens` (PowerLaw+ExternalShear+Sersic re-built in PyAutoLens
     at the true parameters, one fitted flux scale), the twin's residual already
     has chi2/dof ~ 6.6 from PSF/elliptical-radius/theta_E convention differences,
     which the dpsi inversion then happily absorbs. Both chi2 are recorded.
  3. Residual on an arc mask (the module's own `arc_mask_from`, SNR>3, dilated),
     source gradients from the analytic Sersic source (PyAutoLens units x flux
     scale) at the ray-traced positions, then `FitDpsiImaging`: a single
     regularized LINEAR inversion of the residual for dpsi on a mesh `factor` x
     coarser than the data (Koopmans 2005's first iteration; no nonlinear search,
     no nested sampling). dkappa = (1/2) * Laplacian(dpsi) via the mesh operator.
     The true subhalo's tNFW convergence is evaluated on the same mesh for
     comparison.
  4. Statistic candidates on the dkappa map: max |dkappa| (the score we would
     calibrate to 10% FPR on the no-subhalo population, like every other family);
     the same in units of the map's robust (MAD) scatter; position of the peak vs
     the true subhalo; dkappa integrated in a 2-mesh-pixel aperture at the peak,
     times Sigma_crit, as a mass proxy -- reported next to the true tNFW mass in
     the same aperture.
  5. Sweep over the CurvatureMask coefficient, reporting the module's own Bayesian
     log-evidence so the max-evidence value is picked the way the method
     prescribes, plus how the statistics move with it.

Idealizations stated: perfect (oracle) macro model by default; single lens plane;
one fixed dpsi mesh factor; the source is held fixed at the analytic truth during
the dpsi inversion (the joint source+dpsi inversion `FitDpsiSrcImaging` and the
Levenberg-Marquardt `IterFitDpsiSrcImaging` are the full method and are left for
the population runs); first-order linearization in dpsi -- which the results
show breaking down for the most massive Tier-0 subhaloes.
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import autolens as al  # noqa: E402
import autoarray as aa  # noqa: E402
from autolens.potential_correction import FitDpsiImaging, DpsiPixelization, RegularDpsiMesh, AnalyticSrcFactory  # noqa: E402
from autolens.potential_correction import util as pc_util  # noqa: E402
from lenstronomy.SimulationAPI.sim_api import SimAPI  # noqa: E402
from lenstronomy.Cosmo.lens_cosmo import LensCosmo  # noqa: E402
from lenstronomy.LensModel.lens_model import LensModel  # noqa: E402
from lensing.config import tier0_tsang  # noqa: E402
from lensing.simulate import LensRenderer  # noqa: E402

OUT = ROOT / "results" / "family_b"
OUT.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- data
def render_pair(truth, cfg, seed):
    """Noisy image with and without the subhalo (identical noise draw) + noiseless macro image."""
    r = LensRenderer(cfg).render(truth, seed=seed)
    return r["noisy"]["full"], r["noisy"]["no_subhalo"], r["noiseless"]["no_subhalo"], r["noiseless"]["full"]


def noise_map_for(image, kwargs_band, num_pix):
    sim = SimAPI(num_pix=num_pix, kwargs_single_band=kwargs_band,
                 kwargs_model={"lens_model_list": ["EPL", "SHEAR"], "source_light_model_list": ["SERSIC_ELLIPSE"]})
    return np.asarray(sim.estimate_noise(image))


def to_pal(arr):
    """lenstronomy array -> PyAutoLens native orientation (vertical mirror; world coords unchanged)."""
    return np.asarray(arr)[::-1, :].copy()


def build_imaging(data_l, noise_l, kwargs_band):
    ps = kwargs_band["pixel_scale"]
    sigma_psf = kwargs_band["seeing"] / 2.3548  # lenstronomy's Gaussian PSF is specified by its FWHM ("seeing")
    psf = al.Convolver.from_gaussian(shape_native=(11, 11), pixel_scales=ps, sigma=sigma_psf, normalize=True)
    return al.Imaging(data=al.Array2D.no_mask(values=to_pal(data_l), pixel_scales=ps),
                      noise_map=al.Array2D.no_mask(values=to_pal(noise_l), pixel_scales=ps), psf=psf)


def arc_mask(data_l, noise_l, ps, snr_threshold=3.0, ext_size=5):
    """The module's own SNR-based arc mask (True = masked), in PyAutoLens orientation."""
    m = pc_util.arc_mask_from(to_pal(data_l) / to_pal(noise_l), threshold=snr_threshold, ignore_size=25, ext_size=ext_size)
    return al.Mask2D(mask=np.asarray(m, dtype=bool), pixel_scales=ps)


# ------------------------------------------------------------------- smooth model
def smooth_galaxies(truth, flux_scale=1.0):
    lm, s, cos = truth["lens_macro"], truth["source"], truth["cosmology"]
    lens = al.Galaxy(
        redshift=cos["z_lens"],
        mass=al.mp.PowerLaw(centre=(0.0, 0.0), ell_comps=al.convert.ell_comps_from(axis_ratio=lm["q"], angle=lm["phi_deg"]),
                            einstein_radius=lm["theta_E"], slope=lm["gamma"]),
        shear=al.mp.ExternalShear(gamma_1=lm["gamma1"], gamma_2=lm["gamma2"]),
    )
    source = al.Galaxy(
        redshift=cos["z_source"],
        bulge=al.lp.Sersic(centre=(s["y"], s["x"]), ell_comps=al.convert.ell_comps_from(axis_ratio=s["q"], angle=s["phi_deg"]),
                           intensity=s["amp"] * flux_scale, effective_radius=s["R_sersic"], sersic_index=s["n_sersic"]),
    )
    return lens, source


def pal_smooth_model(masked_imaging, truth, flux_scale=1.0):
    lens, source = smooth_galaxies(truth, flux_scale)
    fit = al.FitImaging(dataset=masked_imaging, tracer=al.Tracer(galaxies=[lens, source]))
    return np.asarray(fit.model_data.slim), lens, source


def fit_flux_scale(masked_imaging, truth, target_slim):
    """One linear scalar: the two codes' surface-brightness units differ by a constant."""
    model, _, _ = pal_smooth_model(masked_imaging, truth, 1.0)
    w = 1.0 / np.asarray(masked_imaging.noise_map.slim) ** 2
    return float(np.sum(w * target_slim * model) / np.sum(w * model * model))


# ------------------------------------------------------------------ dpsi inversion
def dpsi_inversion(masked_imaging, residual, truth, flux_scale, reg_coeff, factor, preloads=None):
    """One linear potential-correction inversion of `residual` (slim, data units)."""
    lens, source = smooth_galaxies(truth, flux_scale)
    grid = np.asarray(masked_imaging.grid.slim)
    src_pos = grid - np.asarray(lens.deflections_yx_2d_from(grid=masked_imaging.grid.slim))
    src_grad = AnalyticSrcFactory(source).eval_grad(src_pos[:, 1], src_pos[:, 0])  # (dS/dy, dS/dx) at ray-traced positions

    pix = DpsiPixelization(mesh=RegularDpsiMesh(factor=factor), regularization=aa.reg.CurvatureMask(coefficient=reg_coeff))
    fit = FitDpsiImaging(masked_imaging=masked_imaging, image_residual=residual, source_gradient=src_grad,
                         dpsi_pixelization=pix, preloads=preloads)
    dpsi = np.asarray(fit.solve_dpsi())
    pair = fit.pair_dpsi_data_obj
    noise = np.asarray(masked_imaging.noise_map.slim)
    return {"fit": fit, "dpsi": dpsi, "dkappa": 0.5 * np.asarray(pair.hamiltonian_dpsi @ dpsi), "pair": pair,
            "log_evidence": float(fit.log_evidence),
            "chi2_residual_in": float(np.sum((residual / noise) ** 2)),
            "chi2_after_dpsi": float(np.sum(((residual - np.asarray(fit.model_image_residual_slim)) / noise) ** 2)),
            "n_data": int(residual.size), "n_dpsi": int(dpsi.size), "preloads": {"psf_mat": fit.psf_mat}}


def true_dkappa_on_mesh(truth, pair):
    """The subhalo's tNFW convergence on the dpsi mesh (lenstronomy, same world coordinates)."""
    kw = LensRenderer._kwargs_lens(truth, include_subhalo=True, include_multipole=False)[-1]
    return np.asarray(LensModel(["TNFW"]).kappa(pair.xgrid_dpsi_1d, pair.ygrid_dpsi_1d, [kw]))


def to_native(slim, mask2d):
    out = np.full(mask2d.shape, np.nan)
    out[~mask2d] = slim
    return out


def statistic(res, truth, lc, true_dk=None):
    """Peak-based candidates on the dkappa mesh map."""
    pair, dk = res["pair"], res["dkappa"]
    mad = 1.4826 * np.median(np.abs(dk - np.median(dk))) + 1e-30
    i = int(np.argmax(np.abs(dk)))
    peak_x, peak_y = float(pair.xgrid_dpsi_1d[i]), float(pair.ygrid_dpsi_1d[i])
    sub = truth.get("subhalo")
    r = np.hypot(pair.xgrid_dpsi_1d - peak_x, pair.ygrid_dpsi_1d - peak_y)
    aper = r <= 2.0 * pair.dpix_dpsi
    area = pair.dpix_dpsi ** 2
    m_proxy = float(lc.sigma_crit_angle * np.sum(dk[aper]) * area)
    out = {"peak_abs_dkappa": float(np.abs(dk[i])), "peak_signed_dkappa": float(dk[i]), "robust_std_dkappa": float(mad),
           "peak_significance_mad": float(np.abs(dk[i]) / mad), "peak_x": peak_x, "peak_y": peak_y,
           "peak_offset_from_true_subhalo_arcsec": float(np.hypot(peak_x - sub["x"], peak_y - sub["y"])) if sub else None,
           "aperture_mass_proxy_Msun": m_proxy, "n_dpsi_pixels": int(dk.size)}
    if true_dk is not None:
        r_true = np.hypot(pair.xgrid_dpsi_1d - sub["x"], pair.ygrid_dpsi_1d - sub["y"])
        ap_true = r_true <= 2.0 * pair.dpix_dpsi
        out["true_tnfw_mass_in_same_aperture_at_true_position_Msun"] = float(lc.sigma_crit_angle * np.sum(true_dk[ap_true]) * area)
        out["recovered_mass_in_aperture_at_true_position_Msun"] = float(lc.sigma_crit_angle * np.sum(dk[ap_true]) * area)
        out["true_dkappa_peak_on_mesh"] = float(true_dk.max())
        out["dkappa_at_true_subhalo_mesh_pixel"] = float(dk[int(np.argmin(r_true))])
        c = np.corrcoef(dk, true_dk)[0, 1]
        out["corr_with_true_dkappa"] = float(c) if np.isfinite(c) else None
    return out


# ----------------------------------------------------------------------- figure
def figure(res_sub, res_twin, truth, dsub, dtwin, true_dk, mask2d, ps, path, reg_coeff, smooth_label):
    L = mask2d.shape[0] * ps / 2
    ext = [-L, L, -L, L]
    sub = truth["subhalo"]
    md = res_sub["pair"].mask_dpsi
    vmaxk = np.nanmax(np.abs(np.concatenate([res_sub["dkappa"], res_twin["dkappa"]])))
    lc = LensCosmo(z_lens=truth["cosmology"]["z_lens"], z_source=truth["cosmology"]["z_source"])
    fig, axes = plt.subplots(2, 5, figsize=(18.5, 7.4))
    for row, (res, data, label) in enumerate([(res_sub, dsub, "subhalo present"), (res_twin, dtwin, "twin: no subhalo, same noise")]):
        mi = res["fit"].masked_imaging
        norm_resid = to_native(np.asarray(res["fit"].input_image_residual) / np.asarray(mi.noise_map.slim), mask2d)
        st = statistic(res, truth, lc)
        last = (to_native(true_dk, md), "magma", (0, None), r"true tNFW $\kappa$ on mesh") if row == 0 else \
               (to_native(res_sub["dkappa"] - res_twin["dkappa"], md), "RdBu_r", (-vmaxk, vmaxk), r"$\delta\kappa$(subhalo) $-$ $\delta\kappa$(twin)")
        panels = [(np.arcsinh(to_pal(data) / to_pal(data).max() * 10), "inferno", (None, None), f"data ({label})"),
                  (norm_resid, "RdBu_r", (-5, 5), "residual / noise fed to inversion"),
                  (to_native(res["dpsi"], md), "RdBu_r", "sym", r"$\delta\psi$ (mesh)"),
                  (to_native(res["dkappa"], md), "RdBu_r", (-vmaxk, vmaxk), r"$\delta\kappa=\frac{1}{2}\nabla^2\delta\psi$"),
                  last]
        for col, (img, cmap, v, title) in enumerate(panels):
            ax = axes[row, col]
            kw = dict(origin="upper", extent=ext, cmap=cmap)
            if v == "sym":
                m = np.nanmax(np.abs(img)); kw.update(vmin=-m, vmax=m)
            else:
                kw.update(vmin=v[0], vmax=v[1])
            ax.imshow(img, **kw)
            ax.plot(sub["x"], sub["y"], "o", mfc="none", mec="lime" if row == 0 else "gray", ms=14, mew=1.8)
            if col in (2, 3):
                ax.plot(st["peak_x"], st["peak_y"], "x", color="k", ms=9, mew=1.5)
            ax.set_title(title, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Family B prototype, idx {truth['_index']} -- PyAutoLens potential_correction, linear $\\delta\\psi$ inversion, "
                 f"{smooth_label} macro model, CurvatureMask coeff={reg_coeff:g} (max evidence)\n"
                 f"circle: true subhalo (log10 M={sub['log10_M200']:.2f}, c={sub['concentration']:.0f}); x: |dkappa| peak", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int, default=104, help="test_fixed60 index")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--factor", type=int, default=2, help="dpsi mesh coarsening factor")
    ap.add_argument("--smooth", choices=["oracle", "pyautolens"], default="oracle")
    ap.add_argument("--reg", type=float, nargs="+", default=[1e1, 1e2, 1e3, 1e4, 1e5, 1e6], help="CurvatureMask coefficients")
    args = ap.parse_args()

    truths = [json.loads(l) for l in open(ROOT / "data/test_fixed60/truth.jsonl")]
    manifest = json.loads((ROOT / "data/test_fixed60/manifest.json").read_text())
    kwargs_band, num_pix = manifest["kwargs_band"], manifest["image_shape"][0]
    ps = kwargs_band["pixel_scale"]
    truth = truths[args.idx]
    cfg = tier0_tsang(truth["subhalo"]["concentration_mode"])
    lc = LensCosmo(z_lens=truth["cosmology"]["z_lens"], z_source=truth["cosmology"]["z_source"])

    t0 = time.perf_counter()
    d_sub, d_twin, macro_noiseless, full_noiseless = render_pair(truth, cfg, args.seed)
    n_sub, n_twin = noise_map_for(d_sub, kwargs_band, num_pix), noise_map_for(d_twin, kwargs_band, num_pix)
    mask = arc_mask(d_sub, n_sub, ps)
    mask2d = np.asarray(mask)
    mi_sub = build_imaging(d_sub, n_sub, kwargs_band).apply_mask(mask=mask)
    mi_twin = build_imaging(d_twin, n_twin, kwargs_band).apply_mask(mask=mask)
    macro_slim = to_pal(macro_noiseless)[~mask2d]
    t_setup = time.perf_counter() - t0

    # flux scale + cross-code mismatch diagnostic (PyAutoLens smooth model vs lenstronomy's noiseless macro image)
    k = fit_flux_scale(mi_sub, truth, macro_slim)
    pal_model, _, _ = pal_smooth_model(mi_sub, truth, k)
    noise_slim = np.asarray(mi_sub.noise_map.slim)
    chi2_crosscode = float(np.sum(((macro_slim - pal_model) / noise_slim) ** 2))
    if args.smooth == "oracle":
        resid_sub, resid_twin = np.asarray(mi_sub.data.slim) - macro_slim, np.asarray(mi_twin.data.slim) - macro_slim
    else:
        resid_sub, resid_twin = np.asarray(mi_sub.data.slim) - pal_model, np.asarray(mi_twin.data.slim) - pal_model
    sig_slim = to_pal(full_noiseless - macro_noiseless)[~mask2d]
    m_lens = np.pi * truth["lens_macro"]["theta_E"] ** 2 * lc.sigma_crit_angle
    print(f"idx={args.idx} q={truth['lens_macro']['q']:.3f} thetaE={truth['lens_macro']['theta_E']:.3f} "
          f"subhalo log10M={truth['subhalo']['log10_M200']:.2f} (= {100*10**truth['subhalo']['log10_M200']/m_lens:.1f}% of M(<thetaE)) "
          f"at ({truth['subhalo']['x']:.3f},{truth['subhalo']['y']:.3f}); unmasked={int((~mask2d).sum())}; setup {t_setup:.1f}s")
    print(f"flux scale PyAutoLens->lenstronomy {k:.4g}; cross-code smooth-model mismatch chi2/N = {chi2_crosscode/macro_slim.size:.2f}; "
          f"true signature: peak {np.abs(sig_slim/noise_slim).max():.0f} sigma/pixel, chi2/N = {np.sum((sig_slim/noise_slim)**2)/sig_slim.size:.1f}")

    out = {"idx": args.idx, "seed": args.seed, "dpsi_factor": args.factor, "smooth_model": args.smooth, "flux_scale": k,
           "n_unmasked": int((~mask2d).sum()), "truth": {"lens_macro": truth["lens_macro"], "subhalo": truth["subhalo"], "source": truth["source"]},
           "subhalo_mass_fraction_of_M_within_thetaE": float(10 ** truth["subhalo"]["log10_M200"] / m_lens),
           "crosscode_smooth_model_chi2_per_pixel": chi2_crosscode / macro_slim.size,
           "true_signature_peak_sigma_per_pixel": float(np.abs(sig_slim / noise_slim).max()),
           "true_signature_chi2_per_pixel": float(np.sum((sig_slim / noise_slim) ** 2) / sig_slim.size), "sweep": []}

    best, preloads, true_dk = None, None, None
    for reg in args.reg:
        t1 = time.perf_counter()
        rs = dpsi_inversion(mi_sub, resid_sub, truth, k, reg, args.factor, preloads)
        t_first = time.perf_counter() - t1
        preloads = rs["preloads"]  # PSF matrix depends only on the mask; reuse across fits
        t2 = time.perf_counter()
        rt = dpsi_inversion(mi_twin, resid_twin, truth, k, reg, args.factor, preloads)
        t_second = time.perf_counter() - t2
        if true_dk is None:
            true_dk = true_dkappa_on_mesh(truth, rs["pair"])
        ss, st = statistic(rs, truth, lc, true_dk), statistic(rt, truth, lc, true_dk)
        row = {"reg_coeff": reg, "seconds_first_fit_incl_psf_matrix": t_first, "seconds_fit_with_preloads": t_second,
               "n_dpsi_pixels": rs["n_dpsi"], "n_data_pixels": rs["n_data"],
               "subhalo": {k_: rs[k_] for k_ in ("log_evidence", "chi2_residual_in", "chi2_after_dpsi")} | ss,
               "twin": {k_: rt[k_] for k_ in ("log_evidence", "chi2_residual_in", "chi2_after_dpsi")} | st}
        out["sweep"].append(row)
        print(f"reg={reg:8.3g} | SUB: logZ={rs['log_evidence']:9.1f} chi2/N {rs['chi2_residual_in']/rs['n_data']:.2f}->{rs['chi2_after_dpsi']/rs['n_data']:.2f} "
              f"peak|dk|={ss['peak_abs_dkappa']:.3f} sig={ss['peak_significance_mad']:.1f} off={ss['peak_offset_from_true_subhalo_arcsec']:.2f}\" "
              f"corr={ss['corr_with_true_dkappa']:.2f} Map={ss['recovered_mass_in_aperture_at_true_position_Msun']:.2e}/{ss['true_tnfw_mass_in_same_aperture_at_true_position_Msun']:.2e} "
              f"| TWIN: logZ={rt['log_evidence']:9.1f} chi2/N {rt['chi2_residual_in']/rt['n_data']:.2f}->{rt['chi2_after_dpsi']/rt['n_data']:.2f} "
              f"peak|dk|={st['peak_abs_dkappa']:.3f} sig={st['peak_significance_mad']:.1f} | {t_first:.1f}s/{t_second:.1f}s")
        if best is None or rs["log_evidence"] > best[0]:
            best = (rs["log_evidence"], reg, rs, rt)

    _, reg_best, rs, rt = best
    out["max_evidence_reg_coeff_subhalo_image"] = reg_best
    out["max_evidence_reg_coeff_twin"] = max(out["sweep"], key=lambda r: r["twin"]["log_evidence"])["reg_coeff"]
    tag = f"idx{args.idx}" + ("" if args.smooth == "oracle" and args.factor == 2 else f"_{args.smooth}_f{args.factor}")
    figure(rs, rt, truth, d_sub, d_twin, true_dk, mask2d, ps, OUT / f"prototype_{tag}.png", reg_best, args.smooth)
    (OUT / f"prototype_{tag}.json").write_text(json.dumps(out, indent=2, default=float))
    combined_path = OUT / "prototype.json"
    combined = json.loads(combined_path.read_text()) if combined_path.exists() else {}
    combined[tag] = out
    combined_path.write_text(json.dumps(combined, indent=2, default=float))
    print(f"\nmax-evidence reg: subhalo image {reg_best:g}, twin {out['max_evidence_reg_coeff_twin']:g}; wrote {OUT}/prototype_{tag}.png/.json and prototype.json[{tag}]")


if __name__ == "__main__":
    main()
