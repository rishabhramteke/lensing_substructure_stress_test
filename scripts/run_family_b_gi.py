"""Family B-GI: full gravitational imaging (Koopmans 2005; Vegetti & Koopmans 2009) with the joint pixelized source and
pixelized potential-correction inversion of PyAutoLens `autolens.potential_correction` (Cao et al. 2025), scored by
the Bayesian evidence gain of the potential corrections.

Per lens:
  1. the smooth EPL+shear+Sersic(+lens light) fit of Family A (`fit_smooth`, same near-truth idealization), lens light
     subtracted; the lens model is converted to PyAutoLens (theta_E rescaled per lens so the deflections match
     lenstronomy's to machine precision) and held fixed;
  2. source-only inversion: pixelized source (adaptive-density rectangular mesh, constant regularization), its
     regularization strength set by maximizing the evidence -> lnZ_src;
  3. joint inversion [F_src | -B D_s D_psi] with the source gradient taken from the step-2 pixelized source (no
     parametric source anywhere) and curvature regularization on dpsi, both regularization strengths set by maximizing
     the evidence -> lnZ_joint;
  4. statistic delta_lnZ = lnZ_joint - lnZ_src (written also as `delta_chi2`, the alias Family A's evaluators read);
     position = peak of |dkappa| = |0.5 lap dpsi| of the joint solution.

The evidences are computed from the library's matrices (same terms as FitDpsiSrcImaging.log_evidence) so the two
regularization strengths can be optimized without rebuilding the mapping matrices.

    python scripts/run_family_b_gi.py --population test_fixed60 --seed 0 --n 300 --out results/baseline_bgi/test_fixed60
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("PYAUTO_SKIP_WORKSPACE_VERSION_CHECK", "1")
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

import numpy as np
from scipy.optimize import minimize, minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))

import autolens as al  # noqa: E402
import autoarray as aa  # noqa: E402
from autolens.potential_correction import FitDpsiSrcImaging, DpsiPixelization, RegularDpsiMesh, PixSrcFactoryITP  # noqa: E402
from autolens.potential_correction import util as pc_util  # noqa: E402
from autogalaxy.analysis.adapt_images.adapt_images import AdaptImages  # noqa: E402
from lenstronomy.LensModel.lens_model import LensModel  # noqa: E402
from lenstronomy.Cosmo.lens_cosmo import LensCosmo  # noqa: E402
from lenstronomy.Util import param_util  # noqa: E402
from baseline_a.fit import fit_smooth, _unpack3, lens_light_image  # noqa: E402
from run_family_b import to_pal, noise_map_for, build_imaging, arc_mask, CHI2_DOF_UNRELIABLE  # noqa: E402

_LM = LensModel(["EPL", "SHEAR"])


def pyautolens_lens(kwargs_lens, z_lens, grid_slim):
    """PyAutoLens PowerLaw+ExternalShear whose deflections equal lenstronomy's EPL+SHEAR on the masked grid (the two
    codes normalize theta_E differently for q < 1; the rescale is fitted, the rest maps one to one)."""
    kl, ks = kwargs_lens[0], kwargs_lens[1]
    phi, q = param_util.ellipticity2phi_q(kl["e1"], kl["e2"])
    y, x = grid_slim[:, 0], grid_slim[:, 1]
    ax, ay = _LM.alpha(x, y, kwargs_lens[:2])
    gi = al.Grid2DIrregular(values=np.stack([y, x], 1))

    def lens(theta):
        return al.Galaxy(redshift=z_lens,
                         mass=al.mp.PowerLaw(centre=(0.0, 0.0), ell_comps=al.convert.ell_comps_from(axis_ratio=float(q), angle=float(np.degrees(phi))),
                                             einstein_radius=float(theta), slope=kl["gamma"]),
                         shear=al.mp.ExternalShear(gamma_1=ks["gamma1"], gamma_2=ks["gamma2"]))

    def cost(theta):
        a = np.asarray(lens(theta).deflections_yx_2d_from(grid=gi))
        return float(np.sum((a[:, 0] - ay) ** 2 + (a[:, 1] - ax) ** 2))

    t = minimize_scalar(cost, bounds=(0.8 * kl["theta_E"], 1.3 * kl["theta_E"]), method="bounded", options={"xatol": 1e-7}).x
    return lens(t), float(np.sqrt(cost(t) / len(y)))


def build_imaging_exact(data_l, noise_l, kwargs_band, num_pix):
    """PyAutoLens imaging with the simulator's own PSF response (lenstronomy's multi-Gaussian convolution at the
    rendering resolution, measured on a unit impulse), so the forward model matches the one the data were made with."""
    from lenstronomy.SimulationAPI.sim_api import SimAPI
    ps = kwargs_band["pixel_scale"]
    sim = SimAPI(num_pix=num_pix, kwargs_single_band=kwargs_band, kwargs_model={"lens_model_list": ["EPL"], "source_light_model_list": ["SERSIC_ELLIPSE"]})
    conv = sim.image_model_class(kwargs_numerics={"supersampling_factor": 1}).ImageNumerics.convolution_class
    imp = np.zeros((num_pix, num_pix)); c = num_pix // 2; imp[c, c] = 1.0
    k = conv.convolution2d(imp)[c - 5:c + 6, c - 5:c + 6]
    psf = al.Convolver(kernel=al.Array2D.no_mask(values=k[::-1, :].copy(), pixel_scales=ps), normalize=True)
    return al.Imaging(data=al.Array2D.no_mask(values=to_pal(data_l), pixel_scales=ps),
                      noise_map=al.Array2D.no_mask(values=to_pal(noise_l), pixel_scales=ps), psf=psf)


def logdet_pd(m):
    s, v = np.linalg.slogdet(m)
    if s != 1:
        raise np.linalg.LinAlgError("not positive definite")
    return v


class Evidence:
    """lnZ(lam) for a linear inversion with mapping M (columns grouped in blocks) and block regularizations lam_k R_k."""
    def __init__(self, data, noise, blocks, regs):
        w = 1.0 / noise ** 2
        self.M = np.hstack(blocks); self.d = data; self.w = w
        self.F = self.M.T @ (w[:, None] * self.M); self.D = self.M.T @ (w * data)
        self.sizes = [b.shape[1] for b in blocks]
        self.R = [np.asarray(r.toarray() if hasattr(r, "toarray") else r) for r in regs]
        self.ld_R = [logdet_pd(r) for r in self.R]
        self.noise_term = -0.5 * float(np.sum(np.log(2 * np.pi * noise ** 2)))

    def __call__(self, log_lams, full=False):
        lams = 10.0 ** np.asarray(log_lams)
        R = np.zeros_like(self.F); o = 0
        for lam, r, n in zip(lams, self.R, self.sizes):
            R[o:o + n, o:o + n] = lam * r; o += n
        A = self.F + R
        try:
            sol = np.linalg.solve(A, self.D); ld_A = logdet_pd(A)
        except np.linalg.LinAlgError:
            return (-1e30, None) if full else -1e30
        resid = self.d - self.M @ sol
        chi2 = float(np.sum(self.w * resid ** 2))
        ld_R = sum(n * np.log(lam) + l for lam, n, l in zip(lams, self.sizes, self.ld_R))
        lnz = self.noise_term - 0.5 * chi2 - 0.5 * float(sol @ R @ sol) - 0.5 * ld_A + 0.5 * ld_R
        return (lnz, {"sol": sol, "chi2": chi2}) if full else lnz


def maximize(ev, x0, bounds):
    f = lambda x: -ev(np.clip(x, [b[0] for b in bounds], [b[1] for b in bounds]))  # noqa: E731
    if len(x0) == 1:
        r = minimize_scalar(lambda t: f([t]), bounds=bounds[0], method="bounded", options={"xatol": 0.02})
        x = np.array([r.x])
    else:
        starts = [(a, b) for a in np.linspace(*bounds[0], 5) for b in np.linspace(*bounds[1], 7)] + [tuple(x0)]
        best = min((f(np.array(st)), st) for st in starts)[1]
        r = minimize(f, np.array(best), method="Nelder-Mead", options={"xatol": 0.02, "fatol": 0.01, "maxfev": 150})
        x = np.clip(r.x, [b[0] for b in bounds], [b[1] for b in bounds])
    lnz, extra = ev(x, full=True)
    return lnz, x, extra


def source_blocks(mi, lens, src_pix, adapt_img, mesh_grid=None):
    """The source-only PyAutoLens inversion at the fixed lens: mapper, PSF-convolved mapping matrix and regularization
    matrix at coefficient 1 (the mesh adapts to the parametric fit's lensed-source image, as in the published
    pipelines)."""
    src = al.Galaxy(redshift=1.0, pixelization=src_pix)
    kw = {}
    if adapt_img is not None:
        kw["galaxy_image_dict"] = {src: adapt_img}
    if mesh_grid is not None:
        kw["galaxy_image_plane_mesh_grid_dict"] = {src: mesh_grid}
    adapt = AdaptImages(**kw) if kw else None
    fi = al.FitImaging(dataset=mi, tracer=al.Tracer(galaxies=[lens, src]), adapt_images=adapt,
                       settings=aa.Settings(use_positive_only_solver=False, use_border_relocator=True))
    inv = fi.inversion
    return inv.linear_obj_list[0], np.asarray(inv.operated_mapping_matrix), inv.regularization_matrix


def gi_lens(data_sub, noise, kwargs_band, ps, kwargs_lens, z_lens, src_shape, factor, lensed_src=None, mesh="adapt_image", weight_power=1.0, weight_floor=0.1):
    mask = arc_mask(data_sub, noise, ps)
    # point sampling at pixel centres and the simulator's PSF: the forward model of the simulations (and of Family A)
    mi = build_imaging_exact(data_sub, noise, kwargs_band, data_sub.shape[0]).apply_mask(mask=mask)
    mi = mi.apply_over_sampling(over_sample_size_lp=1, over_sample_size_pixelization=1)
    lens, defl_rms = pyautolens_lens(kwargs_lens, z_lens, np.asarray(mi.grid.slim))
    mesh_grid, reg = None, al.reg.Constant(coefficient=1.0)
    if mesh == "adapt_image":
        m = al.mesh.RectangularBilinearAdaptImage(shape=src_shape, weight_power=weight_power, weight_floor=weight_floor)
        adapt = al.Array2D(values=np.asarray(to_pal(lensed_src))[~np.asarray(mask)], mask=mask)
    elif mesh == "delaunay":
        # source vertices = every `skip`-th unmasked image pixel ray-traced to the source plane (Vegetti & Koopmans
        # 2009), so the source resolution follows the magnification; `src_shape[0]` is the skip
        adapt = None
        skip = int(src_shape[0]); m2 = np.asarray(mask)
        rr, cc = np.nonzero(~m2)
        keep = (rr % skip == 0) & (cc % skip == 0)
        c0 = (m2.shape[0] - 1) / 2
        mesh_grid = al.Grid2DIrregular(values=np.stack([(c0 - rr[keep]) * ps, (cc[keep] - c0) * ps], 1))
        m, reg = al.mesh.Delaunay(pixels=int(keep.sum())), al.reg.ConstantSplit(coefficient=1.0)
    else:
        m, adapt = al.mesh.RectangularBilinearAdaptDensity(shape=src_shape), None
    src_pix = al.Pixelization(mesh=m, regularization=reg)
    mapper, Fs, Rs = source_blocks(mi, lens, src_pix, adapt, mesh_grid)
    dpsi_pix = DpsiPixelization(mesh=RegularDpsiMesh(factor=factor), regularization=aa.reg.CurvatureMask(coefficient=1.0))
    fit = FitDpsiSrcImaging(masked_imaging=mi, lens_start=lens, source_start=None, dpsi_pixelization=dpsi_pix, src_pixelization=src_pix,
                            preloads={"src_mapper": mapper, "src_map_mat": Fs, "src_reg_mat": Rs})
    d, n = np.asarray(fit.masked_imaging.data), np.asarray(fit.masked_imaging.noise_map)

    ev0 = Evidence(d, n, [Fs], [Rs])
    lnz0, x0, e0 = maximize(ev0, [0.0], [(-4.0, 4.0)])
    s_hat = e0["sol"]
    fit.source_start = PixSrcFactoryITP(points=np.asarray(fit.src_mapper.source_plane_mesh_grid), values=s_hat)
    Fd, Rd = np.asarray(fit.dpsi_mapping_matrix), fit.dpsi_regularization_matrix
    ev1 = Evidence(d, n, [Fs, Fd], [Rs, Rd])
    lnz1, x1, e1 = maximize(ev1, [x0[0], 12.0], [(-4.0, 4.0), (0.0, 12.0)])   # the start at the null's regularization and dpsi -> 0 bounds lnZ_joint below by lnZ_src
    pair = fit.pair_dpsi_data_obj
    dpsi = e1["sol"][Fs.shape[1]:]
    dk = 0.5 * np.asarray(pair.hamiltonian_dpsi @ dpsi)
    i = int(np.argmax(np.abs(dk)))
    return {"delta_lnZ": float(lnz1 - lnz0), "lnZ_src": float(lnz0), "lnZ_joint": float(lnz1),
            "log10_lam_src_null": float(x0[0]), "log10_lam_src": float(x1[0]), "log10_lam_dpsi": float(x1[1]),
            "chi2_src": e0["chi2"], "chi2_joint": e1["chi2"], "n_unmasked": int(len(d)), "n_src": int(Fs.shape[1]), "n_dpsi": int(Fd.shape[1]),
            "peak_abs_dkappa": float(np.abs(dk[i])), "best_x": float(pair.xgrid_dpsi_1d[i]), "best_y": float(pair.ygrid_dpsi_1d[i]),
            "deflection_match_rms": defl_rms}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--population", required=True)
    p.add_argument("--data-root", type=Path, default=ROOT / "data")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--seed", type=int, default=0, help="subsample selection seed (same rule as run_family_b.py)")
    p.add_argument("--only", type=int, nargs="*", help="run only these population indices (tests)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--shard", help="k/K: run every K-th lens of the subsample starting at k (parallel workers append to the same file)")
    p.add_argument("--src-shape", type=int, default=50)
    p.add_argument("--factor", type=int, default=2)
    p.add_argument("--mesh", choices=["adapt_image", "adapt_density", "delaunay"], default="adapt_image")
    p.add_argument("--lens-light-components", type=int, default=1, choices=[1, 2])
    args = p.parse_args()

    pop_dir = args.data_root / args.population
    manifest = json.loads((pop_dir / "manifest.json").read_text())
    kwargs_band, num_pix, ps = manifest["kwargs_band"], manifest["image_shape"][0], manifest["kwargs_band"]["pixel_scale"]
    truths = [json.loads(l) for l in open(pop_dir / "truth.jsonl")]
    noisy = pop_dir / "images_full_noisy.npy"
    images = np.load(noisy if noisy.exists() else pop_dir / "images_full_noiseless.npy", mmap_mode="r")

    rng_sel = np.random.default_rng(args.seed)
    n = min(args.n, len(truths))
    idx = np.sort(rng_sel.choice(len(truths), size=n, replace=False)) if n < len(truths) else np.arange(len(truths))
    if args.only:
        idx = np.array(args.only)
    if args.shard:
        k_, K_ = map(int, args.shard.split("/"))
        idx = idx[k_::K_]

    args.out.mkdir(parents=True, exist_ok=True)
    out_file = args.out / "scan_results.jsonl"
    done = set()
    if out_file.exists():
        for l in open(out_file):
            try:
                done.add(json.loads(l)["index"])
            except (json.JSONDecodeError, KeyError):
                pass   # a line cut by a killed worker; the lens is redone
    t0 = time.time(); n_err = 0
    with open(out_file, "a") as f:
        for k, i in enumerate(idx):
            if int(i) in done:
                continue
            truth = truths[i]
            rec = {"index": int(i), "has_subhalo": truth.get("subhalo") is not None,
                   "log10_M200_true": (truth["subhalo"] or {}).get("log10_M200"),
                   "concentration_true": (truth["subhalo"] or {}).get("concentration"),
                   "has_multipole": truth.get("multipole") is not None}
            try:
                t1 = time.time()
                data = np.asarray(images[i])
                noise = noise_map_for(data, kwargs_band, num_pix)
                vec, chi2_smooth, image_model, _ = fit_smooth(data, truth, kwargs_band, num_pix, rng=np.random.default_rng([args.seed + 1000, int(i)]),
                                                    maxiter=60, lens_light_components=args.lens_light_components)
                kwargs_lens, kwargs_source, kwargs_ll = _unpack3(vec)
                lensed = np.asarray(image_model.image(kwargs_lens=kwargs_lens, kwargs_source=kwargs_source, kwargs_lens_light=kwargs_ll, lens_light_add=False))
                ll_img = lens_light_image(vec, kwargs_band, num_pix) if kwargs_ll else 0.0
                chi2_dof = chi2_smooth / (num_pix * num_pix)
                t_fit = time.time() - t1
                res = gi_lens(data - ll_img, noise, kwargs_band, ps, kwargs_lens, truth["cosmology"]["z_lens"], (args.src_shape, args.src_shape), args.factor,
                              lensed_src=lensed, mesh=args.mesh)
                rec.update(res)
                rec.update({"delta_chi2": res["delta_lnZ"], "chi2_smooth_per_dof": float(chi2_dof), "reliable_fit": bool(chi2_dof < CHI2_DOF_UNRELIABLE),
                            "best_log10_m": None, "t_fit_s": t_fit, "t_scan_s": time.time() - t1 - t_fit})
            except Exception as e:  # noqa: BLE001
                n_err += 1
                rec.update({"error": f"{type(e).__name__}: {e}", "reliable_fit": False, "delta_chi2": 0.0, "delta_lnZ": 0.0,
                            "best_x": None, "best_y": None, "best_log10_m": None, "t_scan_s": 0.0})
            f.write(json.dumps(rec, default=float) + "\n"); f.flush()
            if (k + 1) % 10 == 0 or k + 1 == len(idx):
                print(f"  {args.population} {k+1}/{len(idx)}  {time.time()-t0:.0f}s  {n_err} errors", flush=True)

    if not args.only and not args.shard:
        (args.out / "manifest.json").write_text(json.dumps({
            "family": "B-GI (joint pixelized source + pixelized dpsi, evidence-set regularizations, PyAutoLens potential_correction; Cao et al. 2025)",
            "population": args.population, "n": n, "seed": args.seed, "src_shape": args.src_shape, "mesh": args.mesh, "mesh_factor": args.factor,
            "statistic": "delta_lnZ = lnZ(source+dpsi) - lnZ(source), written also as delta_chi2 (alias)",
            "chi2_dof_unreliable_threshold": CHI2_DOF_UNRELIABLE, "n_errors": n_err, "elapsed_s": time.time() - t0,
            "source_manifest": manifest}, indent=2, default=str))


if __name__ == "__main__":
    main()
