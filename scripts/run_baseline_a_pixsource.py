"""Family A with a REGULARIZED PIXELIZED source -- the published pipelines' source model.

Referee round 9, point M4, second attempt. Every multipole false-positive rate in this paper is
measured with an analytic Sersic source, and we call those rates upper limits because a pixelized
source can absorb part of an m=4 residual. A first attempt with an unregularized shapelet basis
failed: a free source plus a free perturber absorb noise and the statistic loses its null
(`run_baseline_a_shapelet.py`). The missing ingredient was the regularization, so here the source is
a rectangular pixel grid with curvature regularization, inverted by PyAutoLens exactly as in
gravitational imaging, and the statistic is the Bayesian log evidence, whose Occam term is what
keeps a free source honest.

    smooth fit  : EPL + shear (from the Sersic-source fit) + pixelized regularized source
    scan        : add a TNFW at each of 3 radii x 8 angles x 3 masses, re-invert the source
    statistic   : Delta log Z = max_grid (log Z_perturbed - log Z_smooth)

    OMP_NUM_THREADS=1 python scripts/run_baseline_a_pixsource.py --population multipole_m4_a3 \
        --n 60 --seed 3 --workers 3 --out results/baseline_a_pixsource/multipole_m4_a3
"""
import argparse
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
os.environ.setdefault("PYAUTO_SKIP_WORKSPACE_VERSION_CHECK", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import autolens as al  # noqa: E402
import autoarray as aa  # noqa: E402
from lenstronomy.SimulationAPI.sim_api import SimAPI  # noqa: E402
from lenstronomy.Cosmo.lens_cosmo import LensCosmo  # noqa: E402
from lenstronomy.Util import param_util  # noqa: E402
from baseline_a.fit import fit_smooth  # noqa: E402

LC = LensCosmo(z_lens=0.5, z_source=1.0)
G = {}


def _init(pop_dir, cfg):
    man = json.loads((pop_dir / "manifest.json").read_text())
    G.update(cfg)
    G["kb"], G["npix"] = man["kwargs_band"], man["image_shape"][0]
    G["ps"] = man["kwargs_band"]["pixel_scale"]
    G["truths"] = [json.loads(l) for l in open(pop_dir / "truth.jsonl")]
    G["images"] = np.load(pop_dir / "images_full_noisy.npy", mmap_mode="r")


def _imaging(data):
    """PyAutoLens Imaging from one lenstronomy image, with lenstronomy's own noise map."""
    sim = SimAPI(num_pix=G["npix"], kwargs_single_band=G["kb"],
                 kwargs_model={"lens_model_list": ["EPL"], "source_light_model_list": ["SERSIC_ELLIPSE"]})
    noise = sim.estimate_noise(data)
    img = aa.Array2D.no_mask(values=np.asarray(data, dtype=float), pixel_scales=G["ps"])
    nse = aa.Array2D.no_mask(values=np.asarray(noise, dtype=float), pixel_scales=G["ps"])
    psf_sigma = G["kb"].get("seeing", G["ps"]) / 2.3548
    psf = al.Convolver.from_gaussian(shape_native=(11, 11), pixel_scales=G["ps"], sigma=psf_sigma, normalize=True)
    return al.Imaging(data=img, noise_map=nse, psf=psf), noise


def _lens_galaxy(vec, extra=None):
    """EPL + external shear from the Sersic-source macro fit, optionally plus a TNFW perturber."""
    theta_E, gamma, e1, e2, g1, g2 = vec[:6]
    phi, q = param_util.ellipticity2phi_q(e1, e2)
    mass = al.mp.PowerLaw(centre=(0.0, 0.0),
                          ell_comps=al.convert.ell_comps_from(axis_ratio=float(q), angle=float(np.degrees(phi))),
                          einstein_radius=float(theta_E), slope=float(gamma))
    phi_s, g_s = param_util.shear_cartesian2polar(g1, g2)
    shear = al.mp.ExternalShear(gamma_2=float(g2), gamma_1=float(g1))
    kw = {"redshift": 0.5, "mass": mass, "shear": shear}
    if extra is not None:
        kw["perturber"] = extra
    return al.Galaxy(**kw)


def _perturber(log10_m, x, y, tau=20.0):
    Rs, alpha_Rs = LC.nfw_physical2angle(M=10 ** log10_m, c=float(G["conc"]))
    kappa_s = alpha_Rs / (4.0 * Rs)   # PyAutoLens parameterises NFW by kappa_s, not alpha_Rs
    return al.mp.NFWTruncatedSph(centre=(float(y), float(x)), kappa_s=float(kappa_s),
                                 scale_radius=float(Rs), truncation_radius=float(tau * Rs))


def _source_galaxy():
    pix = al.Pixelization(mesh=al.mesh.RectangularUniform(shape=(G["mesh"], G["mesh"])),
                          regularization=aa.reg.Constant(coefficient=G["reg"]))
    return al.Galaxy(redshift=1.0, pixelization=pix)


def _log_evidence(imaging, vec, extra=None):
    try:
        tracer = al.Tracer(galaxies=[_lens_galaxy(vec, extra), _source_galaxy()])
        fit = al.FitImaging(dataset=imaging, tracer=tracer)
        v = float(fit.log_evidence)
        return v if np.isfinite(v) else None
    except Exception:
        return None


def one_lens(args):
    i, fit_seed = args
    t0 = time.time()
    truth = G["truths"][i]
    data = np.asarray(G["images"][i])
    # macro parameters come from the Sersic-source fit, exactly as Family B does: this isolates the
    # effect of the SOURCE model, with the macro model held at what the cheap fit already found
    vec, chi2_sersic, _, _ = fit_smooth(data, truth, G["kb"], G["npix"],
                                        rng=np.random.default_rng(fit_seed), maxiter=60)
    imaging, noise = _imaging(data)
    mask = al.Mask2D.circular(shape_native=imaging.shape_native, pixel_scales=G["ps"], radius=G["mask_radius"])
    imaging = imaging.apply_mask(mask=mask)
    t_fit = time.time() - t0

    lnz0 = _log_evidence(imaging, vec)
    t1 = time.time()
    best = {"delta_chi2": -np.inf, "x": None, "y": None, "log10_m": None}
    if lnz0 is not None:
        theta_E = truth["lens_macro"]["theta_E"]
        for r in np.linspace(0.5, 1.4, 3) * theta_E:
            for a in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                x, y = r * np.cos(a), r * np.sin(a)
                for log10_m in (8.5, 9.5, 10.5):
                    lnz = _log_evidence(imaging, vec, _perturber(log10_m, x, y))
                    if lnz is None:
                        continue
                    d = lnz - lnz0
                    if d > best["delta_chi2"]:
                        best = {"delta_chi2": float(d), "x": float(x), "y": float(y), "log10_m": float(log10_m)}
    return {"index": int(i), "has_subhalo": truth.get("subhalo") is not None,
            "log10_M200_true": (truth["subhalo"] or {}).get("log10_M200"),
            "concentration_true": (truth["subhalo"] or {}).get("concentration"),
            "has_multipole": truth.get("multipole") is not None,
            # `delta_chi2` holds Delta log Z so the existing evaluators read this unchanged;
            # `reliable_fit` marks a usable inversion rather than a chi^2/dof gate
            "delta_chi2": best["delta_chi2"] if np.isfinite(best["delta_chi2"]) else None,
            "log_evidence_smooth": lnz0, "reliable_fit": bool(lnz0 is not None and np.isfinite(best["delta_chi2"])),
            "chi2_smooth_per_dof": chi2_sersic / G["npix"] ** 2,
            "best_x": best["x"], "best_y": best["y"], "best_log10_m": best["log10_m"],
            "source": "pixelized_regularized", "mesh": G["mesh"], "reg_coefficient": G["reg"],
            "t_fit_s": t_fit, "t_scan_s": time.time() - t1}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--population", required=True)
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--n-subsample", type=int, default=300)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--mesh", type=int, default=20, help="source pixel grid is mesh x mesh")
    p.add_argument("--reg", type=float, default=1.0, help="constant regularization coefficient")
    p.add_argument("--mask-radius", type=float, default=2.0)
    p.add_argument("--concentration", type=float, default=15.0)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    _init(ROOT / "data" / args.population,
          {"mesh": args.mesh, "reg": args.reg, "mask_radius": args.mask_radius, "conc": args.concentration})
    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(len(G["truths"]), size=min(args.n_subsample, len(G["truths"])), replace=False))[: args.n]
    jobs = [(int(i), int(args.seed) * 100000 + k) for k, i in enumerate(idx)]
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    import multiprocessing as mp
    with mp.get_context("fork").Pool(args.workers) as pool, open(args.out / "scan_results.jsonl", "w") as f:
        for k, rec in enumerate(pool.imap_unordered(one_lens, jobs)):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if (k + 1) % 5 == 0:
                print(f"  {k+1}/{len(jobs)}  ({time.time()-t0:.0f}s)", flush=True)
    recs = sorted((json.loads(l) for l in open(args.out / "scan_results.jsonl")), key=lambda r: r["index"])
    with open(args.out / "scan_results.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    (args.out / "manifest.json").write_text(json.dumps({
        "population": args.population, "n": len(recs), "seed": args.seed,
        "source_model": "pixelized rectangular mesh with constant regularization",
        "mesh": args.mesh, "reg_coefficient": args.reg, "mask_radius": args.mask_radius,
        "statistic": "Delta log Bayesian evidence", "concentration_assumed": args.concentration,
        "n_failed_inversions": sum(1 for r in recs if not r["reliable_fit"]),
        "elapsed_s": time.time() - t0, "workers": args.workers}, indent=2))
    print(f"wrote {len(recs)} pixelized-source scans to {args.out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
