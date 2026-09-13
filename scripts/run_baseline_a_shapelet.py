"""Family A with a FLEXIBLE (shapelet) source instead of the analytic Sersic.

Referee round 9, point M4. Every multipole false-positive rate in this paper is measured with an
analytic Sersic source, and we have been calling those rates upper limits because a pixelized or
regularized source -- which the published pipelines fit -- could absorb part of an m=4 residual
and so lower the rate. That is a caveat we can turn into a measurement.

The source here is a shapelet basis of order n_max (28 coefficients at n_max=6). Its amplitudes
are linear in the data, so lenstronomy solves them analytically at every step
(`ImageLinearFit.image_linear_solve`); only the 6 macro parameters plus the shapelet scale beta
and centre are fitted non-linearly. The scan is unchanged: a TNFW is added at each of 72 grid
cells and the source is re-solved, so Delta chi^2 is measured against a source free to
re-arrange itself in response to the perturber -- the thing an analytic Sersic cannot do.

    OMP_NUM_THREADS=1 python scripts/run_baseline_a_shapelet.py --population multipole_m4_a3 \
        --n 100 --seed 3 --workers 4 --out results/baseline_a_shapelet/multipole_m4_a3
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lenstronomy.SimulationAPI.sim_api import SimAPI  # noqa: E402
from lenstronomy.ImSim.image_linear_solve import ImageLinearFit  # noqa: E402
from lenstronomy.Cosmo.lens_cosmo import LensCosmo  # noqa: E402

CHI2_DOF_UNRELIABLE = 10.0
LC = LensCosmo(z_lens=0.5, z_source=1.0)
G = {}
# theta_E, gamma, e1, e2, g1, g2, beta, src_x, src_y
BOUNDS = [(0.3, 3.0), (1.2, 2.9), (-0.6, 0.6), (-0.6, 0.6), (-0.3, 0.3), (-0.3, 0.3),
          (0.02, 0.8), (-1.2, 1.2), (-1.2, 1.2)]
SCALE = [0.5, 0.5, 0.3, 0.3, 0.15, 0.15, 0.1, 0.3, 0.3]


def _init(pop_dir, n_max, conc):
    man = json.loads((pop_dir / "manifest.json").read_text())
    G["kb"], G["npix"], G["n_max"], G["conc"] = man["kwargs_band"], man["image_shape"][0], n_max, conc
    G["truths"] = [json.loads(l) for l in open(pop_dir / "truth.jsonl")]
    G["images"] = np.load(pop_dir / "images_full_noisy.npy", mmap_mode="r")
    G["n_coef"] = int((n_max + 1) * (n_max + 2) / 2)


def _fitter(model_list, data):
    sim = SimAPI(num_pix=G["npix"], kwargs_single_band=G["kb"],
                 kwargs_model={"lens_model_list": model_list, "source_light_model_list": ["SHAPELETS"]})
    im = sim.image_model_class(kwargs_numerics={"supersampling_factor": 1})
    dc = sim.data_class
    dc.update_data(data)
    lin = ImageLinearFit(data_class=dc, psf_class=sim.psf_class, lens_model_class=im.LensModel,
                         source_model_class=im.SourceModel, kwargs_numerics={"supersampling_factor": 1})
    return sim, lin


def _kl(v, extra=None):
    kl = [{"theta_E": v[0], "gamma": v[1], "e1": v[2], "e2": v[3], "center_x": 0.0, "center_y": 0.0},
          {"gamma1": v[4], "gamma2": v[5], "ra_0": 0.0, "dec_0": 0.0}]
    return kl + ([extra] if extra else [])


def _ks(v):
    return [{"n_max": G["n_max"], "beta": v[6], "center_x": v[7], "center_y": v[8],
             "amp": np.zeros(G["n_coef"])}]


def _model(lin, v, extra=None):
    try:
        model, _, _, _ = lin.image_linear_solve(kwargs_lens=_kl(v, extra), kwargs_source=_ks(v))
    except Exception:
        return None
    return model


def sub_kwargs(log10_m, x, y, tau=20.0):
    Rs, aRs = LC.nfw_physical2angle(M=10 ** log10_m, c=float(G["conc"]))
    return {"Rs": Rs, "alpha_Rs": aRs, "r_trunc": tau * Rs, "center_x": x, "center_y": y}


def one_lens(args):
    i, fit_seed, maxiter = args
    t0 = time.time()
    truth = G["truths"][i]
    data = np.asarray(G["images"][i])
    lm = truth["lens_macro"]
    sim0, lin0 = _fitter(["EPL", "SHEAR"], data)
    noise = sim0.estimate_noise(data)
    rng = np.random.default_rng(fit_seed)
    # near-truth macro init, as the Sersic-source runs use; the source scale is generic because a
    # shapelet basis has no "true" beta to jitter around
    x0 = np.array([lm["theta_E"], lm["gamma"], lm["e1"], lm["e2"], lm["gamma1"], lm["gamma2"],
                   float(np.clip(truth["source"].get("R_sersic", 0.15), 0.05, 0.4)),
                   truth["source"]["x"], truth["source"]["y"]])
    jit = 1.0 + rng.uniform(-0.15, 0.15, size=x0.shape)
    jit[np.abs(x0) < 1e-6] = 1.0
    x0 = np.where(np.abs(x0) < 1e-6, x0 + rng.uniform(-0.02, 0.02, size=x0.shape), x0 * jit)
    x0 = np.clip(x0, [b[0] for b in BOUNDS], [b[1] for b in BOUNDS])

    def resid(v, extra=None, lin=lin0):
        model = _model(lin, v, extra)
        if model is None:
            return np.full(G["npix"] ** 2, 1e3)
        return ((data - model) / noise).ravel()

    # beta, the shapelet scale, is the one source parameter with no natural starting guess and a
    # multi-modal chi^2. Choosing it at the (jittered) initial macro does not work -- a very small
    # beta over-fits locally and traps the optimizer -- so we restart the whole non-linear fit from
    # three scales and keep the best converged solution.
    best_res, best_chi2 = None, np.inf
    for b0 in (0.06, 0.12, 0.25):
        xs = x0.copy(); xs[6] = b0
        try:
            r_ = least_squares(resid, xs, bounds=([b[0] for b in BOUNDS], [b[1] for b in BOUNDS]),
                               x_scale=SCALE, method="trf", max_nfev=maxiter * len(xs))
        except Exception:
            continue
        c_ = float(np.sum(r_.fun ** 2))
        if c_ < best_chi2:
            best_chi2, best_res = c_, r_
    res = best_res if best_res is not None else least_squares(
        resid, x0, bounds=([b[0] for b in BOUNDS], [b[1] for b in BOUNDS]),
        x_scale=SCALE, method="trf", max_nfev=maxiter * len(x0))
    v = res.x
    chi2_smooth = float(np.sum(res.fun ** 2))
    reliable = (chi2_smooth / G["npix"] ** 2) < CHI2_DOF_UNRELIABLE
    t_fit = time.time() - t0

    sim1, lin1 = _fitter(["EPL", "SHEAR", "TNFW"], data)
    theta_E = lm["theta_E"]
    radii = np.linspace(0.5, 1.4, 3) * theta_E
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    best = {"delta_chi2": -np.inf, "x": None, "y": None, "log10_m": None}
    t1 = time.time()
    for r in radii:
        for a in angles:
            x, y = r * np.cos(a), r * np.sin(a)
            for log10_m in (8.5, 9.5, 10.5):
                f = resid(v, sub_kwargs(log10_m, x, y), lin1)
                d = chi2_smooth - float(np.sum(f ** 2))
                if d > best["delta_chi2"]:
                    best = {"delta_chi2": float(d), "x": float(x), "y": float(y), "log10_m": float(log10_m)}
    return {"index": int(i), "has_subhalo": truth.get("subhalo") is not None,
            "log10_M200_true": (truth["subhalo"] or {}).get("log10_M200"),
            "concentration_true": (truth["subhalo"] or {}).get("concentration"),
            "has_multipole": truth.get("multipole") is not None,
            "chi2_smooth_per_dof": chi2_smooth / G["npix"] ** 2, "reliable_fit": reliable,
            "delta_chi2": best["delta_chi2"], "best_x": best["x"], "best_y": best["y"],
            "best_log10_m": best["log10_m"], "source": "shapelets", "n_max": G["n_max"],
            "beta_fitted": float(v[6]), "t_fit_s": t_fit, "t_scan_s": time.time() - t1}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--population", required=True)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--n-subsample", type=int, default=300)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--n-max", type=int, default=6)
    p.add_argument("--maxiter", type=int, default=40)
    p.add_argument("--concentration", type=float, default=15.0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    _init(ROOT / "data" / args.population, args.n_max, args.concentration)
    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(len(G["truths"]), size=min(args.n_subsample, len(G["truths"])), replace=False))[: args.n]
    jobs = [(int(i), int(args.seed) * 100000 + k, args.maxiter) for k, i in enumerate(idx)]
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    import multiprocessing as mp
    with mp.get_context("fork").Pool(args.workers) as pool, open(args.out / "scan_results.jsonl", "w") as f:
        for k, rec in enumerate(pool.imap_unordered(one_lens, jobs)):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if (k + 1) % 10 == 0:
                print(f"  {k+1}/{len(jobs)}  ({time.time()-t0:.0f}s)", flush=True)
    recs = sorted((json.loads(l) for l in open(args.out / "scan_results.jsonl")), key=lambda r: r["index"])
    with open(args.out / "scan_results.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    (args.out / "manifest.json").write_text(json.dumps({
        "population": args.population, "n": len(recs), "seed": args.seed, "source_model": "SHAPELETS",
        "n_max": args.n_max, "n_source_coefficients": G["n_coef"], "maxiter": args.maxiter,
        "concentration_assumed": args.concentration, "linear_source_solve": True,
        "n_unreliable_fits": sum(1 for r in recs if not r["reliable_fit"]),
        "elapsed_s": time.time() - t0, "workers": args.workers}, indent=2))
    print(f"wrote {len(recs)} shapelet scans to {args.out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
