"""Family A with the fit-then-scan shortcut removed: joint macro-model re-fit at every grid cell,
with the macro-basin control a referee asked for (round 4, 2026-09-12).

At each of the 3 radii x 8 angles grid positions the 13 macro+source parameters AND log10 M are
re-fitted jointly (initialised at the smooth fit, log10 M = 9). The raw statistic is
Delta chi^2 = chi^2_smooth - min_cell chi^2_joint. Two controls guard it:

  * polish control (round 3): the smooth fit is continued from its own solution with the joint
    fits' total optimizer budget. This only tests local convergence.
  * macro-basin control (round 4): every cell's joint macro solution is taken, the perturber is
    REMOVED, and the 13 macro parameters are re-optimised from there. If any of those smooth
    re-fits beats the original smooth chi^2, the joint fit had found a better macro basin, not a
    perturber, and that basin is what the smooth model is credited with. The reported
    `delta_chi2` is therefore  min(chi^2_smooth over all starts) - min_cell chi^2_joint.

Every cell's solution is stored (`cells`), so the control can be re-evaluated without re-fitting.
Output format = run_baseline_a.py's, so the existing evaluators run unchanged.

    OMP_NUM_THREADS=1 python scripts/run_baseline_a_joint.py --population multipole_m4_a3 --n 100 \
        --seed 3 --concentration 15 --workers 4 --skip-indices 32 --out results/baseline_a_joint_basin_c15/multipole_m4_a3
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lenstronomy.Cosmo.lens_cosmo import LensCosmo  # noqa: E402
from baseline_a.fit import fit_smooth, _sim, _unpack, BOUNDS, SCALE  # noqa: E402
from lensing.concentration import concentration_dutton_maccio14  # noqa: E402

CHI2_DOF_UNRELIABLE = 10.0
LC = LensCosmo(z_lens=0.5, z_source=1.0)
G = {}


def _init(pop_dir, conc, lens_timeout, basin):
    manifest = json.loads((pop_dir / "manifest.json").read_text())
    G["kwargs_band"], G["num_pix"] = manifest["kwargs_band"], manifest["image_shape"][0]
    G["truths"] = [json.loads(l) for l in open(pop_dir / "truth.jsonl")]
    G["images"] = np.load(pop_dir / "images_full_noisy.npy", mmap_mode="r")
    G["conc"], G["lens_timeout"], G["basin"] = conc, lens_timeout, basin
    sim = _sim(G["kwargs_band"], G["num_pix"], ["EPL", "SHEAR", "TNFW"])
    G["image_model"] = sim.image_model_class(kwargs_numerics={"supersampling_factor": 1})


def c_of(log10_m):
    c = G["conc"]
    return float(concentration_dutton_maccio14(np.array([log10_m]), rng=None, z=0.5, scatter_dex=0.0)[0]) if c == "cm" else float(c)


def sub_kwargs(log10_m, x, y, tau=20.0):
    Rs, alpha_Rs = LC.nfw_physical2angle(M=10 ** log10_m, c=c_of(log10_m))
    return {"Rs": Rs, "alpha_Rs": alpha_Rs, "r_trunc": tau * Rs, "center_x": x, "center_y": y}


def one_lens(args):
    i, fit_seed = args
    t_start = time.time()
    truth = G["truths"][i]; data = np.asarray(G["images"][i]); theta_E = truth["lens_macro"]["theta_E"]
    rng_fit = np.random.default_rng(fit_seed)
    vec, chi2_smooth, _, noise_std = fit_smooth(data, truth, G["kwargs_band"], G["num_pix"], rng=rng_fit, maxiter=60)
    t_fit = time.time() - t_start
    reliable = (chi2_smooth / G["num_pix"] ** 2) < CHI2_DOF_UNRELIABLE
    lo13 = [b[0] for b in BOUNDS]; hi13 = [b[1] for b in BOUNDS]
    sim_s = _sim(G["kwargs_band"], G["num_pix"], ["EPL", "SHEAR"])
    im_s = sim_s.image_model_class(kwargs_numerics={"supersampling_factor": 1})

    def resid_smooth(v):
        kl, ks = _unpack(v)
        return ((data - im_s.image(kwargs_lens=kl, kwargs_source=ks)) / noise_std).ravel()

    # ---- polish control (round 3): same total budget as the 24 joint fits, from the smooth solution
    r0 = least_squares(resid_smooth, vec, bounds=(lo13, hi13), x_scale=SCALE, method="trf", max_nfev=24 * 40 * 14)
    chi2_polished = min(chi2_smooth, float(np.sum(r0.fun ** 2))); polish_nfev = int(r0.nfev)

    # ---- joint scan: 24 cells, macro + mass free, every solution kept
    lo = lo13 + [7.5]; hi = hi13 + [11.5]
    radii = np.linspace(0.5, 1.4, 3) * theta_E; angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    cells, truncated = [], False
    t2 = time.time()
    for r in radii:
        for a in angles:
            if cells and time.time() - t_start > G["lens_timeout"]:
                truncated = True; break
            x, y = r * np.cos(a), r * np.sin(a)

            def resid(v):
                kl, ks = _unpack(v[:13])
                model = G["image_model"].image(kwargs_lens=kl + [sub_kwargs(v[13], x, y)], kwargs_source=ks)
                return ((data - model) / noise_std).ravel()

            res = least_squares(resid, np.concatenate([vec, [9.0]]), bounds=(lo, hi), x_scale=SCALE + [1.0], method="trf", max_nfev=40 * 14)
            cells.append({"x": float(x), "y": float(y), "chi2_joint": float(np.sum(res.fun ** 2)), "log10_m": float(res.x[13]),
                          "vec13": [float(v) for v in res.x[:13]], "nfev": int(res.nfev)})
        if truncated:
            break
    t_scan = time.time() - t2
    best = min(cells, key=lambda c: c["chi2_joint"])
    delta_raw = chi2_smooth - best["chi2_joint"]

    # ---- macro-basin control (round 4): perturber removed, macro re-optimised from every joint solution
    t3 = time.time(); n_basin = 0
    chi2_smooth_basin = chi2_polished
    if G["basin"]:
        for c in cells:
            if n_basin and time.time() - t_start > G["lens_timeout"]:
                truncated = True; break
            x0 = np.clip(np.array(c["vec13"]), lo13, hi13)
            rb = least_squares(resid_smooth, x0, bounds=(lo13, hi13), x_scale=SCALE, method="trf", max_nfev=40 * 14)
            c["chi2_smooth_from_joint_macro"] = float(np.sum(rb.fun ** 2)); c["basin_nfev"] = int(rb.nfev); n_basin += 1
        refits = [c["chi2_smooth_from_joint_macro"] for c in cells if "chi2_smooth_from_joint_macro" in c]
        if refits:
            chi2_smooth_basin = min(chi2_polished, min(refits))
    basin_gain = chi2_smooth - chi2_smooth_basin                       # > 0 iff a joint macro solution led to a better SMOOTH optimum
    delta_basin = chi2_smooth_basin - best["chi2_joint"]               # the perturber's own share of the improvement
    return {"index": int(i), "has_subhalo": truth.get("subhalo") is not None,
            "log10_M200_true": (truth["subhalo"] or {}).get("log10_M200"),
            "concentration_true": (truth["subhalo"] or {}).get("concentration"),
            "has_multipole": truth.get("multipole") is not None,
            "chi2_smooth_per_dof": chi2_smooth / G["num_pix"] ** 2, "reliable_fit": reliable,
            # delta_chi2 = basin-corrected statistic (what the paper reports); the two uncorrected forms are kept alongside
            "delta_chi2": float(delta_basin), "delta_chi2_vs_unpolished_smooth": float(delta_raw),
            "delta_chi2_vs_polished_smooth": float(chi2_polished - best["chi2_joint"]),
            "chi2_polish_gain": float(chi2_smooth - chi2_polished), "basin_gain": float(basin_gain),
            "basin_gain_from_best_cell": float(chi2_smooth - best.get("chi2_smooth_from_joint_macro", chi2_smooth)),
            "chi2_smooth_basin_per_dof": chi2_smooth_basin / G["num_pix"] ** 2,
            "best_x": best["x"], "best_y": best["y"], "best_log10_m": best["log10_m"],
            "n_cells": len(cells), "n_basin_refits": n_basin, "truncated": truncated,
            "t_fit_s": t_fit, "t_scan_s": t_scan, "t_basin_s": time.time() - t3, "joint_refit": True,
            "null_control_polish_nfev": polish_nfev, "cells": cells}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--population", required=True)
    p.add_argument("--n", type=int, default=100, help="first n lenses of the SAME 300-lens subsample run_baseline_a.py drew with --seed")
    p.add_argument("--n-subsample", type=int, default=300)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--concentration", default="15")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--skip-indices", default="", help="comma-separated lens indices to leave out (pathological fits)")
    p.add_argument("--lens-timeout", type=float, default=900.0, help="wall-clock budget per lens; the cell loop stops (and the record says `truncated`) beyond it")
    p.add_argument("--no-basin", action="store_true", help="skip the macro-basin control (round-3 behaviour)")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    pop_dir = ROOT / "data" / args.population
    conc = args.concentration if args.concentration in ("cm",) else float(args.concentration)
    _init(pop_dir, conc, args.lens_timeout, not args.no_basin)
    n_pop = len(G["truths"])
    rng_sel = np.random.default_rng(args.seed)
    idx = np.sort(rng_sel.choice(n_pop, size=min(args.n_subsample, n_pop), replace=False))[: args.n]
    skip = {int(s) for s in args.skip_indices.split(",") if s.strip()}
    jobs = [(int(i), int(args.seed) * 100000 + k) for k, i in enumerate(idx) if int(i) not in skip]
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    with ctx.Pool(args.workers) as pool, open(args.out / "scan_results.jsonl", "w") as f:
        for k, rec in enumerate(pool.imap_unordered(one_lens, jobs)):
            f.write(json.dumps(rec) + "\n"); f.flush()
            if (k + 1) % 10 == 0:
                print(f"  {k+1}/{len(jobs)}  ({time.time()-t0:.0f}s)", flush=True)
    recs = [json.loads(l) for l in open(args.out / "scan_results.jsonl")]
    recs.sort(key=lambda r: r["index"])
    with open(args.out / "scan_results.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    (args.out / "manifest.json").write_text(json.dumps({
        "population": args.population, "n": len(recs), "seed": args.seed, "subsample_n": args.n_subsample, "skipped_indices": sorted(skip),
        "concentration_assumed": args.concentration, "joint_refit_per_cell": True, "grid": "3 radii x 8 angles, mass free in [7.5, 11.5]",
        "controls": {"polish": "smooth fit continued from its solution with max_nfev 24*40*14",
                     "macro_basin": (not args.no_basin) and "every cell's joint macro solution, perturber removed, 13 params re-optimised (max_nfev 40*14); delta_chi2 is measured against the best smooth chi2 over all starts"},
        "lens_timeout_s": args.lens_timeout, "n_truncated": sum(1 for r in recs if r["truncated"]),
        "chi2_dof_unreliable_threshold": CHI2_DOF_UNRELIABLE, "n_unreliable_fits": sum(1 for r in recs if not r["reliable_fit"]),
        "elapsed_s": time.time() - t0, "workers": args.workers}, indent=2))
    print(f"wrote {len(recs)} joint scans to {args.out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
