"""Family A with the fit-then-scan shortcut removed: joint macro-model re-fit at every grid cell,
with the macro-basin control a referee asked for (round 4, 2026-09-12).

At each of the 3 radii x 8 angles grid positions the macro+source parameters AND log10 M are
re-fitted jointly (initialised at the smooth fit, log10 M = 9). The raw statistic is
Delta chi^2 = chi^2_smooth - min_cell chi^2_joint. Two controls guard it:

  * polish control (round 3): the smooth fit is continued from its own solution with the joint
    fits' total optimizer budget. This only tests local convergence.
  * macro-basin control (round 4): every cell's joint macro solution is taken, the perturber is
    REMOVED, and the macro parameters are re-optimised from there. If any of those smooth
    re-fits beats the original smooth chi^2, the joint fit had found a better macro basin, not a
    perturber, and that basin is what the smooth model is credited with. The reported
    `delta_chi2` is therefore  min(chi^2_smooth over all starts) - min_cell chi^2_joint.

Every cell's solution is stored (`cells`), so the control can be re-evaluated without re-fitting.
Output format = run_baseline_a.py's, so the existing evaluators run unchanged.

2026-09-15 (revision for the referee's "run the joint scan throughout"): the macro-model may
carry lens light (`--lens-light-components 1|2`, as run_baseline_a.py), so the joint scan runs on
every stress population, including the lens-light, line-of-sight, shallow and decoy ones; the
parameter vector length is whatever fit_smooth returns (13 + 5 per lens-light component).

    OMP_NUM_THREADS=1 python scripts/run_baseline_a_joint.py --population multipole_m4_a3 --n 100 \
        --seed 3 --concentration 15 --workers 4 --skip-indices 32 --out results/baseline_a_joint_basin_c15/multipole_m4_a3
    OMP_NUM_THREADS=1 python scripts/run_baseline_a_joint.py --population no_subhalo --n 1000 --n-subsample 1000 \
        --seed 0 --workers 9 --out results/baseline_a_joint_full/no_subhalo
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lenstronomy.Cosmo.lens_cosmo import LensCosmo  # noqa: E402
from baseline_a import fit as FIT  # noqa: E402
from baseline_a.fit import fit_smooth, _sim, _unpack3, n_lens_light_components, has_multipole, set_multipole_orders, BOUNDS, SCALE, BOUNDS_LL, SCALE_LL, BOUNDS_MP, SCALE_MP  # noqa: E402
from lensing.concentration import concentration_dutton_maccio14  # noqa: E402

CHI2_DOF_UNRELIABLE = 10.0
LC = LensCosmo(z_lens=0.5, z_source=1.0)
G = {"nfev_mult": 40, "mass_max": 11.5}


def _init(pop_dir, conc, lens_timeout, basin, lens_light_components, free_position=False, position_box=0.5, mp_orders=False, restarts=1, blind=False):
    G["free_position"], G["position_box"] = free_position, position_box
    G["blind"] = bool(blind)
    # 2026-09-18 (referee round 19): the smooth macro-model may carry free multipole terms (a_m, phi_m per order), as
    # post-Lange+2025 pipelines do; the frozen scan had this (run_baseline_a.py --macro-multipole), the joint scan did not.
    G["mp_orders"] = tuple(int(m) for m in mp_orders) if mp_orders else False
    if G["mp_orders"]:
        set_multipole_orders(G["mp_orders"])
    G["n_mp"] = len(G["mp_orders"]) if G["mp_orders"] else 0
    G["restarts"] = int(restarts)
    manifest = json.loads((pop_dir / "manifest.json").read_text())
    G["kwargs_band"], G["num_pix"] = manifest["kwargs_band"], manifest["image_shape"][0]
    G["truths"] = [json.loads(l) for l in open(pop_dir / "truth.jsonl")]
    noisy = pop_dir / "images_full_noisy.npy"
    G["images"] = np.load(noisy if noisy.exists() else pop_dir / "images_full_noiseless.npy", mmap_mode="r")
    G["conc"], G["lens_timeout"], G["basin"] = conc, lens_timeout, basin
    has_ll = any(t.get("lens_light") for t in G["truths"])
    G["n_ll"] = int(lens_light_components) if has_ll else 0
    sim = _sim(G["kwargs_band"], G["num_pix"], ["EPL", "SHEAR", "TNFW"], lens_light=G["n_ll"], multipole=G["n_mp"])
    G["image_model"] = sim.image_model_class(kwargs_numerics={"supersampling_factor": FIT.FIT_SUPERSAMPLING})
    sim_s = _sim(G["kwargs_band"], G["num_pix"], ["EPL", "SHEAR"], lens_light=G["n_ll"], multipole=G["n_mp"])
    G["image_model_smooth"] = sim_s.image_model_class(kwargs_numerics={"supersampling_factor": FIT.FIT_SUPERSAMPLING})


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
    # --blind (referee round 20): the single smooth start is the data-only initialisation instead of the near-truth one
    vec, chi2_smooth, _, noise_std = fit_smooth(data, truth, G["kwargs_band"], G["num_pix"], rng=rng_fit, maxiter=60, blind=G["blind"],
                                                lens_light_components=G["n_ll"], macro_multipole=G["mp_orders"])
    n_starts = 1
    if G["restarts"] > 1:
        # multi-start smooth fit (referee round 19): further near-truth starts with a wider jitter and one data-only
        # start; the best chi^2 is kept, so a lens is gated only when no start fits
        for k in range(1, G["restarts"]):
            rng_k = np.random.default_rng(fit_seed + 7919 * k)
            v_k, c_k, _, _ = fit_smooth(data, truth, G["kwargs_band"], G["num_pix"], jitter_frac=0.3, rng=rng_k, maxiter=60,
                                        lens_light_components=G["n_ll"], macro_multipole=G["mp_orders"])
            n_starts += 1
            if c_k < chi2_smooth:
                vec, chi2_smooth = v_k, c_k
        v_b, c_b, _, _ = fit_smooth(data, truth, G["kwargs_band"], G["num_pix"], rng=rng_fit, maxiter=60, blind=True,
                                    lens_light_components=G["n_ll"], macro_multipole=G["mp_orders"])
        n_starts += 1
        if c_b < chi2_smooth:
            vec, chi2_smooth = v_b, c_b
    t_fit = time.time() - t_start
    reliable = (chi2_smooth / G["num_pix"] ** 2) < CHI2_DOF_UNRELIABLE
    n_par = len(vec); n_ll = n_lens_light_components(vec); n_mp = has_multipole(vec)
    bounds = list(BOUNDS) + list(BOUNDS_LL) * n_ll + list(BOUNDS_MP) * n_mp; scale = list(SCALE) + list(SCALE_LL) * n_ll + list(SCALE_MP) * n_mp
    lo_m = [b[0] for b in bounds]; hi_m = [b[1] for b in bounds]
    im_s = G["image_model_smooth"]

    def resid_smooth(v):
        kl, ks, kll = _unpack3(v)
        return ((data - im_s.image(kwargs_lens=kl, kwargs_source=ks, kwargs_lens_light=kll)) / noise_std).ravel()

    # ---- polish control (round 3): same total budget as the 24 joint fits, from the smooth solution
    r0 = least_squares(resid_smooth, vec, bounds=(lo_m, hi_m), x_scale=scale, method="trf", max_nfev=24 * G["nfev_mult"] * (n_par + 1))
    chi2_polished = min(chi2_smooth, float(np.sum(r0.fun ** 2))); polish_nfev = int(r0.nfev)

    # ---- joint scan: 24 cells, macro (+ lens light) + mass free, every solution kept; with --free-position the
    #      perturber's (x, y) are free too, within `position_box` arcsec of the node, so the 24 nodes are starting
    #      points of a continuous position optimisation and the reported position is a measurement, not a hypothesis
    fp, box = G["free_position"], G["position_box"]
    radii = np.linspace(0.5, 1.4, 3) * theta_E; angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    cells, truncated = [], False
    t2 = time.time()
    for r in radii:
        for a in angles:
            if cells and time.time() - t_start > G["lens_timeout"]:
                truncated = True; break
            x, y = r * np.cos(a), r * np.sin(a)
            if fp:
                lo = lo_m + [7.5, max(x - box, -2.4), max(y - box, -2.4)]; hi = hi_m + [G["mass_max"], min(x + box, 2.4), min(y + box, 2.4)]
                x0 = np.concatenate([vec, [G.get("mass_init", 9.0), x, y]]); xs = scale + [1.0, 0.1, 0.1]
            else:
                lo = lo_m + [7.5]; hi = hi_m + [G["mass_max"]]; x0 = np.concatenate([vec, [G.get("mass_init", 9.0)]]); xs = scale + [1.0]

            def resid(v):
                kl, ks, kll = _unpack3(v[:n_par])
                px, py = (v[n_par + 1], v[n_par + 2]) if fp else (x, y)
                model = G["image_model"].image(kwargs_lens=kl + [sub_kwargs(v[n_par], px, py)], kwargs_source=ks, kwargs_lens_light=kll)
                return ((data - model) / noise_std).ravel()

            res = least_squares(resid, x0, bounds=(lo, hi), x_scale=xs, method="trf", max_nfev=G["nfev_mult"] * len(x0))
            cells.append({"x": float(res.x[n_par + 1]) if fp else float(x), "y": float(res.x[n_par + 2]) if fp else float(y),
                          "node_x": float(x), "node_y": float(y), "chi2_joint": float(np.sum(res.fun ** 2)), "log10_m": float(res.x[n_par]),
                          "vec_macro": [float(v) for v in res.x[:n_par]], "nfev": int(res.nfev)})
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
            x0 = np.clip(np.array(c["vec_macro"]), lo_m, hi_m)
            rb = least_squares(resid_smooth, x0, bounds=(lo_m, hi_m), x_scale=scale, method="trf", max_nfev=G["nfev_mult"] * (n_par + 1))
            c["chi2_smooth_from_joint_macro"] = float(np.sum(rb.fun ** 2)); c["basin_nfev"] = int(rb.nfev); n_basin += 1
        refits = [c["chi2_smooth_from_joint_macro"] for c in cells if "chi2_smooth_from_joint_macro" in c]
        if refits:
            chi2_smooth_basin = min(chi2_polished, min(refits))
    basin_gain = chi2_smooth - chi2_smooth_basin                       # > 0 iff a joint macro solution led to a better SMOOTH optimum
    delta_basin = chi2_smooth_basin - best["chi2_joint"]               # the perturber's own share of the improvement
    sub = truth.get("subhalo") or {}
    los = truth.get("los_halo") or {}
    return {"index": int(i), "has_subhalo": truth.get("subhalo") is not None,
            "log10_M200_true": sub.get("log10_M200"),
            "concentration_true": sub.get("concentration"),
            "has_multipole": truth.get("multipole") is not None,
            "has_los_halo": truth.get("los_halo") is not None, "log10_M200_los": los.get("log10_M200"),
            "has_lens_light": truth.get("lens_light") is not None, "lens_light_components_fitted": n_ll, "n_params_macro": n_par,
            "macro_multipole_orders": list(G["mp_orders"]) if G["mp_orders"] else None, "n_smooth_starts": n_starts, "blind_initialization": G["blind"],
            "chi2_smooth_per_dof": chi2_smooth / G["num_pix"] ** 2, "reliable_fit": reliable,
            # delta_chi2 = basin-corrected statistic (what the paper reports); the two uncorrected forms are kept alongside
            "delta_chi2": float(delta_basin), "delta_chi2_vs_unpolished_smooth": float(delta_raw),
            "delta_chi2_vs_polished_smooth": float(chi2_polished - best["chi2_joint"]),
            "chi2_polish_gain": float(chi2_smooth - chi2_polished), "basin_gain": float(basin_gain),
            "basin_gain_from_best_cell": float(chi2_smooth - best.get("chi2_smooth_from_joint_macro", chi2_smooth)),
            "chi2_smooth_basin_per_dof": chi2_smooth_basin / G["num_pix"] ** 2,
            "best_x": best["x"], "best_y": best["y"], "best_log10_m": best["log10_m"], "free_position": bool(fp),
            "n_cells": len(cells), "n_basin_refits": n_basin, "truncated": truncated,
            "t_fit_s": t_fit, "t_scan_s": t_scan, "t_basin_s": time.time() - t3, "joint_refit": True,
            "null_control_polish_nfev": polish_nfev, "cells": cells}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--population", required=True)
    p.add_argument("--n", type=int, default=100, help="first n lenses of the SAME subsample run_baseline_a.py drew with --seed and --n-subsample")
    p.add_argument("--n-subsample", type=int, default=300, help="size of the subsample the frozen scan drew (300 for the seed sets, 1000 for a full population)")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--concentration", default="15")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--skip-indices", default="", help="comma-separated lens indices to leave out (pathological fits)")
    p.add_argument("--lens-timeout", type=float, default=900.0, help="wall-clock budget per lens; the cell loop stops (and the record says `truncated`) beyond it")
    p.add_argument("--no-basin", action="store_true", help="skip the macro-basin control (round-3 behaviour)")
    p.add_argument("--lens-light-components", type=int, default=1, choices=[1, 2], help="Sersic components fitted for lens light when the population has it (as run_baseline_a.py)")
    p.add_argument("--free-position", action="store_true", help="let the perturber's (x, y) vary within --position-box arcsec of each grid node")
    p.add_argument("--position-box", type=float, default=0.5)
    p.add_argument("--macro-multipole", action="store_true", help="free multipole terms (a_m, phi_m per order) in the smooth AND joint macro-models, as post-Lange+2025 pipelines do")
    p.add_argument("--macro-multipole-orders", default="4", help="comma-separated orders carried by --macro-multipole (default 4; '3,4' tests the cost of the extra freedom)")
    p.add_argument("--restarts", type=int, default=1, help="smooth-fit starts (1 = the single near-truth start used throughout; >1 adds wider-jitter starts and one data-only start, best chi^2 kept)")
    p.add_argument("--blind", action="store_true", help="initialise the (first) smooth fit from the data alone, fit._blind_init_vec, instead of near the truth")
    p.add_argument("--indices", default="", help="comma-separated lens indices to run INSTEAD of the seeded subsample (rescue pass on gated lenses)")
    p.add_argument("--mass-max", type=float, default=11.5, help="upper bound of the perturber's log10 M200 prior (11.5 throughout; 10.5 is the tight-prior control, referee round 25)")
    p.add_argument("--nfev-mult", type=int, default=40, help="function-evaluation cap per cell fit and basin re-fit, in units of (n_par+1) (40 throughout the paper; 80 is the cap-doubling control)")
    p.add_argument("--mass-init", type=float, default=9.0, help="starting log10 M200 of every cell fit (9 throughout; 11 is the high-mass rescue, referee round 28)")
    p.add_argument("--free-centre", action="store_true", help="fit the lens centre as two further macro parameters (referee round 26)")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.free_centre:
        FIT.set_free_centre(True)
    pop_dir = ROOT / "data" / args.population
    conc = args.concentration if args.concentration in ("cm",) else float(args.concentration)
    mp_orders = tuple(int(m) for m in args.macro_multipole_orders.split(",")) if args.macro_multipole else False
    G["nfev_mult"] = int(args.nfev_mult); G["mass_max"] = float(args.mass_max); G["mass_init"] = float(args.mass_init)
    _init(pop_dir, conc, args.lens_timeout, not args.no_basin, args.lens_light_components, args.free_position, args.position_box, mp_orders, args.restarts, args.blind)
    n_pop = len(G["truths"])
    rng_sel = np.random.default_rng(args.seed)
    n_sub = min(args.n_subsample, n_pop)
    idx = np.sort(rng_sel.choice(n_pop, size=n_sub, replace=False)) if n_sub < n_pop else np.arange(n_pop)
    idx = idx[: args.n]
    if args.indices.strip():
        idx = np.array(sorted({int(s) for s in args.indices.split(",") if s.strip()}))
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
        "concentration_assumed": args.concentration, "joint_refit_per_cell": True, "grid": f"3 radii x 8 angles, mass free in [7.5, {args.mass_max:g}]", "mass_max": args.mass_max,
        "lens_light_components": G["n_ll"], "free_position": args.free_position, "position_box_arcsec": args.position_box if args.free_position else None,
        "macro_multipole": bool(mp_orders), "macro_multipole_orders": list(mp_orders) if mp_orders else None, "smooth_restarts": args.restarts, "nfev_mult": args.nfev_mult, "blind_initialization": args.blind, "free_centre": args.free_centre, "mass_init": args.mass_init, "fit_supersampling": FIT.FIT_SUPERSAMPLING,
        "explicit_indices": sorted(int(i) for i in idx) if args.indices.strip() else None,
        "controls": {"polish": "smooth fit continued from its solution with max_nfev 24*40*(n_par+1)",
                     "macro_basin": (not args.no_basin) and "every cell's joint macro solution, perturber removed, macro params re-optimised (max_nfev 40*(n_par+1)); delta_chi2 is measured against the best smooth chi2 over all starts"},
        "lens_timeout_s": args.lens_timeout, "n_truncated": sum(1 for r in recs if r["truncated"]),
        "chi2_dof_unreliable_threshold": CHI2_DOF_UNRELIABLE, "n_unreliable_fits": sum(1 for r in recs if not r["reliable_fit"]),
        "elapsed_s": time.time() - t0, "workers": args.workers}, indent=2))
    print(f"wrote {len(recs)} joint scans to {args.out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
