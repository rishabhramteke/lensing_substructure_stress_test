"""Family A with the fit-then-scan shortcut removed: joint macro-model re-fit at every grid cell.

The frozen-macro-model shortcut was shown to cause the scan's mass bias
(`joint_refit_mass_test.py`). The referee's follow-up question is whether the same
shortcut also drives the scan's headline false-positive rate under the multipole
confounder and its 15% localization. This script answers it on the SAME lenses as
`results/baseline_a` (first `--n` indices of the identical 300-lens subsample, same
seeds 0/1/3): at each of the 3 radii x 8 angles grid positions the 13 macro+source
parameters AND log10 M are re-fitted jointly (initialised at the smooth fit, log10 M=9),
and the statistic is Delta chi^2 = chi^2_smooth - min_cell chi^2_joint. Position =
winning cell, mass = its fitted log10 M. Output format = run_baseline_a.py's, so the
existing evaluators run unchanged.

    OMP_NUM_THREADS=1 python scripts/run_baseline_a_joint.py --population multipole_m4_a3 --n 100 \
        --seed 3 --concentration 15 --workers 6 --out results/baseline_a_joint_c15/multipole_m4_a3
"""
import argparse, json, os, sys, time
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


def _init(pop_dir, conc):
    manifest = json.loads((pop_dir / "manifest.json").read_text())
    G["kwargs_band"], G["num_pix"] = manifest["kwargs_band"], manifest["image_shape"][0]
    G["truths"] = [json.loads(l) for l in open(pop_dir / "truth.jsonl")]
    G["images"] = np.load(pop_dir / "images_full_noisy.npy", mmap_mode="r")
    G["conc"] = conc
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
    truth = G["truths"][i]; data = np.asarray(G["images"][i]); theta_E = truth["lens_macro"]["theta_E"]
    rng_fit = np.random.default_rng(fit_seed)
    t1 = time.time()
    vec, chi2_smooth, _, noise_std = fit_smooth(data, truth, G["kwargs_band"], G["num_pix"], rng=rng_fit, maxiter=60)
    t_fit = time.time() - t1
    reliable = (chi2_smooth / G["num_pix"] ** 2) < CHI2_DOF_UNRELIABLE
    lo = [b[0] for b in BOUNDS] + [7.5]; hi = [b[1] for b in BOUNDS] + [11.5]
    radii = np.linspace(0.5, 1.4, 3) * theta_E; angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    best = {"delta_chi2": -np.inf, "x": None, "y": None, "log10_m": None, "nfev": 0}
    t2 = time.time()
    for r in radii:
        for a in angles:
            x, y = r * np.cos(a), r * np.sin(a)

            def resid(v):
                kl, ks = _unpack(v[:13])
                model = G["image_model"].image(kwargs_lens=kl + [sub_kwargs(v[13], x, y)], kwargs_source=ks)
                return ((data - model) / noise_std).ravel()

            res = least_squares(resid, np.concatenate([vec, [9.0]]), bounds=(lo, hi), x_scale=SCALE + [1.0], method="trf", max_nfev=40 * 14)
            chi2 = float(np.sum(res.fun ** 2)); delta = chi2_smooth - chi2
            if delta > best["delta_chi2"]:
                best = {"delta_chi2": float(delta), "x": float(x), "y": float(y), "log10_m": float(res.x[13]), "nfev": int(res.nfev)}
    return {"index": int(i), "has_subhalo": truth.get("subhalo") is not None,
            "log10_M200_true": (truth["subhalo"] or {}).get("log10_M200"),
            "concentration_true": (truth["subhalo"] or {}).get("concentration"),
            "has_multipole": truth.get("multipole") is not None,
            "chi2_smooth_per_dof": chi2_smooth / G["num_pix"] ** 2, "reliable_fit": reliable,
            "delta_chi2": best["delta_chi2"], "best_x": best["x"], "best_y": best["y"], "best_log10_m": best["log10_m"],
            "t_fit_s": t_fit, "t_scan_s": time.time() - t2, "joint_refit": True}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--population", required=True)
    p.add_argument("--n", type=int, default=100, help="first n lenses of the SAME 300-lens subsample run_baseline_a.py drew with --seed")
    p.add_argument("--n-subsample", type=int, default=300)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--concentration", default="15")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    pop_dir = ROOT / "data" / args.population
    conc = args.concentration if args.concentration in ("cm",) else float(args.concentration)
    _init(pop_dir, conc)
    n_pop = len(G["truths"])
    rng_sel = np.random.default_rng(args.seed)
    idx = np.sort(rng_sel.choice(n_pop, size=min(args.n_subsample, n_pop), replace=False))[: args.n]
    # per-lens fit seeds: independent streams so workers are deterministic regardless of scheduling
    jobs = [(int(i), int(args.seed) * 100000 + k) for k, i in enumerate(idx)]
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
        "population": args.population, "n": len(recs), "seed": args.seed, "subsample_n": args.n_subsample,
        "concentration_assumed": args.concentration, "joint_refit_per_cell": True, "grid": "3 radii x 8 angles, mass free in [7.5, 11.5]",
        "chi2_dof_unreliable_threshold": CHI2_DOF_UNRELIABLE, "n_unreliable_fits": sum(1 for r in recs if not r["reliable_fit"]),
        "elapsed_s": time.time() - t0, "workers": args.workers}, indent=2))
    print(f"wrote {len(recs)} joint scans to {args.out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
