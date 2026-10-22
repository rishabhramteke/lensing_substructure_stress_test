"""Complete the macro-basin control for lenses whose per-lens time guard cut it short.

run_baseline_a_joint.py stops the basin loop when a lens exceeds --lens-timeout, so for a truncated
lens the best cell may never have had its perturber removed and its macro-model re-optimised. If
that cell had found a better SMOOTH optimum, the uncorrected statistic credits the improvement to the
perturber (one clean lens of the full control came out at Delta chi^2 = 3.5e4 this way). This script
re-runs the basin re-fit for the K best cells of every such lens and recomputes the corrected
statistic in place, keeping the pre-repair value as `delta_chi2_before_repair`.

    nice -n 10 python scripts/repair_joint_basin.py --roots results/baseline_a_joint_full results/baseline_a_joint_los --workers 2
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from baseline_a.fit import _sim, _unpack3, n_lens_light_components, BOUNDS, SCALE, BOUNDS_LL, SCALE_LL  # noqa: E402

G = {}


def needs_repair(rec, k):
    cells = sorted(rec["cells"], key=lambda c: c["chi2_joint"])[:k]
    return [c for c in cells if "chi2_smooth_from_joint_macro" not in c]


def repair(args):
    rec, k = args
    data = np.asarray(G["images"][rec["index"]])
    noise_std = G["sim_s"].estimate_noise(data)
    im_s = G["im_s"]
    todo = needs_repair(rec, k)
    n_par = rec["n_params_macro"]; n_ll = rec.get("lens_light_components_fitted", 0)
    bounds = list(BOUNDS) + list(BOUNDS_LL) * n_ll; scale = list(SCALE) + list(SCALE_LL) * n_ll
    lo = [b[0] for b in bounds]; hi = [b[1] for b in bounds]

    def resid_smooth(v):
        kl, ks, kll = _unpack3(v)
        return ((data - im_s.image(kwargs_lens=kl, kwargs_source=ks, kwargs_lens_light=kll)) / noise_std).ravel()

    t0 = time.time()
    for c in todo:
        x0 = np.clip(np.array(c["vec_macro"] if "vec_macro" in c else c["vec13"]), lo, hi)
        rb = least_squares(resid_smooth, x0, bounds=(lo, hi), x_scale=scale, method="trf", max_nfev=40 * (n_par + 1))
        c["chi2_smooth_from_joint_macro"] = float(np.sum(rb.fun ** 2)); c["basin_nfev"] = int(rb.nfev); c["repaired"] = True
    # recompute the corrected statistic exactly as run_baseline_a_joint.py does
    chi2_smooth = rec["chi2_smooth_per_dof"] * G["num_pix"] ** 2
    chi2_polished = chi2_smooth - rec["chi2_polish_gain"]
    refits = [c["chi2_smooth_from_joint_macro"] for c in rec["cells"] if "chi2_smooth_from_joint_macro" in c]
    chi2_smooth_basin = min([chi2_polished] + refits)
    best = min(rec["cells"], key=lambda c: c["chi2_joint"])
    rec.setdefault("delta_chi2_before_repair", rec["delta_chi2"])
    rec["delta_chi2"] = float(chi2_smooth_basin - best["chi2_joint"])
    rec["basin_gain"] = float(chi2_smooth - chi2_smooth_basin)
    rec["basin_gain_from_best_cell"] = float(chi2_smooth - best.get("chi2_smooth_from_joint_macro", chi2_smooth))
    rec["chi2_smooth_basin_per_dof"] = chi2_smooth_basin / G["num_pix"] ** 2
    rec["n_basin_refits"] = len(refits)
    rec["repair"] = {"n_cells_refit": len(todo), "t_s": time.time() - t0, "k": k}
    return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--roots", nargs="+", required=True)
    p.add_argument("--k", type=int, default=3, help="best cells (by joint chi2) that must carry a basin re-fit")
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()
    import multiprocessing as mp
    for root in args.roots:
        for popdir in sorted((ROOT / root).glob("*/")):
            if not (popdir / "manifest.json").exists():
                continue
            man = json.loads((popdir / "manifest.json").read_text())
            pop = man["population"]; pop_dir = ROOT / "data" / pop
            recs = [json.loads(l) for l in open(popdir / "scan_results.jsonl")]
            todo = [r for r in recs if needs_repair(r, args.k)]
            if not todo:
                print(f"{popdir}: nothing to repair"); continue
            dman = json.loads((pop_dir / "manifest.json").read_text())
            G["num_pix"] = dman["image_shape"][0]
            noisy = pop_dir / "images_full_noisy.npy"
            G["images"] = np.load(noisy if noisy.exists() else pop_dir / "images_full_noiseless.npy", mmap_mode="r")
            n_ll = todo[0].get("lens_light_components_fitted", 0)
            G["sim_s"] = _sim(dman["kwargs_band"], G["num_pix"], ["EPL", "SHEAR"], lens_light=n_ll)
            G["im_s"] = G["sim_s"].image_model_class(kwargs_numerics={"supersampling_factor": 1})
            t0 = time.time()
            with mp.get_context("fork").Pool(args.workers) as pool:
                fixed = {r["index"]: r for r in pool.imap_unordered(repair, [(r, args.k) for r in todo])}
            out = [fixed.get(r["index"], r) for r in recs]
            changed = sum(1 for r in out if "delta_chi2_before_repair" in r and abs(r["delta_chi2"] - r["delta_chi2_before_repair"]) > 1e-6)
            with open(popdir / "scan_results.jsonl", "w") as f:
                for r in out:
                    f.write(json.dumps(r) + "\n")
            man["basin_repair"] = {"n_lenses": len(todo), "n_changed": changed, "k_best_cells": args.k, "date": time.strftime("%Y-%m-%d")}
            (popdir / "manifest.json").write_text(json.dumps(man, indent=2))
            big = [(r["index"], round(r["delta_chi2_before_repair"], 1), round(r["delta_chi2"], 1)) for r in out if "delta_chi2_before_repair" in r and abs(r["delta_chi2"] - r["delta_chi2_before_repair"]) > 20]
            print(f"{popdir}: repaired {len(todo)} lenses, {changed} statistics changed ({time.time()-t0:.0f}s); changes >20: {big[:8]}", flush=True)


if __name__ == "__main__":
    main()
