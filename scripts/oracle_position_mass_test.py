"""Oracle-position mass test for Family A (added 2026-09-11).

Question: is the scan's systematic mass underestimate caused by its fixed concentration
(the manuscript's first explanation) or by its coarse position grid? Running the scan at
matched c=60 and at the Dutton & Maccio (2014) c-M relation left the bias intact, so the
concentration explanation is out. This script isolates the position grid: for every
subhalo-bearing lens it evaluates the SAME Delta-chi2 statistic, at the SAME c=60, but with
the subhalo placed at the TRUE position (an oracle no real scan has) -- once on the scan's
three-point mass grid and once on a fine 0.25-dex mass grid -- and compares with the
scan's own grid point nearest the truth.

    python scripts/oracle_position_mass_test.py --n 150 --seed 3 --out results/oracle_position_mass
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lenstronomy.Cosmo.lens_cosmo import LensCosmo  # noqa: E402
from baseline_a.fit import fit_smooth, _sim, _unpack  # noqa: E402

CHI2_DOF_UNRELIABLE = 10.0
FINE = np.round(np.arange(8.0, 11.01, 0.25), 2)
COARSE = (8.5, 9.5, 10.5)


def delta_chi2_at(data, noise_std, smooth_vec, kwargs_band, num_pix, chi2_smooth, x, y, log10_m, c, tau=20.0, lc=None):
    sim = _sim(kwargs_band, num_pix, ["EPL", "SHEAR", "TNFW"])
    image_model = sim.image_model_class(kwargs_numerics={"supersampling_factor": 1})
    kwargs_lens_smooth, kwargs_source = _unpack(smooth_vec)
    Rs, alpha_Rs = lc.nfw_physical2angle(M=10 ** log10_m, c=c)
    kwargs_lens = kwargs_lens_smooth + [{"Rs": Rs, "alpha_Rs": alpha_Rs, "r_trunc": tau * Rs, "center_x": x, "center_y": y}]
    model = image_model.image(kwargs_lens=kwargs_lens, kwargs_source=kwargs_source)
    return float(chi2_smooth - np.sum(((data - model) / noise_std) ** 2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--population", default="test_fixed60")
    p.add_argument("--n", type=int, default=150)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--concentration", type=float, default=60.0)
    p.add_argument("--out", type=Path, default=ROOT / "results/oracle_position_mass")
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
    lc = LensCosmo(z_lens=0.5, z_source=1.0)

    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    recs = []
    for k, i in enumerate(idx):
        truth = truths[i]; sub = truth["subhalo"]; theta_E = truth["lens_macro"]["theta_E"]
        vec, chi2_smooth, _, noise_std = fit_smooth(images[i], truth, kwargs_band, num_pix, rng=rng_fit, maxiter=60)
        reliable = (chi2_smooth / num_pix ** 2) < CHI2_DOF_UNRELIABLE
        f = lambda x, y, m: delta_chi2_at(images[i], noise_std, vec, kwargs_band, num_pix, chi2_smooth, x, y, m, args.concentration, lc=lc)
        # (a) true position, scan's three masses
        d_coarse = {str(m): f(sub["x"], sub["y"], m) for m in COARSE}
        # (b) true position, fine mass grid
        d_fine = {str(m): f(sub["x"], sub["y"], float(m)) for m in FINE}
        # (c) the scan's grid point nearest the truth, three masses
        radii = np.linspace(0.5, 1.4, 3) * theta_E; angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        grid = [(r * np.cos(a), r * np.sin(a)) for r in radii for a in angles]
        gx, gy = min(grid, key=lambda g: np.hypot(g[0] - sub["x"], g[1] - sub["y"]))
        d_near = {str(m): f(gx, gy, m) for m in COARSE}
        recs.append({"index": int(i), "reliable_fit": reliable, "log10_M200_true": sub["log10_M200"], "true_x": sub["x"], "true_y": sub["y"],
                     "nearest_grid_x": float(gx), "nearest_grid_y": float(gy), "nearest_grid_offset_arcsec": float(np.hypot(gx - sub["x"], gy - sub["y"])),
                     "dchi2_true_pos_coarse": d_coarse, "dchi2_true_pos_fine": d_fine, "dchi2_nearest_grid_coarse": d_near})
        if (k + 1) % 25 == 0:
            print(f"  {k+1}/{len(idx)}  ({time.time()-t0:.0f}s)", flush=True)

    with open(args.out / "records.jsonl", "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")

    # summary
    rel = [r for r in recs if r["reliable_fit"]]
    def argmax(d): return float(max(d, key=d.get))
    tm = np.array([r["log10_M200_true"] for r in rel])
    best_true_coarse = np.array([argmax(r["dchi2_true_pos_coarse"]) for r in rel])
    best_true_fine = np.array([argmax(r["dchi2_true_pos_fine"]) for r in rel])
    best_near = np.array([argmax(r["dchi2_nearest_grid_coarse"]) for r in rel])
    # restrict to lenses the statistic actually "detects" at the true position (Delta chi2 > 20), so we
    # compare mass estimates only where there is signal to estimate from
    sig = np.array([max(r["dchi2_true_pos_fine"].values()) > 20 for r in rel])
    def stats(est, mask):
        e = est[mask] - tm[mask]
        return {"n": int(mask.sum()), "median_error_dex": float(np.median(e)) if mask.any() else None,
                "n_underestimated": int((e < 0).sum()), "n_within_0p5dex": int((np.abs(e) < 0.5).sum()),
                "frac_on_smallest_hypothesis": float((est[mask] <= 8.5).mean()) if mask.any() else None}
    summary = {"population": args.population, "n_subhalo_lenses": len(recs), "n_reliable": len(rel), "concentration": args.concentration,
               "n_with_signal_dchi2_gt_20_at_true_position": int(sig.sum()),
               "median_nearest_grid_offset_arcsec": float(np.median([r["nearest_grid_offset_arcsec"] for r in rel])),
               "true_position_three_masses": stats(best_true_coarse, sig),
               "true_position_fine_masses": stats(best_true_fine, sig),
               "nearest_grid_point_three_masses": stats(best_near, sig),
               "all_reliable_true_position_fine": stats(best_true_fine, np.ones(len(rel), bool)),
               "note": "Same Delta-chi2 statistic, same c, same smooth fit as the scan; only the subhalo position is oracle. If mass is recovered here but not at the nearest grid point, the scan's mass bias is caused by its position grid, not by its concentration assumption."}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
