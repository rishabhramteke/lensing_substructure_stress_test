"""Effective lens-plane mass of a line-of-sight halo, in the scan's own template (referee round 22, M8).

A lens-plane model assigns a line-of-sight halo an effective mass that differs from the halo's own M200. To separate
that expected rescaling from the detector's error, each line-of-sight lens is fitted, with the macro-model and source
held at their TRUE values, by one lens-plane TNFW in the scan's template (c = 15, tau = 20) whose mass and position are
free, against the noiseless multi-plane image with the noise level of the observed one. The best-fit mass is the halo's
effective lens-plane mass for that template; the scan's reported mass is then compared with it.

    python scripts/los_effective_mass.py --workers 2      -> results/los_effective_mass.json
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
import run_baseline_a_joint as J  # noqa: E402
from baseline_a.fit import _sim  # noqa: E402

ARMS = {"z025": ("los_halo_z025", "results/baseline_a_joint_los/los_z025", "results/baseline_a_joint_freepos/los_z025"),
        "z075": ("los_halo_z075", "results/baseline_a_joint_los/los_z075", "results/baseline_a_joint_freepos/los_z075")}
G = {}


def one(i):
    t = G["truths"][i]; lm = t["lens_macro"]; s = t["source"]; h = t["los_halo"]
    noisy = np.asarray(G["noisy"][i]); clean = np.asarray(G["clean"][i])
    sim = _sim(G["kwargs_band"], G["num_pix"], ["EPL", "SHEAR"]); sigma = sim.estimate_noise(noisy)
    simt = _sim(G["kwargs_band"], G["num_pix"], ["EPL", "SHEAR", "TNFW"]); im = simt.image_model_class(kwargs_numerics={"supersampling_factor": 1})
    kl0 = [{"theta_E": lm["theta_E"], "gamma": lm["gamma"], "e1": lm["e1"], "e2": lm["e2"], "center_x": 0.0, "center_y": 0.0},
           {"gamma1": lm["gamma1"], "gamma2": lm["gamma2"], "ra_0": 0.0, "dec_0": 0.0}]
    ks = [{"amp": s["amp"], "R_sersic": s["R_sersic"], "n_sersic": s["n_sersic"], "e1": s["e1"], "e2": s["e2"], "center_x": s["x"], "center_y": s["y"]}]

    def resid(v):
        return ((clean - im.image(kwargs_lens=kl0 + [J.sub_kwargs(v[0], v[1], v[2])], kwargs_source=ks)) / sigma).ravel()
    best = None
    for m0 in (8.5, 9.5, 10.5):
        r = least_squares(resid, [m0, h["x"], h["y"]], bounds=([7.0, h["x"] - 0.5, h["y"] - 0.5], [12.0, h["x"] + 0.5, h["y"] + 0.5]), method="trf", max_nfev=400)
        if best is None or r.cost < best.cost:
            best = r
    return {"index": i, "log10_M200_true": h["log10_M200"], "log10_m_eff": float(best.x[0]), "x_eff": float(best.x[1]), "y_eff": float(best.x[2]),
            "chi2_eff": float(2 * best.cost)}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--workers", type=int, default=2); a = p.parse_args()
    import multiprocessing as mp
    out = {}
    J.G["conc"] = 15.0
    for arm, (pop, grid_root, free_root) in ARMS.items():
        man = json.loads((ROOT / "data" / pop / "manifest.json").read_text())
        G.update({"kwargs_band": man["kwargs_band"], "num_pix": man["image_shape"][0], "truths": [json.loads(l) for l in open(ROOT / "data" / pop / "truth.jsonl")],
                  "noisy": np.load(ROOT / "data" / pop / "images_full_noisy.npy", mmap_mode="r"), "clean": np.load(ROOT / "data" / pop / "images_full_noiseless.npy", mmap_mode="r")})
        runs = {k: {r["index"]: r for r in (json.loads(l) for l in open(ROOT / root / "scan_results.jsonl"))} for k, root in (("grid", grid_root), ("free", free_root))}
        idx = sorted(i for i in set(runs["grid"]) | set(runs["free"]) if any(runs[k].get(i, {}).get("delta_chi2", -1) >= 20 for k in runs))
        with mp.get_context("fork").Pool(a.workers) as pool:
            eff = {r["index"]: r for r in pool.map(one, idx)}
        res = {"n_fitted": len(eff), "offset_eff_minus_true_median": float(np.median([e["log10_m_eff"] - e["log10_M200_true"] for e in eff.values()]))}
        for k in ("grid", "free"):
            det = [r for r in runs[k].values() if r["reliable_fit"] and r["delta_chi2"] >= 20 and r["index"] in eff]
            res[k] = {"n_det": len(det),
                      "err_vs_true_median": float(np.median([r["best_log10_m"] - eff[r["index"]]["log10_M200_true"] for r in det])),
                      "err_vs_eff_median": float(np.median([r["best_log10_m"] - eff[r["index"]]["log10_m_eff"] for r in det])),
                      "err_vs_eff_mad": float(np.median(np.abs(np.array([r["best_log10_m"] - eff[r["index"]]["log10_m_eff"] for r in det]) - np.median([r["best_log10_m"] - eff[r["index"]]["log10_m_eff"] for r in det]))))}
        out[arm] = res
        print(arm, json.dumps(res))
    (ROOT / "results/los_effective_mass.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
