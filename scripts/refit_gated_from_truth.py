"""Are the c=60 lenses the joint scan cannot fit genuinely unfittable, or did the optimizer fail (referee round 26, M5)?
Every lens of the full c=60 population that stays outside the gate after the multi-start rescue is re-fitted from the
exact truth: the smooth model from the true macro and source, and the joint model with the TNFW at the true position
and mass (macro, source and mass free, position free within 0.1"), once with the scan's c=15 template and once with
the true c=60 profile. Reported: chi^2/dof of each fit and whether it passes the gate (chi^2/dof < 10).

    python scripts/refit_gated_from_truth.py --workers 8      -> results/gated_refit_truth.json
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
import run_baseline_a_joint as J  # noqa: E402
from baseline_a.fit import _unpack3, _truth_vec, BOUNDS, SCALE  # noqa: E402

BOX = 0.1


def one(i):
    G = J.G; truth = G["truths"][i]; data = np.asarray(G["images"][i]); s = truth["subhalo"]
    sim = J._sim(G["kwargs_band"], G["num_pix"], ["EPL", "SHEAR"])
    noise = sim.estimate_noise(data)
    lo = [b[0] for b in BOUNDS]; hi = [b[1] for b in BOUNDS]
    v0 = np.clip(_truth_vec(truth), np.array(lo) + 1e-9, np.array(hi) - 1e-9)
    n = len(v0); npix = G["num_pix"] ** 2
    im_s, im_j = G["image_model_smooth"], G["image_model"]

    def rs(v):
        kl, ks, kll = _unpack3(v)
        return ((data - im_s.image(kwargs_lens=kl, kwargs_source=ks, kwargs_lens_light=kll)) / noise).ravel()
    r_s = least_squares(rs, v0, bounds=(lo, hi), x_scale=SCALE, method="trf", max_nfev=200 * (n + 1))
    out = {"index": int(i), "log10_M200_true": s["log10_M200"], "chi2dof_smooth_from_truth": float(np.sum(r_s.fun ** 2) / npix)}
    for conc in (15.0, 60.0):
        G["conc"] = conc

        def rj(v):
            kl, ks, kll = _unpack3(v[:n])
            return ((data - im_j.image(kwargs_lens=kl + [J.sub_kwargs(v[n], v[n + 1], v[n + 2])], kwargs_source=ks, kwargs_lens_light=kll)) / noise).ravel()
        x0 = np.concatenate([v0, [s["log10_M200"], s["x"], s["y"]]])
        lo_j = lo + [7.5, s["x"] - BOX, s["y"] - BOX]; hi_j = hi + [11.5, s["x"] + BOX, s["y"] + BOX]
        x0 = np.clip(x0, np.array(lo_j) + 1e-9, np.array(hi_j) - 1e-9)
        chi2_start = float(np.sum(rj(x0) ** 2) / npix)
        r_j = least_squares(rj, x0, bounds=(lo_j, hi_j), x_scale=list(SCALE) + [1.0, 0.05, 0.05], method="trf", max_nfev=200 * (n + 4))
        out[f"chi2dof_joint_c{int(conc)}_at_truth_start"] = chi2_start
        out[f"chi2dof_joint_c{int(conc)}"] = float(np.sum(r_j.fun ** 2) / npix)
        out[f"log10_m_fit_c{int(conc)}"] = float(r_j.x[n])
    return out


def main():
    p = argparse.ArgumentParser(); p.add_argument("--workers", type=int, default=8); a = p.parse_args()
    root = ROOT / "results/baseline_a_joint_full/test_fixed60"
    recs = [json.loads(l) for l in open(root / "scan_results.jsonl")]
    gated = [r["index"] for r in recs if r["has_subhalo"] and not r["reliable_fit"]]
    J._init(ROOT / "data/test_fixed60", 15.0, 1e12, True, 0)
    import multiprocessing as mp
    with mp.get_context("fork").Pool(a.workers) as pool:
        res = sorted(pool.map(one, gated), key=lambda r: r["index"])
    summ = {"n_gated": len(res)}
    for k in ("chi2dof_smooth_from_truth", "chi2dof_joint_c15", "chi2dof_joint_c60", "chi2dof_joint_c60_at_truth_start"):
        v = np.array([r[k] for r in res])
        summ[k] = {"median": float(np.median(v)), "n_below_10": int((v < 10).sum()), "n_below_2": int((v < 2).sum())}
    (ROOT / "results/gated_refit_truth.json").write_text(json.dumps({"summary": summ, "lenses": res}, indent=2))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
