"""Is the published PyAutoLens pipeline's best subhalo position for c=60 lens 8 (the y-mirror of the truth) a real
optimum of the lens model or a frame error (referee round 29)? Our own joint fit (macro-model and source free, TNFW of
free mass, position free within 0.5") is started at the truth and at the PyAutoLens position, with the scan's c=15
template and the true c=60 profile, from the true macro-model and source; the fitted chi^2 decides.

    python scripts/check_pal_mirror.py      -> results/pyautolens_slam/mirror_check.json
"""
import json, sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
import run_baseline_a_joint as J  # noqa: E402
from baseline_a.fit import _unpack3, _truth_vec, BOUNDS, SCALE  # noqa: E402

I, BOX = 8, 0.5
PAL = json.loads(open(ROOT / "results/pyautolens_slam/test_fixed60/scan_results.jsonl").readline())


def main():
    J._init(ROOT / "data/test_fixed60", 15.0, 1e12, True, 0)
    G = J.G; truth = G["truths"][I]; data = np.asarray(G["images"][I]); s = truth["subhalo"]
    sim = J._sim(G["kwargs_band"], G["num_pix"], ["EPL", "SHEAR"]); noise = sim.estimate_noise(data)
    lo = [b[0] for b in BOUNDS]; hi = [b[1] for b in BOUNDS]
    v0 = np.clip(_truth_vec(truth), np.array(lo) + 1e-9, np.array(hi) - 1e-9); n = len(v0); npix = G["num_pix"] ** 2
    im_s, im_j = G["image_model_smooth"], G["image_model"]

    def rs(v):
        kl, ks, kll = _unpack3(v)
        return ((data - im_s.image(kwargs_lens=kl, kwargs_source=ks, kwargs_lens_light=kll)) / noise).ravel()
    r_s = least_squares(rs, v0, bounds=(lo, hi), x_scale=SCALE, method="trf", max_nfev=200 * (n + 1))
    out = {"index": I, "truth_xy": [s["x"], s["y"]], "log10_M200_true": s["log10_M200"],
           "pal_xy": [PAL["best_x"], PAL["best_y"]], "chi2_smooth": float(np.sum(r_s.fun ** 2)), "fits": []}
    for conc in (15.0, 60.0):
        G["conc"] = conc
        for start, (x, y) in (("truth", (s["x"], s["y"])), ("pal_mirror", (PAL["best_x"], PAL["best_y"]))):
            def rj(v):
                kl, ks, kll = _unpack3(v[:n])
                return ((data - im_j.image(kwargs_lens=kl + [J.sub_kwargs(v[n], v[n + 1], v[n + 2])], kwargs_source=ks, kwargs_lens_light=kll)) / noise).ravel()
            lo_j = lo + [7.5, x - BOX, y - BOX]; hi_j = hi + [11.5, x + BOX, y + BOX]
            x0 = np.clip(np.concatenate([v0, [10.5, x, y]]), np.array(lo_j) + 1e-9, np.array(hi_j) - 1e-9)
            r = least_squares(rj, x0, bounds=(lo_j, hi_j), x_scale=list(SCALE) + [1.0, 0.05, 0.05], method="trf", max_nfev=300 * (n + 4))
            c2 = float(np.sum(r.fun ** 2))
            f = {"template_c": conc, "start": start, "chi2_joint": c2, "chi2dof": c2 / npix, "dchi2_vs_smooth": out["chi2_smooth"] - c2,
                 "fit_xy": [float(r.x[n + 1]), float(r.x[n + 2])], "log10_m_fit": float(r.x[n])}
            out["fits"].append(f); print(json.dumps(f), flush=True)
    (ROOT / "results/pyautolens_slam/mirror_check.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
