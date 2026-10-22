"""Cross-code check of the lens model, lenstronomy against PyAutoLens, on 100 c=60 lenses spanning the axis-ratio and
slope priors (referee round 29, M7). For each lens the true EPL+shear macro-model and TNFW subhalo are built in both
codes: the PyAutoLens power law with the Einstein radius rescaled per lens so that
the deflections agree (the gravitational-imaging pipeline's conversion; the analytic factor
f = [2 sqrt(q) / (1 + q)]^(1 / (gamma - 1)) is reported for comparison and does not reproduce it), and the TNFW with kappa_s = alpha_Rs / (4 Rs (1 - ln 2)). Reported per
lens: the rms and maximum deflection difference on the image grid (macro and subhalo separately, in arcsec and as a
fraction of the local deflection), and the difference between the images of the same lenstronomy source ray-traced
through the two deflection fields and convolved with the same PSF, in units of the data's per-pixel noise sigma
(full image and subhalo imprint). A per-lens fitted theta_E rescale (the gravitational-imaging pipeline's) is the
comparison for the analytic factor.

    python scripts/validate_crosscode.py --n 100      -> results/pyautolens_validation/crosscode.json
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
import autolens as al  # noqa: E402
from lenstronomy.LensModel.lens_model import LensModel  # noqa: E402
from lenstronomy.LightModel.light_model import LightModel  # noqa: E402
from lenstronomy.Util import param_util  # noqa: E402
import run_baseline_a_joint as J  # noqa: E402
from baseline_a.fit import _unpack3, _truth_vec  # noqa: E402
from run_family_b_gi import pyautolens_lens  # noqa: E402


def factor(q, g):
    return (2 * np.sqrt(q) / (1 + q)) ** (1.0 / (g - 1.0))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=100); a = ap.parse_args()
    J._init(ROOT / "data/test_fixed60", 60.0, 1e12, True, 0)
    G = J.G; n_pix = G["num_pix"]; ps = G["kwargs_band"]["pixel_scale"]; c = (n_pix - 1) / 2
    yy, xx = np.mgrid[0:n_pix, 0:n_pix]; X = ((xx - c) * ps).ravel(); Y = ((yy - c) * ps).ravel()
    grid = al.Grid2DIrregular(values=np.stack([Y, X], 1))
    sim = J._sim(G["kwargs_band"], n_pix, ["EPL", "SHEAR"])
    idx = [i for i, t in enumerate(G["truths"]) if t.get("subhalo")][: a.n]
    LM = LensModel(["EPL", "SHEAR"]); LMs = LensModel(["TNFW"]); src = LightModel(["SERSIC_ELLIPSE"])
    sig_psf = G["kwargs_band"].get("seeing", 0.08) / 2.3548 / ps
    out = []
    for i in idx:
        t = G["truths"][i]; kl, ks, _ = _unpack3(_truth_vec(t)); s = t["subhalo"]
        phi, q = param_util.ellipticity2phi_q(kl[0]["e1"], kl[0]["e2"]); g = kl[0]["gamma"]
        ell = al.convert.ell_comps_from(axis_ratio=float(q), angle=float(np.degrees(phi)))
        kws = J.sub_kwargs(s["log10_M200"], s["x"], s["y"])
        mk = lambda th: al.Galaxy(redshift=0.5, mass=al.mp.PowerLaw(centre=(kl[0]["center_y"], kl[0]["center_x"]), ell_comps=ell, einstein_radius=float(th), slope=g),  # noqa: E731
                                  shear=al.mp.ExternalShear(gamma_1=kl[1]["gamma1"], gamma_2=kl[1]["gamma2"]))
        sub = al.mp.NFWTruncatedSph(centre=(kws["center_y"], kws["center_x"]), kappa_s=kws["alpha_Rs"] / (4 * kws["Rs"] * (1 - np.log(2))),
                                    scale_radius=kws["Rs"], truncation_radius=kws["r_trunc"])
        ax_l, ay_l = LM.alpha(X, Y, kl[:2]); sx_l, sy_l = LMs.alpha(X, Y, [kws])
        A = np.asarray(mk(kl[0]["theta_E"] / factor(q, g)).deflections_yx_2d_from(grid=grid)); Sd = np.asarray(sub.deflections_yx_2d_from(grid=grid))
        dmac = np.hypot(A[:, 1] - ax_l, A[:, 0] - ay_l); dsub = np.hypot(Sd[:, 1] - sx_l, Sd[:, 0] - sy_l)
        lens_fit, rms_fit = pyautolens_lens(kl, 0.5, np.stack([Y, X], 1)); th_fit = float(lens_fit.mass.einstein_radius)
        noise = sim.estimate_noise(np.asarray(G["images"][i])).ravel()
        N_true = np.load(ROOT / "data/test_fixed60/images_full_noiseless.npy", mmap_mode="r")[i].ravel()
        def img(axx, ayy):
            return gaussian_filter(src.surface_brightness(X - axx, Y - ayy, [ks[0]]).reshape(n_pix, n_pix), sig_psf).ravel()
        I_l = img(ax_l + sx_l, ay_l + sy_l); k_ = float(np.dot(N_true, I_l) / np.dot(I_l, I_l))   # flux scale of the render to the data
        img0 = img
        img = lambda axx, ayy: k_ * img0(axx, ayy)  # noqa: E731
        Af = np.asarray(lens_fit.deflections_yx_2d_from(grid=grid))   # the fitted-rescale model the pipeline uses
        I_l = img(ax_l + sx_l, ay_l + sy_l); I_p = img(Af[:, 1] + Sd[:, 1], Af[:, 0] + Sd[:, 0])
        I_l0 = img(ax_l, ay_l); I_p0 = img(Af[:, 1], Af[:, 0])
        r_full = (I_p - I_l) / noise; imp_l, imp_p = I_l - I_l0, I_p - I_p0
        arc = I_l / noise > 3
        out.append({"index": i, "q": float(q), "gamma": float(g), "factor": float(factor(q, g)),
                    "macro_defl_rms_analytic_factor": float(np.sqrt(np.mean(dmac ** 2))), "macro_defl_max_analytic_factor": float(dmac.max()),
                    "macro_defl_frac_max_analytic_factor": float(np.max(dmac / np.maximum(np.hypot(ax_l, ay_l), 1e-3))),
                    "macro_defl_rms_fitted_rescale": rms_fit, "theta_ratio_fit": th_fit / kl[0]["theta_E"],
                    "render_vs_truth_resid_sigma_rms_arc": float(np.sqrt(np.mean(((I_l - N_true) / noise)[I_l / noise > 3] ** 2))),
                    "sub_defl_rms": float(np.sqrt(np.mean(dsub ** 2))), "sub_defl_frac_max": float(np.max(dsub / np.maximum(np.hypot(sx_l, sy_l), 1e-6))),
                    "image_resid_sigma_rms_arc": float(np.sqrt(np.mean(r_full[arc] ** 2))), "image_resid_sigma_max": float(np.abs(r_full).max()),
                    "imprint_resid_sigma_max": float(np.abs((imp_p - imp_l) / noise).max()),
                    "imprint_peak_sigma": float(np.abs(imp_l / noise).max()),
                    "imprint_corr": float(np.corrcoef(imp_l, imp_p)[0, 1])})
    keys = [k for k in out[0] if k not in ("index",)]
    summ = {k: {"median": float(np.median([o[k] for o in out])), "max": float(np.max([o[k] for o in out])), "min": float(np.min([o[k] for o in out]))} for k in keys}
    summ["n"] = len(out)
    d = ROOT / "results/pyautolens_validation"; d.mkdir(parents=True, exist_ok=True)
    (d / "crosscode.json").write_text(json.dumps({"summary": summ, "lenses": out}, indent=2))
    for k in keys:
        print(f"{k:32s} median {summ[k]['median']:.3g}  min {summ[k]['min']:.3g}  max {summ[k]['max']:.3g}")


if __name__ == "__main__":
    main()
