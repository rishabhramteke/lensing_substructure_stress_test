"""Smooth-model MLE fit and subhalo grid scan, both using lenstronomy's own
forward model and noise estimator -- see the package docstring for the two
stated idealizations (near-truth init; fixed-concentration grid scan)."""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.SimulationAPI.sim_api import SimAPI

PARAM_NAMES = ["theta_E", "gamma", "e1", "e2", "g1", "g2", "src_x", "src_y", "R_sersic", "n_sersic", "e1_s", "e2_s", "amp"]


N_MACRO = 13          # EPL+shear+Sersic-source parameters
N_LENS_LIGHT = 5      # optional single-Sersic lens light: amp, R, n, e1, e2 (centre fixed at the lens centre)


def _sim(kwargs_band, num_pix, model_list, lens_light=0):
    """`lens_light` = number of single-Sersic lens-light components in the model (0, 1 or 2)."""
    kwargs_model = {"lens_model_list": model_list, "source_light_model_list": ["SERSIC_ELLIPSE"]}
    n_ll = int(lens_light)
    if n_ll:
        kwargs_model["lens_light_model_list"] = ["SERSIC_ELLIPSE"] * n_ll
    return SimAPI(num_pix=num_pix, kwargs_single_band=kwargs_band, kwargs_model=kwargs_model)


def _unpack(vec):
    theta_E, gamma, e1, e2, g1, g2, sx, sy, R, n, e1s, e2s, amp = vec[:N_MACRO]
    kwargs_lens = [
        {"theta_E": theta_E, "gamma": gamma, "e1": e1, "e2": e2, "center_x": 0.0, "center_y": 0.0},
        {"gamma1": g1, "gamma2": g2, "ra_0": 0.0, "dec_0": 0.0},
    ]
    kwargs_source = [{"amp": max(amp, 1e-3), "R_sersic": max(R, 1e-3), "n_sersic": np.clip(n, 0.3, 6.0), "e1": e1s, "e2": e2s, "center_x": sx, "center_y": sy}]
    return kwargs_lens, kwargs_source


def n_lens_light_components(vec):
    return (len(vec) - N_MACRO) // N_LENS_LIGHT


def _unpack3(vec):
    """(kwargs_lens, kwargs_source, kwargs_lens_light-or-None). A 13-vector has no lens
    light; each further block of 5 is one Sersic lens-light component (1 = the pipeline's
    usual single Sersic, 2 = a correctly specified double Sersic; added 2026-09-11/12)."""
    kl, ks = _unpack(vec)
    n_ll = n_lens_light_components(vec)
    if n_ll:
        kll = []
        for k in range(n_ll):
            amp_l, R_l, n_l, e1_l, e2_l = vec[N_MACRO + k * N_LENS_LIGHT:N_MACRO + (k + 1) * N_LENS_LIGHT]
            kll.append({"amp": max(amp_l, 1e-3), "R_sersic": max(R_l, 1e-3), "n_sersic": np.clip(n_l, 0.3, 8.0), "e1": e1_l, "e2": e2_l, "center_x": 0.0, "center_y": 0.0})
        return kl, ks, kll
    return kl, ks, None


def _truth_vec(truth):
    lm, s = truth["lens_macro"], truth["source"]
    return np.array([lm["theta_E"], lm["gamma"], lm["e1"], lm["e2"], lm["gamma1"], lm["gamma2"], s["x"], s["y"], s["R_sersic"], s["n_sersic"], s["e1"], s["e2"], s["amp"]])


def _init_vec(truth, data):
    """Family A always fits a Sersic source model (`_sim`'s hardcoded
    `SERSIC_ELLIPSE`), and its stated idealization is a near-truth init (see
    module docstring). For a Tier-0 Sersic truth that's `_truth_vec` above.

    A Tier-1 COSMOS truth has no Sersic parameters at all -- the true source
    isn't Sersic-shaped, so there is no "true R_sersic/n_sersic/e1_s/e2_s/amp"
    to jitter near. This is itself the honest point of running Family A on
    Tier 1: its parametric model is now genuinely misspecified relative to
    the source, not just imperfectly initialized. We keep the idealization
    for what IS known exactly regardless of source type -- the true macro-
    lens parameters and the true source *position* -- and fall back to
    generic, non-truth-informed defaults for the Sersic proxy's shape
    (mid-range size/index, round) and amplitude (estimated from the data's
    own peak pixel, the way a real fit would have to)."""
    lm, s = truth["lens_macro"], truth["source"]
    if s.get("type") == "cosmos":
        amp_guess = float(np.clip(data.max() * 3.0, 1.0, 1000.0))
        return np.array([lm["theta_E"], lm["gamma"], lm["e1"], lm["e2"], lm["gamma1"], lm["gamma2"],
                          s["x"], s["y"], 0.15, 1.5, 0.0, 0.0, amp_guess])
    return _truth_vec(truth)


def neg_log_likelihood(vec, image_model, data, noise_std):
    kwargs_lens, kwargs_source, kwargs_ll = _unpack3(vec)
    model = image_model.image(kwargs_lens=kwargs_lens, kwargs_source=kwargs_source, kwargs_lens_light=kwargs_ll)
    chi2 = np.sum(((data - model) / noise_std) ** 2)
    return 0.5 * chi2


def _residuals(vec, image_model, data, noise_std):
    kwargs_lens, kwargs_source, kwargs_ll = _unpack3(vec)
    model = image_model.image(kwargs_lens=kwargs_lens, kwargs_source=kwargs_source, kwargs_lens_light=kwargs_ll)
    return ((data - model) / noise_std).ravel()


def lens_light_image(vec, kwargs_band, num_pix):
    """The fitted single-Sersic lens light alone (PSF-convolved), for subtraction."""
    kl, ks, kll = _unpack3(vec)
    sim = _sim(kwargs_band, num_pix, ["EPL", "SHEAR"], lens_light=len(kll))
    image_model = sim.image_model_class(kwargs_numerics={"supersampling_factor": 1})
    return image_model.lens_surface_brightness(kwargs_lens_light=kll)


# (theta_E, gamma, e1, e2, g1, g2, src_x, src_y, R_sersic, n_sersic, e1_s, e2_s, amp) --
# loose physical bounds, matching the simulator's own priors (config.py), to keep the
# optimizer from wandering into unphysical (q->0, R<0) regions where the EPL profile
# diverges. SCALE is the natural step size of each parameter -- without it, a
# derivative-free/finite-difference optimizer badly misjudges step sizes when amp (~100)
# and e1/e2 (~0.1) sit 3 orders of magnitude apart; this was the actual bug behind the
# optimizer *diverging from the true parameters* seen in initial testing (2026-09-11).
BOUNDS = [(0.3, 3.0), (1.2, 2.9), (-0.6, 0.6), (-0.6, 0.6), (-0.3, 0.3), (-0.3, 0.3),
          (-1.2, 1.2), (-1.2, 1.2), (0.02, 1.0), (0.3, 6.0), (-0.6, 0.6), (-0.6, 0.6), (1.0, 1000.0)]
SCALE = [0.5, 0.5, 0.3, 0.3, 0.15, 0.15, 0.3, 0.3, 0.15, 1.5, 0.3, 0.3, 100.0]
# single-Sersic lens light (amp, R, n, e1, e2): generic bounds/scales; centre fixed at the lens centre
BOUNDS_LL = [(0.1, 2000.0), (0.05, 3.0), (0.5, 8.0), (-0.6, 0.6), (-0.6, 0.6)]
SCALE_LL = [50.0, 0.3, 2.0, 0.3, 0.3]


def _blind_init_vec(data, kwargs_band, num_pix, noise_std):
    """Data-driven initialization that uses NO truth: Einstein radius from the
    flux-weighted mean radius of the >5-sigma pixels, isothermal slope, round lens,
    zero shear, source at the lens centre with a generic mid-range Sersic and an
    amplitude from the peak pixel. Added 2026-09-11 (referee point: the near-truth
    initialization was never tested)."""
    ps = kwargs_band["pixel_scale"]
    yy, xx = np.mgrid[:num_pix, :num_pix]
    r = np.hypot((xx - (num_pix - 1) / 2) * ps, (yy - (num_pix - 1) / 2) * ps)
    sig = np.median(noise_std) if np.ndim(noise_std) else noise_std
    w = np.clip(data - 5 * sig, 0, None)
    theta_E = float(np.sum(w * r) / np.sum(w)) if w.sum() > 0 else 1.0
    theta_E = float(np.clip(theta_E, 0.5, 1.6))
    amp = float(np.clip(data.max() * 3.0, 1.0, 1000.0))
    return np.array([theta_E, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.15, 1.5, 0.0, 0.0, amp])


def fit_smooth(data, truth, kwargs_band, num_pix, jitter_frac=0.15, rng=None, maxiter=200, blind=False, lens_light_components=1):
    """MLE fit of EPL+shear+Sersic, initialized near the truth (see module
    docstring: this is the stated idealization, not a blind global fit) -- or,
    with blind=True, from the data alone (`_blind_init_vec`).
    Returns (best_vec, chi2, image_model, noise_std)."""
    rng = rng or np.random.default_rng(0)
    has_ll = truth.get("lens_light") is not None
    n_ll = int(lens_light_components) if has_ll else 0
    sim = _sim(kwargs_band, num_pix, ["EPL", "SHEAR"], lens_light=n_ll)
    image_model = sim.image_model_class(kwargs_numerics={"supersampling_factor": 1})
    noise_std = sim.estimate_noise(data)

    if blind:
        x_init = _blind_init_vec(data, kwargs_band, num_pix, noise_std)
    else:
        x0 = _init_vec(truth, data)
        jitter = 1.0 + rng.uniform(-jitter_frac, jitter_frac, size=x0.shape)
        jitter[np.abs(x0) < 1e-6] = 1.0  # additive params near 0 (e1,e2,shear) get an additive jitter instead
        x_init = np.where(np.abs(x0) < 1e-6, x0 + rng.uniform(-0.02, 0.02, size=x0.shape), x0 * jitter)
    bounds, scale = list(BOUNDS), list(SCALE)
    if has_ll:
        # the TRUE lens light is two-component. lens_light_components=1 is the pipeline's usual single
        # Sersic (misspecified); =2 is a correctly specified double Sersic. Either way the initialisation
        # is generic and data-driven (central pixel, R=0.5"/1.0", n=4/1, round) -- never from the truth.
        c = (num_pix - 1) // 2
        amp_guess = float(np.clip(data[c - 1:c + 2, c - 1:c + 2].mean() * 0.5, 0.5, 2000.0))
        inits = [[amp_guess, 0.5, 4.0, 0.0, 0.0], [amp_guess * 0.3, 1.0, 1.0, 0.0, 0.0]][:n_ll]
        for ini in inits:
            x_init = np.concatenate([x_init, ini]); bounds += BOUNDS_LL; scale += SCALE_LL
    x_init = np.clip(x_init, [b[0] for b in bounds], [b[1] for b in bounds])

    lo = [b[0] for b in bounds]; hi = [b[1] for b in bounds]
    res = least_squares(_residuals, x_init, args=(image_model, data, noise_std), bounds=(lo, hi),
                         x_scale=scale, method="trf", max_nfev=maxiter * len(x_init))
    chi2 = float(np.sum(res.fun ** 2))
    return res.x, chi2, image_model, noise_std


def scan_subhalo(data, noise_std, smooth_vec, kwargs_band, num_pix, theta_E, chi2_smooth,
                  z_lens=0.5, z_source=1.0, concentration=15.0, tau=20.0,
                  n_radii=3, n_angles=8, log10_mass_grid=(8.5, 9.5, 10.5)):
    """Grid-scan an NFW subhalo on top of the fitted smooth model. Fixed
    concentration (see module docstring); position on an annulus around
    theta_E; mass on a coarse log-grid. Returns the max Delta-chi2 over the
    grid (the detection statistic) and the argmax (position, mass)."""
    kwargs_lens_smooth, kwargs_source, kwargs_ll = _unpack3(smooth_vec)
    sim = _sim(kwargs_band, num_pix, ["EPL", "SHEAR", "TNFW"], lens_light=0 if kwargs_ll is None else len(kwargs_ll))
    image_model = sim.image_model_class(kwargs_numerics={"supersampling_factor": 1})
    lc = LensCosmo(z_lens=z_lens, z_source=z_source)

    radii = np.linspace(0.5, 1.4, n_radii) * theta_E
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    best = {"delta_chi2": -np.inf, "x": None, "y": None, "log10_m": None}
    for r in radii:
        for a in angles:
            x, y = r * np.cos(a), r * np.sin(a)
            for log10_m in log10_mass_grid:
                # `concentration` may be a number (the fixed-c idealization) or a callable
                # c(log10_m) -- e.g. a concentration-mass relation -- added 2026-09-11 after a
                # self-review flagged that fixing c=15 against a c=60 population makes the mass
                # bias partly self-inflicted; see first_results.md "concentration variants".
                c_here = concentration(log10_m) if callable(concentration) else concentration
                Rs, alpha_Rs = lc.nfw_physical2angle(M=10**log10_m, c=c_here)
                kwargs_lens = kwargs_lens_smooth + [{"Rs": Rs, "alpha_Rs": alpha_Rs, "r_trunc": tau * Rs, "center_x": x, "center_y": y}]
                model = image_model.image(kwargs_lens=kwargs_lens, kwargs_source=kwargs_source, kwargs_lens_light=kwargs_ll)
                chi2 = np.sum(((data - model) / noise_std) ** 2)
                delta = chi2_smooth - chi2
                if delta > best["delta_chi2"]:
                    best = {"delta_chi2": float(delta), "x": float(x), "y": float(y), "log10_m": float(log10_m)}
    return best
