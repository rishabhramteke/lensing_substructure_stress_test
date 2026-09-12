"""Real-world sanity check for Family A (`src/baseline_a/fit.py`): run our
fast smooth-fit + subhalo-scan pipeline on an ACTUAL published gravitational
lens, not a simulated one, and see whether it lands near what an independent,
peer-reviewed analysis reported for the same real system.

This is NOT a ground-truth test (no real lens has known ground truth -- see
first_results.md's "real lens catalogs" discussion) and NOT an attempt to
reproduce Sengul+2022's own multi-plane nested-sampling pipeline exactly.
It is a cheaper, independent question: does a *reasonable* single-plane fit
+ scan on the same real image land in the same rough neighborhood (position,
mass scale) as their published result, using completely different code?

Target system: JVAS B1938+666, Sengul, Dvorkin, Ostdiek & Tsang 2022
("Substructure Detection Reanalyzed: Dark Perturber shown to be a
Line-of-Sight Halo", arXiv:2112.00749, MNRAS 515, 4391). Their own code +
real HST/NICMOS F160W image (drizzled, `HST_7255_07_NIC_NIC1_F160W_drz.fits`)
are public: https://github.com/acagansengul/interlopers_with_lenstronomy.
Their own fitting script (`jvas_nested2.py`) needs two intermediate arrays
(`bckg.npy`, `pois.npy`) that are NOT in that public repo (only the notebooks
and the raw FITS are) -- so their *exact* pixel-level reduction (e.g. their
source-subtracted residual image) cannot be reproduced here. What IS used
here: the same real FITS image, real redshifts (z_lens=0.881,
z_source=2.059), real pixel scale (0.025"/px) and real PSF FWHM (0.14")
straight from their own script -- and our own independent smooth-fit + scan,
not theirs.

Their own reported prior range for the interloper's best-fit region
(`results/priors.txt`, "narrowprior", clearly set narrow around their actual
best fit): x_int in [-0.15, 0.15]", y_int in [0.4, 0.6]", M200 up to
3.5x10^10 Msun. That range -- not a single point estimate we could not
otherwise extract without their exact MCMC chain files -- is what this
script's result is compared against.

    source ~/myenv/bin/activate
    python scripts/validate_real_lens.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.SimulationAPI.sim_api import SimAPI

FITS_PATH = Path(__file__).resolve().parents[1] / "data/real_lens_b1938/HST_7255_07_NIC_NIC1_F160W_drz.fits"
Z_LENS, Z_SOURCE = 0.881, 2.059  # Sengul+2022's own values, jvas_nested2.py
PIXEL_SCALE = 0.025  # arcsec/px, real NICMOS/F160W drizzled pixel scale, their own script
PSF_FWHM = 0.14  # arcsec, their own script
NUM_PIX = 70
RESCALE = 500.0  # brings real peak flux (~0.14) into the same numerical range our fit's bounds/x_scale assume (~tens-hundreds); chi2 is invariant under a uniform data+noise rescale


def load_real_cutout():
    hdul = fits.open(FITS_PATH)
    sci, wht = hdul[1].data.astype(np.float64), hdul[2].data.astype(np.float64)
    valid = wht > 0
    sm = gaussian_filter(np.nan_to_num(sci), sigma=2)
    sm[~valid] = 0
    cy, cx = np.unravel_index(np.argmax(sm), sm.shape)  # locate the lens system itself, not image center
    half = NUM_PIX // 2
    data = sci[cy - half:cy + half, cx - half:cx + half] * RESCALE

    # The WHT extension's absolute calibration didn't produce a physically
    # sensible noise level here (came out ~60x larger than the peak signal,
    # for an image where the lens is clearly visible by eye -- some
    # drizzle-specific weight normalization is being missed). Falling back to
    # a simple, standard, defensible estimate instead: robust sigma of the
    # background-dominated corner regions of this same cutout.
    corner_size = 12
    corners = np.concatenate([
        sci[cy - half:cy - half + corner_size, cx - half:cx - half + corner_size].ravel(),
        sci[cy - half:cy - half + corner_size, cx + half - corner_size:cx + half].ravel(),
        sci[cy + half - corner_size:cy + half, cx - half:cx - half + corner_size].ravel(),
        sci[cy + half - corner_size:cy + half, cx + half - corner_size:cx + half].ravel(),
    ])
    bg_std = float(np.std(corners)) * RESCALE
    noise_std = np.full_like(data, max(bg_std, 1e-6))
    print(f"background sigma estimate (corners, rescaled): {bg_std:.3f}; data max {data.max():.1f} -> peak SNR ~{data.max()/bg_std:.1f}")
    return data, noise_std, (cy, cx)


def kwargs_band_real():
    return {"pixel_scale": PIXEL_SCALE, "psf_type": "GAUSSIAN", "seeing": PSF_FWHM,
            "exposure_time": 1.0, "sky_brightness": 0.0, "num_exposures": 1,
            "background_noise": True, "magnitude_zero_point": 25.0}


PARAM_NAMES = ["theta_E", "gamma", "e1", "e2", "g1", "g2", "src_x", "src_y", "R_sersic", "n_sersic", "e1_s", "e2_s", "amp"]
BOUNDS = [(0.25, 0.9), (1.0, 3.0), (-0.6, 0.6), (-0.6, 0.6), (-0.5, 0.5), (-0.5, 0.5),
          (-0.6, 0.6), (-0.6, 0.6), (0.02, 1.2), (0.3, 6.0), (-0.6, 0.6), (-0.6, 0.6), (10.0, 50000.0)]
SCALE = [0.08, 0.4, 0.2, 0.2, 0.15, 0.15, 0.2, 0.2, 0.2, 1.0, 0.25, 0.25, 5000.0]


def unpack(vec):
    theta_E, gamma, e1, e2, g1, g2, sx, sy, R, n, e1s, e2s, amp = vec
    kwargs_lens = [{"theta_E": theta_E, "gamma": gamma, "e1": e1, "e2": e2, "center_x": 0.0, "center_y": 0.0},
                   {"gamma1": g1, "gamma2": g2, "ra_0": 0.0, "dec_0": 0.0}]
    kwargs_source = [{"amp": max(amp, 1e-3), "R_sersic": max(R, 1e-3), "n_sersic": np.clip(n, 0.3, 6.0), "e1": e1s, "e2": e2s, "center_x": sx, "center_y": sy}]
    return kwargs_lens, kwargs_source


def residuals(vec, image_model, data, noise_std):
    kwargs_lens, kwargs_source = unpack(vec)
    model = image_model.image(kwargs_lens=kwargs_lens, kwargs_source=kwargs_source)
    return ((data - model) / noise_std).ravel()


def main():
    data, noise_std, center_px = load_real_cutout()
    print(f"real B1938+666 cutout: {data.shape}, centered on FITS pixel {center_px}, flux sum={data.sum():.1f} (rescaled x{RESCALE:.0f})")

    kwargs_band = kwargs_band_real()
    sim = SimAPI(num_pix=NUM_PIX, kwargs_single_band=kwargs_band, kwargs_model={"lens_model_list": ["EPL", "SHEAR"], "source_light_model_list": ["SERSIC_ELLIPSE"]})
    image_model = sim.image_model_class(kwargs_numerics={"supersampling_factor": 1})

    # No ground truth exists for real data -- a generic, physically-reasonable
    # blind starting point (not "near truth", since there is no truth to be near):
    # theta_E and gamma from the isothermal-ish middle of Sengul+2022's own
    # prior range; everything else neutral/centered. This is *more* realistic
    # than this project's usual near-truth-init idealization (see
    # src/baseline_a/__init__.py) -- a real blind fit has no truth to start near.
    x0 = np.array([0.47, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 1.5, 0.0, 0.0, data.max() * 3])
    rng = np.random.default_rng(0)
    x_init = x0 * (1 + rng.uniform(-0.05, 0.05, size=x0.shape))
    lo = [b[0] for b in BOUNDS]; hi = [b[1] for b in BOUNDS]
    x_init = np.clip(x_init, lo, hi)

    res = least_squares(residuals, x_init, args=(image_model, data, noise_std), bounds=(lo, hi), x_scale=SCALE, method="trf", max_nfev=60 * len(x0))
    chi2_smooth = float(np.sum(res.fun ** 2))
    dof = data.size
    print(f"smooth fit: status={res.status} success={res.success} nfev={res.nfev} chi2/dof={chi2_smooth/dof:.2f}")
    print("smooth params:", dict(zip(PARAM_NAMES, np.round(res.x, 3))))
    theta_E_fit = res.x[0]

    # Subhalo/interloper scan -- Sengul+2022's own reported result sits at
    # y~0.4-0.6" from center, OUTSIDE this project's usual 0.5-1.4*theta_E
    # annulus (theta_E~0.47" here -> 0.24-0.66"); widened to 0.3-1.6*theta_E
    # for this real-data run specifically (not changed in fit.py's shared
    # scan_subhalo, which keeps its Tier-0/Tier-1 default).
    lc = LensCosmo(z_lens=Z_LENS, z_source=Z_SOURCE)
    radii = np.linspace(0.3, 1.6, 5) * theta_E_fit
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    mass_grid = (8.0, 8.5, 9.0, 9.5, 10.0, 10.5)  # log10 Msun, spans Sengul+2022's own M200 range (up to ~3.5e10)
    kwargs_lens_smooth, kwargs_source = unpack(res.x)
    sim_sub = SimAPI(num_pix=NUM_PIX, kwargs_single_band=kwargs_band, kwargs_model={"lens_model_list": ["EPL", "SHEAR", "TNFW"], "source_light_model_list": ["SERSIC_ELLIPSE"]})
    im_sub = sim_sub.image_model_class(kwargs_numerics={"supersampling_factor": 1})
    best = {"delta_chi2": -np.inf}
    for r in radii:
        for a in angles:
            x, y = r * np.cos(a), r * np.sin(a)
            for log10_m in mass_grid:
                Rs, alpha_Rs = lc.nfw_physical2angle(M=10 ** log10_m, c=15.0)
                kwargs_lens = kwargs_lens_smooth + [{"Rs": Rs, "alpha_Rs": alpha_Rs, "r_trunc": 20 * Rs, "center_x": x, "center_y": y}]
                model = im_sub.image(kwargs_lens=kwargs_lens, kwargs_source=kwargs_source)
                chi2 = np.sum(((data - model) / noise_std) ** 2)
                delta = chi2_smooth - chi2
                if delta > best["delta_chi2"]:
                    best = {"delta_chi2": float(delta), "x": float(x), "y": float(y), "log10_m": float(log10_m)}

    print(f"\nbest-fit interloper/subhalo candidate: x={best['x']:.3f}\" y={best['y']:.3f}\"  "
          f"log10(M/Msun)={best['log10_m']:.1f}  delta_chi2={best['delta_chi2']:.1f}")
    print(f"Sengul+2022's own reported region (their narrow prior, arXiv:2112.00749): "
          f"x in [-0.15, 0.15]\", y in [0.4, 0.6]\", M200 up to 3.5e10 Msun (log10 ~10.5)")

    in_x = -0.15 <= best["x"] <= 0.15
    in_y = 0.4 <= best["y"] <= 0.6
    print(f"\nposition within their reported region: x {'YES' if in_x else 'NO'}, y {'YES' if in_y else 'NO'}")


if __name__ == "__main__":
    main()
