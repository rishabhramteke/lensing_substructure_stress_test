"""Close the gap left by validate_pyautolens.py: cross-check the TNFW *subhalo*
forward model between lenstronomy (Family A's engine) and real PyAutoLens.

    source ~/myenv/bin/activate
    python scripts/validate_pyautolens_tnfw.py --n 5 --out results/pyautolens_validation

The macro model (EPL+shear+Sersic) was already shown to agree (r=0.94-1.00,
`validate_pyautolens.py`). The subhalo was NOT: lenstronomy's TNFW takes
(Rs, alpha_Rs, r_trunc), PyAutoLens's NFWTruncatedSph takes (kappa_s,
scale_radius, truncation_radius). Since Family A's delta-chi2 detection
statistic is literally "does adding this profile improve the fit", the
profile itself is load-bearing.

Parameter conversion, derived from both packages' own source and then pinned
down NUMERICALLY (isolated-profile deflection/convergence, 1e-3 to 5 Rs, three
(M, c, tau) combinations -- see `isolated_profile_check`):

    kappa_s          = alpha_Rs / (4 * Rs * (1 + ln(1/2)))     [alpha_Rs, Rs in arcsec]
    scale_radius     = Rs                                       [arcsec]
    truncation_radius= r_trunc                                  [arcsec; tau = r_trunc/Rs in both codes]

Why that form: lenstronomy.LensCosmo.nfw_physical2angle defines
alpha_Rs = 4 rho0 Rs^2 (1 + ln 1/2) / Sigma_crit (the UNtruncated NFW deflection
at Rs -- TNFW.alpha2rho0 explicitly "neglects the truncation"), while
PyAutoLens defines kappa_s = rho_s r_s / Sigma_crit. Both use the Baltz,
Marshall & Oguri 2009 truncation rho ~ tau^2/(tau^2 + (r/r_s)^2) with
tau = r_t/r_s and r_t given directly as an angle, so no further conversion is
needed for the truncation. Deflections agree to 1.00000 at every radius from
1e-3 Rs to 5 Rs (below that lenstronomy's own r >= 0.001 Rs clamp kicks in);
convergence agrees to <1e-4 everywhere except at exactly R = Rs, where
lenstronomy's TNFW._F takes a special-case branch (X == 1) that yields a
slightly negative kappa for one grid point -- a lenstronomy numerical
edge-case, not a physics difference, and irrelevant to ray-traced images
(deflections, not kappa, produce the image).

Full-image check: for lenses from test_fixed60 that contain a subhalo, render
macro+subhalo in both codes (pre-PSF, same row-flip handling as the macro
check) and compare (a) the subhalo-only RESIDUAL (image with subhalo minus
image without), which is exactly the quantity the detection statistic
responds to, and (b) the subhalo's deflection at the pixel where that
residual peaks, in both codes directly. Two things learned the hard way while
building this, both handled explicitly below:

1. The two codes use different absolute flux-unit conventions (lenstronomy's
   SimAPI applies the band's zero-point/exposure scaling; PyAutoLens's
   `intensity` does not) -- raw peak ratios of ~25-160x, varying per lens.
   The earlier macro check never saw this because it normalized each image by
   its own max. Here each residual is expressed as a fraction of ITS OWN
   code's total image flux, so shape AND relative amplitude are compared, and
   the raw ratio is recorded as a diagnostic rather than hidden.
2. The macro model itself does NOT agree for every lens: for some lenses the
   pure-macro (no-subhalo) images differ substantially between the codes
   (r ~ 0.5-0.7), and matching the pixel sampling (lenstronomy supersampling
   vs PyAutoLens over_sample_size) does not change that -- so it is a
   modelling-convention difference in the EPL/PowerLaw macro profile, not
   rendering numerics. That is OUT OF SCOPE here (flagged in first_results.md)
   but it would contaminate a subhalo-residual comparison, so lenses are only
   used for the subhalo test if their macro images agree (r > MACRO_MIN_CORR);
   the excluded ones are listed with their (gamma, q) so the pattern is on
   record.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import autolens as al
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.SimulationAPI.sim_api import SimAPI

LN_HALF_TERM = 1.0 + np.log(0.5)  # 0.30685...
MACRO_MIN_CORR = 0.97
MAX_SCAN = 80  # how many truths to look through when selecting lenses


def kappa_s_from_alpha_rs(alpha_rs: float, rs: float) -> float:
    return alpha_rs / (4.0 * rs * LN_HALF_TERM)


def subhalo_angles(truth):
    sh = truth["subhalo"]
    lc = LensCosmo(z_lens=truth["cosmology"]["z_lens"], z_source=truth["cosmology"]["z_source"])
    rs, alpha_rs = lc.nfw_physical2angle(M=10 ** sh["log10_M200"], c=sh["concentration"])
    return float(rs), float(alpha_rs), float(sh["tau"] * rs)


# ---------------------------------------------------------------- lenstronomy
def lenstronomy_image(truth, kwargs_band, num_pix, with_subhalo: bool):
    kwargs_band = dict(kwargs_band)
    kwargs_band["psf_type"] = "NONE"
    model_list = ["EPL", "SHEAR"] + (["TNFW"] if with_subhalo else [])
    sim = SimAPI(num_pix=num_pix, kwargs_single_band=kwargs_band,
                 kwargs_model={"lens_model_list": model_list, "source_light_model_list": ["SERSIC_ELLIPSE"]})
    im = sim.image_model_class(kwargs_numerics={"supersampling_factor": 1})
    lm, s = truth["lens_macro"], truth["source"]
    kwargs_lens = [
        {"theta_E": lm["theta_E"], "gamma": lm["gamma"], "e1": lm["e1"], "e2": lm["e2"], "center_x": 0.0, "center_y": 0.0},
        {"gamma1": lm["gamma1"], "gamma2": lm["gamma2"], "ra_0": 0.0, "dec_0": 0.0},
    ]
    if with_subhalo:
        sh = truth["subhalo"]
        rs, alpha_rs, r_trunc = subhalo_angles(truth)
        kwargs_lens.append({"Rs": rs, "alpha_Rs": alpha_rs, "r_trunc": r_trunc, "center_x": sh["x"], "center_y": sh["y"]})
    kwargs_source = [{"amp": s["amp"], "R_sersic": s["R_sersic"], "n_sersic": s["n_sersic"],
                      "e1": s["e1"], "e2": s["e2"], "center_x": s["x"], "center_y": s["y"]}]
    return im.image(kwargs_lens=kwargs_lens, kwargs_source=kwargs_source)


# ----------------------------------------------------------------- PyAutoLens
def pyautolens_profiles(truth, with_subhalo: bool):
    lm = truth["lens_macro"]
    lens_ell = al.convert.ell_comps_from(axis_ratio=lm["q"], angle=lm["phi_deg"])
    profiles = dict(
        mass=al.mp.PowerLaw(centre=(0.0, 0.0), ell_comps=lens_ell, einstein_radius=lm["theta_E"], slope=lm["gamma"]),
        shear=al.mp.ExternalShear(gamma_1=lm["gamma1"], gamma_2=lm["gamma2"]),
    )
    if with_subhalo:
        sh = truth["subhalo"]
        rs, alpha_rs, r_trunc = subhalo_angles(truth)
        profiles["subhalo"] = al.mp.NFWTruncatedSph(
            centre=(sh["y"], sh["x"]),  # PyAutoLens is (y, x)
            kappa_s=kappa_s_from_alpha_rs(alpha_rs, rs), scale_radius=rs, truncation_radius=r_trunc,
        )
    return profiles


def pyautolens_image(truth, pixel_scale, num_pix, with_subhalo: bool):
    s = truth["source"]
    grid = al.Grid2D.uniform(shape_native=(num_pix, num_pix), pixel_scales=pixel_scale, over_sample_size=1)
    src_ell = al.convert.ell_comps_from(axis_ratio=s["q"], angle=s["phi_deg"])
    lens_galaxy = al.Galaxy(redshift=truth["cosmology"]["z_lens"], **pyautolens_profiles(truth, with_subhalo))
    source_galaxy = al.Galaxy(
        redshift=truth["cosmology"]["z_source"],
        bulge=al.lp.Sersic(centre=(s["y"], s["x"]), ell_comps=src_ell, intensity=s["amp"],
                           effective_radius=s["R_sersic"], sersic_index=s["n_sersic"]),
    )
    tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])
    img = np.array(tracer.image_2d_from(grid=grid).native)
    return img[::-1, :]  # lenstronomy/PyAutoLens vertical-axis convention differ (see validate_pyautolens.py)


def subhalo_deflection_both(truth, x, y):
    """Deflection of the subhalo ALONE at sky position (x, y) in both codes."""
    sh = truth["subhalo"]
    rs, alpha_rs, r_trunc = subhalo_angles(truth)
    kw = [{"Rs": rs, "alpha_Rs": alpha_rs, "r_trunc": r_trunc, "center_x": sh["x"], "center_y": sh["y"]}]
    lax, lay = LensModel(["TNFW"]).alpha(np.array([x]), np.array([y]), kw)
    prof = al.mp.NFWTruncatedSph(centre=(sh["y"], sh["x"]), kappa_s=kappa_s_from_alpha_rs(alpha_rs, rs),
                                 scale_radius=rs, truncation_radius=r_trunc)
    d = np.array(prof.deflections_yx_2d_from(grid=al.Grid2DIrregular(values=[(y, x)])))
    return (float(lax[0]), float(lay[0])), (float(d[0, 1]), float(d[0, 0]))


# ------------------------------------------------------ isolated-profile check
def isolated_profile_check(z_l=0.5, z_s=1.0, cases=((1e10, 60.0, 20.0), (1e9, 15.0, 20.0), (1e10, 60.0, 5.0))):
    """Deflection + convergence of a lone TNFW in both codes, 1e-3 to 5 Rs along the x-axis."""
    lc = LensCosmo(z_lens=z_l, z_source=z_s)
    lm = LensModel(["TNFW"])
    out = []
    for M, c, tau in cases:
        rs, alpha_rs = lc.nfw_physical2angle(M=M, c=c)
        r_trunc = tau * rs
        ks = kappa_s_from_alpha_rs(alpha_rs, rs)
        etas = np.concatenate([np.logspace(-3, np.log10(0.04), 8), np.linspace(0.05, 5.0, 120)])
        xs = etas * rs
        prof = al.mp.NFWTruncatedSph(centre=(0.0, 0.0), kappa_s=ks, scale_radius=rs, truncation_radius=r_trunc)
        g = al.Grid2DIrregular(values=[(0.0, float(x)) for x in xs])
        pal_ax = np.array(prof.deflections_yx_2d_from(grid=g))[:, 1]
        pal_k = np.array(prof.convergence_2d_from(grid=g))
        kw = [{"Rs": rs, "alpha_Rs": alpha_rs, "r_trunc": r_trunc, "center_x": 0.0, "center_y": 0.0}]
        len_ax, _ = lm.alpha(xs, np.zeros_like(xs), kw)
        len_k = lm.kappa(xs, np.zeros_like(xs), kw)
        r_a = pal_ax / len_ax
        mask = np.abs(etas - 1.0) > 0.02  # exclude lenstronomy's X==1 branch point
        r_k = (pal_k / len_k)[mask]
        i1 = int(np.argmin(np.abs(etas - 1.0)))
        out.append({
            "M200": M, "c": c, "tau": tau, "Rs_arcsec": rs, "alpha_Rs_arcsec": alpha_rs, "kappa_s": ks,
            "eta_range": [float(etas.min()), float(etas.max())],
            "deflection_ratio_pal_over_lens": {"mean": float(r_a.mean()), "min": float(r_a.min()), "max": float(r_a.max())},
            "convergence_ratio_pal_over_lens_excluding_R_eq_Rs": {"mean": float(r_k.mean()), "min": float(r_k.min()), "max": float(r_k.max())},
            "lenstronomy_kappa_at_R_eq_Rs": float(len_k[i1]), "pyautolens_kappa_at_R_eq_Rs": float(pal_k[i1]),
        })
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--population", default="test_fixed60")
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--out", type=Path, default=Path("results/pyautolens_validation"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    pop_dir = args.data_root / args.population
    manifest = json.loads((pop_dir / "manifest.json").read_text())
    kwargs_band = manifest["kwargs_band"]
    num_pix = manifest["image_shape"][0]
    pixel_scale = kwargs_band["pixel_scale"]
    truths = [json.loads(l) for l in open(pop_dir / "truth.jsonl")]

    print("isolated-profile check (kappa_s = alpha_Rs / (4 Rs (1+ln 1/2))):")
    iso = isolated_profile_check()
    for r in iso:
        print(f"  M={r['M200']:.0e} c={r['c']} tau={r['tau']}: deflection ratio {r['deflection_ratio_pal_over_lens']['min']:.5f}-"
              f"{r['deflection_ratio_pal_over_lens']['max']:.5f} over eta {r['eta_range'][0]:.3f}-{r['eta_range'][1]:.1f}; convergence ratio (excl. R=Rs) "
              f"{r['convergence_ratio_pal_over_lens_excluding_R_eq_Rs']['min']:.5f}-{r['convergence_ratio_pal_over_lens_excluding_R_eq_Rs']['max']:.5f}")

    # ---- select lenses: has a subhalo AND the macro model agrees between codes ----
    used, excluded = [], []
    for t in truths[:MAX_SCAN]:
        if t.get("subhalo") is None:
            continue
        a_wo = lenstronomy_image(t, kwargs_band, num_pix, False)
        b_wo = pyautolens_image(t, pixel_scale, num_pix, False)
        macro_corr = float(np.corrcoef((a_wo / a_wo.sum()).ravel(), (b_wo / b_wo.sum()).ravel())[0, 1])
        rec = {"index": int(t["_index"]), "macro_corr_no_subhalo": macro_corr,
               "gamma": t["lens_macro"]["gamma"], "q": t["lens_macro"]["q"], "phi_deg": t["lens_macro"]["phi_deg"]}
        if macro_corr >= MACRO_MIN_CORR:
            used.append((t, a_wo, b_wo, rec))
        else:
            excluded.append(rec)
        if len(used) >= args.n:
            break
    print(f"\nselected {len(used)} lenses with macro-model agreement r>={MACRO_MIN_CORR}; excluded {len(excluded)} where the "
          f"pure-macro images disagree (out of scope here, see docstring):")
    for e in excluded:
        print(f"  excluded idx={e['index']}: macro r={e['macro_corr_no_subhalo']:.3f}  gamma={e['gamma']:.3f} q={e['q']:.3f}")

    rows = []
    fig, axes = plt.subplots(len(used), 3, figsize=(9.5, 3 * len(used)))
    for row, (t, a_wo, b_wo, sel) in enumerate(used):
        i = sel["index"]
        a_with = lenstronomy_image(t, kwargs_band, num_pix, True)
        b_with = pyautolens_image(t, pixel_scale, num_pix, True)

        res_a = (a_with - a_wo) / float(a_with.sum())   # fractional residual, each in its own units
        res_b = (b_with - b_wo) / float(b_with.sum())
        res_corr = float(np.corrcoef(res_a.ravel(), res_b.ravel())[0, 1])
        rms_ratio = float(np.sqrt((res_b ** 2).mean()) / np.sqrt((res_a ** 2).mean()))
        pk_a = np.unravel_index(np.argmax(np.abs(res_a)), res_a.shape)
        pk_b = np.unravel_index(np.argmax(np.abs(res_b)), res_b.shape)
        c0 = (num_pix - 1) / 2.0
        xpk, ypk = (pk_a[1] - c0) * pixel_scale, (pk_a[0] - c0) * pixel_scale
        (lax, lay), (pax, pay) = subhalo_deflection_both(t, xpk, ypk)
        rs, alpha_rs, r_trunc = subhalo_angles(t)
        rec = {
            **sel,
            "log10_M200": t["subhalo"]["log10_M200"], "concentration": t["subhalo"]["concentration"],
            "Rs_arcsec": rs, "alpha_Rs_arcsec": alpha_rs, "kappa_s": kappa_s_from_alpha_rs(alpha_rs, rs), "r_trunc_arcsec": r_trunc,
            "raw_peak_ratio_pal_over_lens_flux_units": float(b_with.max() / a_with.max()),
            "residual_pearson": res_corr,
            "residual_rms_ratio_pal_over_lens": rms_ratio,
            "residual_peak_pixel_lenstronomy_yx": [int(pk_a[0]), int(pk_a[1])],
            "residual_peak_pixel_pyautolens_yx": [int(pk_b[0]), int(pk_b[1])],
            "subhalo_deflection_at_residual_peak": {"lenstronomy_xy": [lax, lay], "pyautolens_xy": [pax, pay],
                                                    "max_abs_diff": float(max(abs(lax - pax), abs(lay - pay)))},
        }
        rows.append(rec)
        print(f"idx={i} logM={rec['log10_M200']:.2f} c={rec['concentration']:.0f} (macro r={sel['macro_corr_no_subhalo']:.4f}): "
              f"residual r={res_corr:.4f} rms ratio={rms_ratio:.4f} peaks {tuple(pk_a)} vs {tuple(pk_b)} | "
              f"subhalo deflection at peak: len=({lax:.5f},{lay:.5f}) pal=({pax:.5f},{pay:.5f})")

        ax_row = axes[row] if len(used) > 1 else axes
        v = float(max(np.abs(res_a).max(), np.abs(res_b).max()))
        ax_row[0].imshow(res_a, origin="lower", cmap="RdBu_r", vmin=-v, vmax=v); ax_row[0].set_title(f"lenstronomy subhalo residual (idx={i})", fontsize=9)
        ax_row[1].imshow(res_b, origin="lower", cmap="RdBu_r", vmin=-v, vmax=v); ax_row[1].set_title("PyAutoLens residual (row-flipped)", fontsize=9)
        ax_row[2].imshow(res_b - res_a, origin="lower", cmap="RdBu_r", vmin=-v, vmax=v); ax_row[2].set_title(f"difference, r={res_corr:.3f}, rms ratio={rms_ratio:.3f}", fontsize=9)
        for a in ax_row:
            a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Subhalo-only imprint (fraction of each code's own total flux), lenses with agreeing macro models", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out / "tnfw_comparison.png", dpi=130)
    plt.close(fig)

    summary = {
        "conversion": {
            "kappa_s": "alpha_Rs / (4 * Rs * (1 + ln(1/2)))  [alpha_Rs, Rs in arcsec]",
            "scale_radius": "Rs [arcsec]",
            "truncation_radius": "r_trunc [arcsec]; tau = r_trunc/Rs identical in both codes (Baltz+2009 form)",
            "note": "lenstronomy's alpha_Rs is the UNtruncated NFW deflection at Rs (TNFW.alpha2rho0 'neglects the truncation'); "
                    "PyAutoLens's kappa_s = rho_s r_s / Sigma_crit. Row-flip applied to PyAutoLens images as in validate_pyautolens.py. "
                    "Absolute flux units differ between codes (SimAPI band scaling vs PyAutoLens intensity); residuals compared as "
                    "fractions of each code's own total flux.",
        },
        "isolated_profile_check": iso,
        "macro_agreement_filter": {"min_corr": MACRO_MIN_CORR, "excluded_lenses": excluded,
                                   "note": "pure-macro (no subhalo) images disagree between codes for some lenses independent of pixel "
                                           "sampling -- an EPL/PowerLaw macro-model convention question, out of scope for the subhalo check "
                                           "and flagged separately in first_results.md"},
        "n_lenses_used": len(rows),
        "min_residual_pearson": float(min(r["residual_pearson"] for r in rows)),
        "residual_rms_ratio_range": [float(min(r["residual_rms_ratio_pal_over_lens"] for r in rows)),
                                     float(max(r["residual_rms_ratio_pal_over_lens"] for r in rows))],
        "max_subhalo_deflection_abs_diff_arcsec": float(max(r["subhalo_deflection_at_residual_peak"]["max_abs_diff"] for r in rows)),
        "per_lens": rows,
    }
    (args.out / "tnfw_check.json").write_text(json.dumps(summary, indent=2))
    print(f"\nused {len(rows)} lenses: min residual r={summary['min_residual_pearson']:.4f}  residual rms ratio "
          f"{summary['residual_rms_ratio_range'][0]:.4f}-{summary['residual_rms_ratio_range'][1]:.4f}  "
          f"max |delta deflection|={summary['max_subhalo_deflection_abs_diff_arcsec']:.2e} arcsec")
    print(f"wrote {args.out}/tnfw_check.json + tnfw_comparison.png")


if __name__ == "__main__":
    main()
