"""Cross-check Family A's forward model (lenstronomy EPL+SHEAR+SERSIC_ELLIPSE)
against real PyAutoLens's equivalent (PowerLaw+ExternalShear+Sersic), on the
same physical parameters, for a handful of lenses.

    source ~/myenv/bin/activate
    python scripts/validate_pyautolens.py --n 5 --out results/pyautolens_validation

Scope, stated plainly (see src/baseline_a/__init__.py): this is NOT a
reproduction of PyAutoLens's own published nested-sampling detection
pipeline (its own papers report O(days)/lens for that -- the entire reason
Family A exists as a fast reimplementation). It is a forward-model
consistency check -- do the two independent codebases agree on the *image*
produced by the identical macro-lens model, given identical parameters?
That's the quantity Family A's chi2_smooth and its detection statistic are
built on, so agreement here is the relevant validation for trusting those
numbers, without needing to run PyAutoLens's actual (slow) optimizer.

Compares the pre-PSF ray-traced image only (lenstronomy rendered with
psf_type="NONE"), sidestepping a second, unrelated question (do the two
codes' PSF-convolution kernels match pixel-for-pixel) that isn't what
Family A's physics claim depends on. Subhalo (TNFW) forward-model agreement
is NOT checked here: lenstronomy parameterizes the truncated-NFW profile by
(Rs, alpha_Rs, r_trunc) while PyAutoLens's NFWTruncatedSph uses
(kappa_s, scale_radius, truncation_radius) -- a real unit conversion, not
attempted in this pass -- flagged as a gap, not glossed over.

One coordinate-convention difference was found and is handled explicitly:
lenstronomy's image array and PyAutoLens's `.native` array are vertical
mirror images of each other (row 0 <-> row 63, opposite y-axis direction) --
confirmed by peak-position matching after a row-flip. This is a known
axis-convention difference between the two packages, not a physics bug; the
comparison below always flips before comparing.
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
from lenstronomy.SimulationAPI.sim_api import SimAPI


def lenstronomy_image(truth, kwargs_band, num_pix):
    kwargs_band = dict(kwargs_band)
    kwargs_band["psf_type"] = "NONE"
    kwargs_model = {"lens_model_list": ["EPL", "SHEAR"], "source_light_model_list": ["SERSIC_ELLIPSE"]}
    sim = SimAPI(num_pix=num_pix, kwargs_single_band=kwargs_band, kwargs_model=kwargs_model)
    im = sim.image_model_class(kwargs_numerics={"supersampling_factor": 1})
    lm, s = truth["lens_macro"], truth["source"]
    kwargs_lens = [
        {"theta_E": lm["theta_E"], "gamma": lm["gamma"], "e1": lm["e1"], "e2": lm["e2"], "center_x": 0.0, "center_y": 0.0},
        {"gamma1": lm["gamma1"], "gamma2": lm["gamma2"], "ra_0": 0.0, "dec_0": 0.0},
    ]
    kwargs_source = [{"amp": s["amp"], "R_sersic": s["R_sersic"], "n_sersic": s["n_sersic"], "e1": s["e1"], "e2": s["e2"], "center_x": s["x"], "center_y": s["y"]}]
    return im.image(kwargs_lens=kwargs_lens, kwargs_source=kwargs_source)


def pyautolens_image(truth, pixel_scale, num_pix):
    lm, s = truth["lens_macro"], truth["source"]
    grid = al.Grid2D.uniform(shape_native=(num_pix, num_pix), pixel_scales=pixel_scale)
    lens_ell = al.convert.ell_comps_from(axis_ratio=lm["q"], angle=lm["phi_deg"])
    src_ell = al.convert.ell_comps_from(axis_ratio=s["q"], angle=s["phi_deg"])
    lens_galaxy = al.Galaxy(
        redshift=truth["cosmology"]["z_lens"],
        mass=al.mp.PowerLaw(centre=(0.0, 0.0), ell_comps=lens_ell, einstein_radius=lm["theta_E"], slope=lm["gamma"]),
        shear=al.mp.ExternalShear(gamma_1=lm["gamma1"], gamma_2=lm["gamma2"]),
    )
    source_galaxy = al.Galaxy(
        redshift=truth["cosmology"]["z_source"],
        bulge=al.lp.Sersic(centre=(s["y"], s["x"]), ell_comps=src_ell, intensity=s["amp"], effective_radius=s["R_sersic"], sersic_index=s["n_sersic"]),
    )
    tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])
    return np.array(tracer.image_2d_from(grid=grid).native)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--population", default="test_fixed60")
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    pop_dir = args.data_root / args.population
    manifest = json.loads((pop_dir / "manifest.json").read_text())
    kwargs_band = manifest["kwargs_band"]
    num_pix = manifest["image_shape"][0]
    pixel_scale = kwargs_band["pixel_scale"]
    truths = [json.loads(l) for l in open(pop_dir / "truth.jsonl")]

    # a mix of has-subhalo / no-subhalo lenses (subhalo not itself compared,
    # see module docstring, but included so the macro-model check spans both)
    idx = []
    for t in truths:
        if len(idx) >= args.n:
            break
        if (t.get("subhalo") is not None) == (len(idx) % 2 == 0) or len(idx) < 2:
            idx.append(t["_index"])
    idx = idx[: args.n]

    rows = []
    fig, axes = plt.subplots(len(idx), 3, figsize=(9, 3 * len(idx)))
    for row, i in enumerate(idx):
        truth = truths[i]
        img_a = lenstronomy_image(truth, kwargs_band, num_pix)
        img_b = pyautolens_image(truth, pixel_scale, num_pix)
        img_b_flipped = img_b[::-1, :]

        an = img_a / img_a.max()
        bn = img_b_flipped / img_b_flipped.max()
        diff = an - bn
        corr = float(np.corrcoef(an.ravel(), bn.ravel())[0, 1])
        rec = {"index": int(i), "has_subhalo": truth.get("subhalo") is not None,
               "pearson_corr": corr, "max_abs_diff_normalized": float(np.abs(diff).max()),
               "mean_abs_diff_normalized": float(np.abs(diff).mean())}
        rows.append(rec)
        print(f"idx={i}  has_subhalo={rec['has_subhalo']}  corr={corr:.4f}  "
              f"max|diff|={rec['max_abs_diff_normalized']:.3f}  mean|diff|={rec['mean_abs_diff_normalized']:.4f}")

        ax_row = axes[row] if len(idx) > 1 else axes
        ax_row[0].imshow(an, origin="lower", cmap="inferno"); ax_row[0].set_title(f"lenstronomy (idx={i})", fontsize=9)
        ax_row[1].imshow(bn, origin="lower", cmap="inferno"); ax_row[1].set_title("PyAutoLens (row-flipped)", fontsize=9)
        im = ax_row[2].imshow(diff, origin="lower", cmap="RdBu_r", vmin=-0.3, vmax=0.3); ax_row[2].set_title(f"diff, corr={corr:.3f}", fontsize=9)
        for a in ax_row:
            a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    fig.savefig(args.out / "comparison_grid.png", dpi=130)
    plt.close(fig)

    summary = {
        "population": args.population, "n": len(rows),
        "mean_pearson_corr": float(np.mean([r["pearson_corr"] for r in rows])),
        "min_pearson_corr": float(np.min([r["pearson_corr"] for r in rows])),
        "per_lens": rows,
        "scope_note": "pre-PSF macro-model (EPL+shear+Sersic) forward-model comparison only; "
                       "subhalo/TNFW forward model not cross-checked (kappa_s vs alpha_Rs "
                       "parameterization conversion not attempted); not a comparison of "
                       "PyAutoLens's own published nested-sampling fit.",
    }
    (args.out / "results.json").write_text(json.dumps(summary, indent=2))
    print(f"\nmean corr={summary['mean_pearson_corr']:.4f}  min corr={summary['min_pearson_corr']:.4f}")
    print(f"wrote {args.out}/results.json + comparison_grid.png")


if __name__ == "__main__":
    main()
