# Family B prototype — pixelized potential correction with PyAutoLens (2026-09-11)

Single-lens feasibility for the free-form "gravitational imaging" family (Koopmans 2005;
Vegetti & Koopmans 2009; Cao et al. 2025's implementation as ported into
`autolens.potential_correction`, PyAutoLens 2026.9.8.1). Script: `scripts/family_b_prototype.py`.
Figures/JSON: `results/family_b/prototype_idx{104,332,451,188}.png|.json`, all runs merged in
`prototype.json` (key = `idx<i>`; `idx332_pyautolens_f2` is the cross-code-smooth-model variant).
**No population run was launched.** Everything below is one noise draw (seed 7) per lens.

## Working API pattern (the linear route, no nonlinear search)

```python
import autolens as al, autoarray as aa
from autolens.potential_correction import FitDpsiImaging, DpsiPixelization, RegularDpsiMesh, AnalyticSrcFactory
from autolens.potential_correction import util as pc_util

psf  = al.Convolver.from_gaussian(shape_native=(11,11), pixel_scales=0.08, sigma=0.08/2.3548, normalize=True)  # unnormalized by default!
img  = al.Imaging(data=al.Array2D.no_mask(data[::-1,:], 0.08), noise_map=al.Array2D.no_mask(noise[::-1,:], 0.08), psf=psf)
mask = al.Mask2D(mask=pc_util.arc_mask_from(snr[::-1,:], threshold=3.0, ignore_size=25, ext_size=5), pixel_scales=0.08)
mi   = img.apply_mask(mask=mask)                                  # 64x64 must be divisible by the mesh factor
residual = mi.data.slim - smooth_model_slim                       # data - blurred macro model, on unmasked pixels
src_pos  = mi.grid.slim - lens_galaxy.deflections_yx_2d_from(grid=mi.grid.slim)
src_grad = AnalyticSrcFactory(source_galaxy).eval_grad(src_pos[:,1], src_pos[:,0])   # (dS/dy, dS/dx)
pix  = DpsiPixelization(mesh=RegularDpsiMesh(factor=2), regularization=aa.reg.CurvatureMask(coefficient=lam))
fit  = FitDpsiImaging(masked_imaging=mi, image_residual=residual, source_gradient=src_grad, dpsi_pixelization=pix,
                      preloads={"psf_mat": previous_fit.psf_mat})  # PSF matrix depends only on the mask; reuse
dpsi   = fit.solve_dpsi();  logZ = fit.log_evidence
dkappa = 0.5 * fit.pair_dpsi_data_obj.hamiltonian_dpsi @ dpsi      # module's visualize.py omits the 1/2
xg, yg = fit.pair_dpsi_data_obj.xgrid_dpsi_1d, fit.pair_dpsi_data_obj.ygrid_dpsi_1d   # mesh world coords, dpix 0.16"
```

Gotchas found: lenstronomy and PyAutoLens `.native` arrays are vertical mirror images (flip rows once;
world coordinates then agree, PyAutoLens is (y, x)); `al.Kernel2D` no longer exists (PSF is a
`Convolver`); `al.mesh.Rectangular` is `RectangularUniform`; flux units differ by ~155x
(PyAutoLens→lenstronomy scale 0.0064 fitted linearly); PyAutoLens's smooth model rebuilt at the
true parameters does **not** match lenstronomy's noiseless image at the noise level
(chi2/pixel 0.95–12 for q≈0.98–0.999, **52.6** for q=0.955: EPL-vs-PowerLaw theta_E and
PSF/oversampling conventions) — so the default prototype uses lenstronomy's own noiseless
no-subhalo image as the macro model ("oracle" idealization) and only the source *gradient* comes
from PyAutoLens. The `--smooth pyautolens` variant is kept for honesty.

## Does dkappa appear where the subhalo is?

Mesh factor 2 (0.16" pixels, 400–800 dpsi pixels for 1700–3400 data pixels). Values at the fixed
coefficient lam=1e5 (see below for why fixed) unless noted; "twin" = same lens, same noise, no subhalo.

| idx | log10 M | M_sub / M(<θE) | signature peak (σ/pix) | peak \|δκ\| sub / twin | peak offset from true subhalo | corr(δκ, true κ) | aperture mass at true position: recovered / true |
|---|---|---|---|---|---|---|---|
| 104 | 10.67 | 12.2 % | 358 | 2.49 / 0.098 | 0.39" (0.75" at max-evidence lam=100) | 0.11 | −2.4e9 / 8.3e9 (fails) |
| 332 | 10.28 | 6.2 % | 35 | 0.570 / 0.073 | **0.06"** (< 1 mesh pixel) | 0.59 | 4.1e9 / 4.35e9 |
| 451 | 9.56 | 0.9 % | 139 (sits on a cusp knot) | 0.248 / 0.088 | 0.55" | 0.20 | 2.2e9 / 1.1e9 |
| 188 | 9.00 | 0.3 % | 4 | 0.097 / 0.078 | **0.11"** | 0.17 (0.43 at lam=1e6) | 5.3e8 / 3.9e8 |

Appearance: for idx 332 and 188 the δψ map is a smooth potential well centred on the subhalo and
δκ a compact positive lump on the subhalo position; the twin's δψ has only edge structure and its
δκ is featureless noise (max-evidence coefficient on every twin is the largest tried, i.e. the
evidence prefers *no* correction, as it should). For idx 451 the δψ well is again centred on the
subhalo but δκ is a ± dipole/ringing pattern along the bright arc just below it (Laplacian of a
sharply peaked well at 0.16" resolution) — the peak is 3 mesh pixels off. For idx 104 the linear
approximation fails outright: a 12%-of-lens-mass subhalo shifts the whole ring by ~0.7 pixel,
the residual is 358σ/pixel, chi2/N only drops 418→8.8, and δψ/δκ are a checkerboard with
|δκ| ≈ 20 and zero correlation with the truth (the twin's inversion is clean). This is the regime
where the module's iterative `IterFitDpsiSrcImaging` (re-ray-tracing with `InputPotential`) or
Family A is the right tool, not the first-order inversion.

Cross-code smooth model on idx 332 (`--smooth pyautolens`): still localizes (0.06"), but the
convention-mismatch residual (chi2/N 1.9 → 2.9 on the twin's residual) raises the twin's
peak |δκ| from 0.23 to 0.37 at lam=1e4 — i.e. a macro-model error goes straight into the null
distribution of the statistic. In a real pipeline the macro model comes from a fit, so this is a
feature of the family, not a bug of the prototype.

## Regularization sensitivity (the Galan+2022 caveat, confirmed)

CurvatureMask coefficient swept over 10 … 1e6 (a 1e-3–10 sweep first did nothing: the data term
dominates because the noise map is ~1e-2 in these flux units). Twin peak |δκ| falls from ≈5–6
(lam=10, pure noise amplification) to 0.02–0.04 (lam=1e6): two orders of magnitude. The subhalo
lump only localizes for lam ≥ 1e4 (idx 332: offset 0.91"→0.06" at 1e4; idx 188: 2.19"→0.11" at
1e5; idx 451: 2.40"→0.44" at 1e6), while the aperture mass keeps growing with lam (idx 332:
2.9e9 → 4.8e9 over 1e3 → 1e6, true 4.35e9). Max-evidence coefficient on the subhalo image: 1e2
(104), 1e4 (332), 1e5 (451), 1e6 = "no correction" (188). So the coefficient cannot be left per
image without also making the score depend on the selection rule.

## Recommended detection statistic (compatible with `metrics_definitions.md` §1–2)

`score = max |δκ|` over the dpsi mesh at a **fixed** CurvatureMask coefficient lam = 1e5, mesh
factor 2, arc mask from `arc_mask_from(SNR>3)`, macro model = the pipeline's smooth fit; the
threshold is the 90th percentile of `score` on the `no_subhalo` population, exactly as for
Families A/C/D. On these four single draws the ratio sub/twin at lam=1e5 is 25, 7.8, 2.8, 1.2
(log10 M 10.67, 10.28, 9.56, 9.00). Secondary candidates worth carrying in the population run:
(i) `max_lam logZ − logZ(lam_max)` ("evidence for any correction": 31700, 890, 77, 0 on the
subhalo images vs 0 on every twin — Bayesian, but coarse on a 6-point grid); (ii) the peak of
the smoothed δκ or of −δψ, which is more stable than the raw Laplacian for cusp cases like 451;
(iii) integrated δκ × dpix² × Σ_crit in a 2-mesh-pixel aperture as the mass estimate for the
RQ4 mass-error comparison (within a factor ~2 of the true aperture mass for the three
linear-regime lenses here). Localization criterion: peak within 2 mesh pixels (0.32") of truth.

## Cost

Setup (render twin pair, noise map, mask): 0.4–0.8 s. First `FitDpsiImaging` on a lens
(builds the PSF matrix): 0.1–0.9 s (0.3 s typical; 17 s once on the very first call of the session,
presumably numba compilation). Each further coefficient / twin with `preloads={"psf_mat": …}`:
0.1–0.7 s (scales with unmasked-pixel count, 1700–3400). A 6-value sweep on both twins is
~2–4 s per lens with OMP_NUM_THREADS=2, so a 1000-lens population at one fixed coefficient is
minutes, not hours — **excluding** the macro-model fit the pipeline must run first (Family A's
`t_fit_s`) and excluding the joint source+dpsi inversion / LM iterations of the full method.

## Feasibility verdict

Feasible for the paper as a fourth family under the stated idealizations: the published
implementation runs on our images, recovers a localized δκ at the true subhalo for
log10 M ≈ 9.0–10.3 on the ring and returns a featureless map on the noise-sharing twin, at
seconds per lens. Two things the population run must handle: (1) the coefficient must be fixed by
rule (and the sensitivity reported), (2) the most massive Tier-0 subhaloes (≳ 5–10 % of the lens
mass, i.e. the upper mass bins where the U-Net and Family A are complete) break the first-order
inversion — run `IterFitDpsiSrcImaging` there or report the linear failure as a finding. Neither
fallback (herculens; in-house δψ inversion) is needed.

Idealizations (all stated in the script docstring): oracle macro model (lenstronomy's own
noiseless no-subhalo image); source held fixed at the analytic truth during the inversion (the
joint `FitDpsiSrcImaging` is the full method); single plane; one mesh factor; single noise draw
per lens; first-order linearization in δψ.
