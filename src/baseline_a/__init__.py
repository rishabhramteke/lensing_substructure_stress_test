"""Family A baseline: a fast parametric perturber scan, reimplementing the
*algorithm* of Nightingale+2024 / Despali+2022 (fit a smooth macro-model,
scan a grid for an added subhalo, threshold on a likelihood-based detection
statistic) directly on `lenstronomy` + `scipy.optimize`, rather than running
PyAutoLens's own nested-sampling pipeline.

Why not PyAutoLens directly, even though it installs cleanly here (unlike
paltas): its own papers report nested-sampling fits taking O(days) per real
lens (Nightingale+2024's 54-lens sample was itself described as a large
undertaking). At that rate, evaluating our 1,000-image test populations is
not tractable in this project's timeframe. This module keeps the same
statistical logic — a real Gaussian likelihood from lenstronomy's own noise
model (`SimAPI.estimate_noise`, not the true noise draw), a genuine
optimizer, a genuine grid scan — at a runtime budget that scales to our
populations, at two explicit costs stated up front and revisited in
`../../first_results.md`:

1. **The smooth-model fit is initialized near the true parameters** (small
   jitter), not from a blind global search — a "best-case initialization"
   idealization, not a from-scratch fit. A real pipeline's fit is harder
   than this.
2. **The subhalo scan fixes concentration** to one assumed value per run
   (matching common practice of assuming a mass-concentration relation
   rather than fitting concentration freely) and searches a coarse grid of
   (position, mass), not a continuous optimization.

**Cross-checked against real PyAutoLens (2026-09-11, `validate_pyautolens.py`)**
— not a comparison against PyAutoLens's own nested-sampling *fit* (still not
tractable at O(days)/lens), but a forward-model consistency check: build the
identical macro-lens model (EPL+shear+Sersic source, same numeric truth
parameters) in both lenstronomy and real PyAutoLens (`PowerLaw`+
`ExternalShear`+`Sersic`), render both, and compare. Found and corrected one
real coordinate-convention difference along the way: lenstronomy's image
array and PyAutoLens's `.native` array are vertical mirror images of each
other (a known axis-direction difference between the packages, not a physics
bug — confirmed by exact peak-position matching after a row-flip). On 5
lenses from `test_fixed60` (mix of has/no subhalo), pre-PSF images agree at
Pearson r = 0.94–1.00 (mean 0.979) once that flip is applied — see
`results/pyautolens_validation/`. This validates the exact forward model
Family A's `chi2_smooth` and detection statistic are built on. The TNFW
subhalo profile is now cross-checked too (`validate_pyautolens_tnfw.py`,
`results/pyautolens_validation/tnfw_check.json`): the conversion is
kappa_s = alpha_Rs / (4 Rs (1 + ln 1/2)), scale_radius = Rs,
truncation_radius = r_trunc (all arcsec; both codes use the Baltz+2009
truncation with tau = r_trunc/Rs), pinned down numerically — the two codes'
TNFW deflections agree to 1.00000 from 1e-3 Rs to 5 Rs for three (M, c, tau)
combinations, and to 5 decimals at the pixels where the subhalo's image
imprint peaks. One caveat surfaced in the process and is out of this
baseline's scope: for *elliptical* lenses the two codes' EPL/PowerLaw
Einstein-radius conventions differ by a uniform factor (0.6% at q=0.8,
1.6% at q=0.7, ~6% at q=0.5 for gamma=2; gamma-dependent once q<1), so the
macro-model agreement quoted above
is tightest for near-round, near-isothermal lenses — see first_results.md.

**Tier-1 (COSMOS source) extension (2026-09-11):** the smooth-model fit always
assumes a Sersic source (`fit.py`'s hardcoded `SERSIC_ELLIPSE`); on a Tier-1
truth there is no true Sersic shape to jitter-init near (see `fit.py::_init_vec`).
The macro-lens parameters and the source's true *position* are still used
(known exactly regardless of source type); the Sersic proxy's shape/amplitude
falls back to generic, non-truth-informed defaults. This makes Family A's
model genuinely misspecified against a Tier-1 source, on top of idealization
(1) above, not a third idealization removed — arguably a more realistic test
than Tier 0, where the fitted model family exactly matches the true source.
"""
