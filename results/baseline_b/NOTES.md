# Family B at population scale — pixelized potential correction, Tier 0 (2026-09-11)

Population run of the free-form "gravitational imaging" family (Koopmans 2005; Vegetti &
Koopmans 2009; Cao+2025's port in PyAutoLens `autolens.potential_correction`, linear δψ
inversion). Prototype, API pattern and idealizations: `results/family_b/NOTES.md`,
`scripts/family_b_prototype.py`. Driver: `scripts/run_family_b.py`; launcher
`scripts/launch_family_b.sh`; Family-B-specific summary `scripts/summarize_family_b.py` →
`summary_family_b.json`. Same five populations, same 300-lens subsamples (seed 0/0/1/2/3) and
same `index` values as Family A's `results/baseline_a`, so every comparison below is on the
identical images. Output format mirrors Family A (`scan_results.jsonl` + `manifest.json` per
population; `delta_chi2` is a documented alias of `peak_abs_dkappa`) so
`evaluate_baseline_a.py`, `evaluate_baseline_a_fixed_threshold.py` and
`plot_localization_accuracy.py --a-root results/baseline_b/fitted` ran unchanged
(`fitted/summary`, `fitted/summary_fixed_threshold`, `results/localization/*_familyB.*`).

## Design fixed before looking at results

* **Statistic** `max|δκ|` over the δψ mesh, δκ = ½∇²δψ, `RegularDpsiMesh(factor=2)` (0.16″ mesh
  pixels), `CurvatureMask` regularization at a **fixed λ = 1e5**; λ = 1e4 and 1e6 stored per lens
  as a sensitivity band. λ is never chosen per lens by evidence (prototype: evidence prefers "no
  correction" on subhalo-free images and mis-selects on very massive ones).
* **Macro model, two variants.** `fitted` (primary, the practitioner's pipeline): Family A's own
  smooth EPL+shear+Sérsic fit (`fit_smooth`, near-truth init, `maxiter=60`, same RNG → the fitted
  χ²/N is bit-identical to Family A's on 294/300 `test_fixed60` indices), rendered noiselessly by
  lenstronomy; residual = data − that render on the SNR>3 arc mask; source gradient from the
  *fitted* Sérsic parameters. Family A's reliable-fit exclusion (χ²/N < 10) applies verbatim.
  `oracle` (ceiling): lenstronomy's own noiseless no-subhalo (or no-multipole) render of the truth
  as macro model, truth source gradient, nothing excluded.
* Because the residual is lenstronomy-vs-lenstronomy in both variants, the PyAutoLens↔lenstronomy
  convention mismatch (`results/family_b/NOTES.md`) enters only through the response operator
  (deflections for ray-tracing + source gradient), not through the residual.
* Position = δκ peak (mesh world coordinates); mass proxy = Σ_crit·Σδκ·dpix² in a 2-mesh-pixel
  (0.32″) aperture at the peak; `best_log10_m` = log10 of it, floored at 6.0 when ≤ 0 (raw signed
  value kept in `aperture_mass_proxy_Msun`) so the mass-error evaluators stay numeric.
* Threshold: 90th percentile of the score on reliable `no_subhalo` (10% FPR), per variant and per λ.

## Cost (CPU only, 2 worker processes, machine shared with three other training jobs)

Fitted: 1.39 s macro fit + 0.40 s for the three-λ inversion per lens (PSF matrix preloaded across
λ), 300–540 s per 300-lens population; oracle 0.40 s per lens, ~2 min per population. All ten
population jobs: 37 min wall-clock. Errors: 0 / 3000 inversions.

## Headline (seed set 0, λ = 1e5, 10% FPR on `no_subhalo`; fitted, [oracle] in brackets)

Reliable positives per bin, fitted: 51/45/32/42/48/**13** of 52/46/32/43/62/**40** — the top
bin is dominated by the exclusion (27 of the 44 excluded `test_fixed60` fits have log10 M ≥ 10.5;
median excluded mass 10.63). Family A has the same exclusion on the same lenses.

| log10 M bin | 8.0–8.5 | 8.5–9.0 | 9.0–9.5 | 9.5–10.0 | 10.0–10.5 | 10.5–11.0 |
|---|---|---|---|---|---|---|
| completeness c=60 | 13.7 [26.9] | 17.8 [37.0] | 75.0 [90.6] | 88.1 [97.7] | 93.8 [100] | 100 (13/13) [100] |
| completeness c=15 | 11.8 [26.9] | 9.1 [47.8] | 12.5 [68.8] | 65.1 [86.0] | 71.4 [100] | 97.0 [100] |

* Threshold 0.0919 [0.0921]. Paired flip c60-only : c15-only = **58 : 10** (n = 230 pairs)
  [23 : 16, n = 275].
* Confounder FPR (m=4 multipole): a = 0.01 θ_E **39.2 %** [62.0 %], a = 0.03 θ_E **86.4 %** [92.0 %]
  (283 / 264 reliable lenses).
* Localization (detected reliable positives in `test_fixed60`, n = 134 [204]): within 2 data
  pixels (0.16″, the paper's criterion) **51.5 %** (69) [13.7 %]; within 2 mesh pixels (0.32″)
  61.2 % [21.1 %]; median offset 0.142″ [0.80″].
* Aperture-mass proxy among the 0.16″-localized detections: 2 [1] have a non-positive proxy;
  the rest: median **−0.64 dex** [−0.47], MAD 0.23 [0.14], 65/67 [25/27] low, 15 [14] within 0.5 dex.
* Regularization band at 10 % FPR (fitted): λ = 1e4 → c60 11.8/2.2/37.5/64.3/85.4/100, flip 57:10,
  confounder FPR 21.6/50.4 %; λ = 1e6 → 11.8/33.3/75.0/95.2/91.7/100, flip 40:11, FPR 63.6/97.0 %.
  Stronger smoothing buys low-mass completeness and pays for it in confounder FPR; no λ is
  confounder-robust.
* Published-style absolute operating points (fitted, `summary_fixed_threshold/results.json`;
  at n = 295 a 0.1 % FPR is not resolvable, the closest achievable is 1/295):
  `|δκ| > 0.1233` (99th pct of the null, FPR 3/295 = 1.0 %): confounder FPR 16.3 / 56.8 %, c60
  3.9/0.0/28.1/73.8/83.3/92.3 %, 94/231 detected, 62.8 % localized within 0.16″;
  `|δκ| > 0.5601` (99.9th pct, FPR 1/295 = 0.34 %): confounder FPR 0.7 / 4.5 %, c60
  0/0/0/7.1/4.2/23.1 %, 14/231 detected. The null's top two values (0.2585, 0.6856; lens 168,
  χ²/N 4.26) are macro-fit failures that passed the reliability rule — the oracle null's maximum
  is 0.1309 — so the strict operating point is set by fit error, not by noise.

## Same images, same protocol (10 % FPR on `no_subhalo`)

| family | c60 8.0–8.5 | 8.5–9.0 | 9.0–9.5 | 9.5–10.0 | 10.0–10.5 | 10.5–11.0 | flip | FPR a=0.01 | a=0.03 | loc. ≤0.16″ |
|---|---|---|---|---|---|---|---|---|---|---|
| **B fitted (λ=1e5)** | 13.7 | 17.8 | **75.0** | **88.1** | **93.8** | 100 (13/13) | 58:10 | 39.2 | 86.4 | **51.5 %** |
| B oracle (λ=1e5) | 26.9 | 37.0 | 90.6 | 97.7 | 100 | 100 | 23:16 | 62.0 | 92.0 | 13.7 % |
| A parametric (seed 0) | 11.8 | 17.4 | 37.5 | 71.4 | 85.4 | 84.6 (11/13) | 21:4 | 30.7 | 75.0 | 13.9 % |
| C U-Net (4-seed mean) | 11.7 | 8.0 | 14.7 | 28.6 | 51.0 | 63.4 | 199:30 | 11.3 | 10.5 | 56.7 % |

(A and B share the reliability exclusion and the 300-lens subsample; C is evaluated on the full
populations. A's localization and mass numbers: `results/localization/summary.json`.)

## What the numbers say

1. **Detection.** Given the same macro fit as Family A, the free-form correction is the more
   sensitive detector: it doubles completeness in the 9.0–9.5 bin (75 vs 37.5 %) and leads in
   every bin ≥ 9.0, and it beats the U-Net by 2–5× above 10^9. Below 10^9 all three are at
   the 10 % floor. The oracle ceiling adds another 10–20 points per bin.
2. **Concentration.** Not immune: c=15 completeness collapses from 75 → 12.5 % (9.0–9.5) and
   88 → 65 % (9.5–10.0); the paired flip 58:10 is the same sign and similar ratio (5.8:1) as
   Family A (5.3:1) and the U-Net (6.6:1). A diffuse (c=15) subhalo produces a lower, broader δκ
   bump, so a peak statistic is intrinsically concentration-dependent.
3. **Confounder.** The worst of the three families: 39 / 86 % FPR at 10 % nominal. A free-form
   δψ absorbs *any* smooth-model misspecification, and an m=4 multipole is a legitimate potential
   perturbation as far as the inversion is concerned; the refitted macro model absorbs part of it
   (oracle, with the multipole entirely unmodelled: 62 / 92 %). A peak-height statistic cannot tell
   a localized δκ lump from an extended m=4 pattern — a shape/extent criterion on δκ would be the
   natural fix and is not part of the published recipe we re-implemented.
4. **Localization.** 51.5 % within 0.16″ (median offset 0.14″, i.e. within one mesh pixel) is on
   par with the U-Net (56.7 %) and 3.7× Family A (13.9 %). The false alarms of Family A were
   "a distribution, not a position"; Family B's detections are positions.
5. **Mass.** The aperture δκ mass is a projected mass inside 0.32″, not M200: a tight −0.64 dex
   (MAD 0.23) offset, 65/67 low. This is a calibratable aperture bias, unlike Family A's
   frozen-macro-model bias (−1.0 dex; traced by elimination to the fit-then-scan shortcut, not to
   the fixed c=15 — see first_results.md), but the observable still comes
   out a factor ~4 low for every method tested — the "every method under-estimates the mass"
   statement extends to Family B.
6. **The oracle is a ceiling for detection, not for localization.** With the exact macro model
   the residual still contains the subhalo's long-range deflection; the first-order inversion
   spreads that over the mesh (residual χ²/N by bin 1.0/1.1/1.8/8.0/25/220 — the linear regime is
   left above ~10^10) and the δκ peak lands 0.6–1.0″ away (per-bin oracle localization 7/6/24/29/
   8/5 %). Refitting the macro model (the `fitted` pipeline) absorbs that smooth part and leaves
   the compact residual, so localization *improves* to 14/25/50/70/51/38 % on the same lenses.
   Hence the practitioner's pipeline is the primary number and the oracle stays in brackets.
7. **Fit error in the null.** On `no_subhalo` the fitted and oracle null distributions are
   indistinguishable in the bulk (medians 0.0746 vs 0.0747, per-image ratio 1.00, FPR of `fitted`
   at the oracle's threshold 9.8 %, Spearman(score, χ²/N) = +0.20); the smooth-fit error only
   shows in the two-lens tail described above. So the 10 % FPR result is noise-limited, the
   ≤ 1 % FPR results are fit-limited.
8. **Top mass bin / iterative regime.** 27 of 40 `test_fixed60` lenses with log10 M ≥ 10.5 are
   excluded by the reliability rule in `fitted`, and the oracle mislocalizes 100 % of them: for
   subhaloes ≳ 5–10 % of the lens mass the linearized δψ inversion is out of regime, exactly as
   the prototype found on idx 104. The module's iterative `IterFitDpsiSrcImaging` (re-ray-tracing
   with `InputPotential`) is the published fix and is left as future work; the completeness
   deficit it would address is bounded by 27/40 in that bin.

## Caveats (all inherited from the prototype, none removed here)

Analytic Sérsic source held fixed during the inversion (the joint `FitDpsiSrcImaging` is the full
method); one mesh factor; one fixed λ (band reported); near-truth-initialized macro fit (Family A's
idealization, shared deliberately); single plane, Gaussian PSF, Tier 0 only; oracle variant is a
single seed set. Absolute thresholds at ≤ 1 % FPR are set by two fit-failure lenses in a 295-lens
null and should be read as such.

## Replicate seed set (`results/baseline_b_seed200/fitted`, selection seeds 200/200/201/202/203)

Family A's second seed set (`results/baseline_a_seed1`) used selection seed 200 for all five
populations, so only its `test_fixed60`/`test_fixed15` subsamples coincide with this one.
`fitted` only, λ = 1e5, 10 % FPR (`results/baseline_b_seed200/summary_family_b.json`,
`.../fitted/summary/results.json`); mean ± std(ddof=1) over the two seed sets in
`results/baseline_b/aggregate_fitted_2seeds.json` (`scripts/aggregate_family_b_seeds.py`):

| | 8.0–8.5 | 8.5–9.0 | 9.0–9.5 | 9.5–10.0 | 10.0–10.5 | 10.5–11.0 |
|---|---|---|---|---|---|---|
| c=60, seed 200 | 15.2 | 23.7 | 64.3 | 88.6 | 97.4 | 100 |
| **c=60, 2-seed mean ± std** | **14.5 ± 1.1** | **20.7 ± 4.2** | **69.6 ± 7.6** | **88.4 ± 0.4** | **95.6 ± 2.6** | **100 ± 0** |
| c=15, 2-seed mean ± std | 11.3 ± 0.6 | 8.5 ± 0.8 | 20.2 ± 10.9 | 57.0 ± 11.4 | 74.1 ± 3.8 | 94.5 ± 3.4 |

Paired flip **60 ± 4 : 10 ± 1** (230 / 226 pairs); confounder FPR **40.5 ± 1.8 %** (a = 0.01) and
**86.5 ± 0.2 %** (a = 0.03); localized within 0.16″ **51.7 ± 0.2 %** (69/134, 71/137), within 0.32″
60.2 ± 1.5 %; aperture-mass error median **−0.65 ± 0.01 dex**. Threshold 0.0919 / 0.0922 — the 10 %
operating point is seed-stable. The ≤ 1 % operating point is not: the seed-200 `no_subhalo`
subsample has 16 unreliable fits (5 in seed 0) and a heavier fit-failure tail among the reliable
ones (99th pct 0.366 vs 0.123; max 2.17 vs 0.69), so at `|δκ| > p99` its completeness is
0/0/0/15.9/55.3/61.1 % against seed 0's 3.9/0/28.1/73.8/83.3/92.3 % — confirming that the strict
operating points are set by macro-fit failures, not by noise. Unreliable counts seed 200:
44/14/16/10/29. Cost 1.01 s fit + 0.43 s inversion per lens (machine less loaded than for seed 0).
