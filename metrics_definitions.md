# Metrics definitions

`problem_statement.md` §4.3 fixed the *categories* of metric before any result was
looked at (detection, false positives under confounders, population inference, cost,
reproducibility). This doc pins down the exact operational definition each family's
evaluation script actually implements, so that a number in `first_results.md` can be
traced back to a formula rather than re-derived from reading code each time, and so
that a new method family or a new tier can be plugged in against the same definitions
without silently drifting. Written after RQ1/RQ2/RQ4 (Tier 0) and Family D — i.e.
this documents metrics already fixed in code, rather than proposing new ones.

## 1. Detection statistic (per family)

Every family reduces one lens image to a single scalar score; "detected" always
means *score ≥ threshold*, never a family-specific rule.

| Family | Score | Where |
|---|---|---|
| **C (U-Net)** | max sigmoid(logit) over the 64×64 frame — Tsang+2024's own convention ("maximum pixel probability defines image score") | `scripts/evaluate_detector.py::score_population` |
| **A (parametric scan)** | Δχ² = χ²(smooth macro-only fit) − χ²(best subhalo found by the grid scan), i.e. the best improvement in fit found anywhere on a position×mass grid at a fixed assumed concentration | `src/baseline_a/fit.py::scan_subhalo` |
| **D (NPE)** | P(θ > `DETECT_THRESH`=7.5 \| image), estimated by drawing 300 posterior samples per image (θ = log10 subhalo mass; the no-subhalo "floor" sentinel is 6.0, so 7.5 sits between the floor and the real-subhalo prior's lower edge at 8.0) | `scripts/evaluate_baseline_d.py::score_population` |
| **B (potential correction)** | max\|δκ\| over the arc mask, where δκ = ½∇²δψ and δψ is the linear pixelized potential correction (PyAutoLens `autolens.potential_correction`, `RegularDpsiMesh(2)`, `CurvatureMask` regularization at a *fixed* λ=1e5 — chosen by rule, not per-lens evidence, because the prototype showed evidence prefers "no correction" on subhalo-free images and mis-selects on very massive ones). Macro model = the same near-truth-initialized smooth fit Family A uses (`fitted`, primary); an `oracle` variant (truth macro model) is reported only as the inversion's ceiling. Position = δκ-peak; mass proxy = aperture δκ mass at the peak. Written to `scan_results.jsonl` with `delta_chi2` as a documented alias so Family A's evaluators run unchanged. | `scripts/run_family_b.py`, prototype in `scripts/family_b_prototype.py`, `results/family_b/NOTES.md` |

## 2. Threshold calibration — always the same recipe

For every family: draw the score distribution on a **clean, zero-subhalo population**
(`no_subhalo`, disjoint seed from every test set it's compared against), and set the
threshold at the **90th percentile** of that distribution — i.e. a fixed 10% false
positive rate on the calibration set, by construction. This is Tsang+2024's own
reported operating point, chosen so cross-family numbers are read at a common,
literature-matched FPR rather than each family's own preferred cutoff.

`fpr_at_threshold_calibration_set` in every `results.json` should equal 0.10 (or the
nearest achievable value at that population's finite `n`) by construction — it is
reported as a sanity check on the calibration itself, not as a finding.

## 2b. Family A at published-style operating points (added 2026-09-11)

The 10%-FPR calibration is a *common* operating point, chosen so cross-family numbers are
comparable. For Family A it lands at a slightly negative Δχ² (−10.6 / −16.1 on the two seeds),
a bar no practitioner would use: parametric scans report Δln Z > 10 (Nightingale+2024) or
Δlog E ≥ 50 (Despali+2022). So Family A is *additionally* re-scored at absolute thresholds

```
Δχ² > 20    ≈ Δln Z > 10       Δχ² > 100   ≈ Δlog E ≥ 50
```

using Δln L ≈ Δχ²/2 and ignoring the Occam penalty (which would only make the evidence
thresholds stricter). `scripts/evaluate_baseline_a_fixed_threshold.py` reports FPR, completeness,
localization and mass error at each. Both kinds of operating point are reported in the paper;
a finding that appears only at the permissive common threshold (e.g. the count of negative-Δχ²
"detections") is labelled as such.

**Scan concentration is a switch, not a constant.** `scan_subhalo` accepts a fixed number
(15 = the original idealization; 60 = the "oracle" matching the c=60 populations) or the
Dutton & Maccio (2014) concentration–mass relation evaluated at each trial mass
(`run_baseline_a.py --concentration cm`). Mass-error results are reported per concentration
choice, because a fixed c that differs from the population's true c biases the fitted mass
by construction.

## 2c. Mass-estimate diagnostics for Family A (added 2026-09-11)

The scan's `best_log10_m` is evaluated only among *localized* detections (§ 2b criterion, ≤ 2 px).
Because that estimate turned out to be biased by ≈ −1 dex independent of the assumed
concentration, two oracle diagnostics isolate the cause and are reported alongside it:

* `scripts/oracle_position_mass_test.py` — same Δχ² statistic, same smooth fit, same c, subhalo at
  the **true position**; mass on the scan's 3-point grid and on a fine 0.25-dex grid. Reported
  among lenses with Δχ² > 20 at the true position ("signal").
* `scripts/joint_refit_mass_test.py` — subhalo at the true position and true c; (i) macro+source
  frozen at the smooth fit with a continuous free mass (the fit-then-scan shortcut), vs (ii) all
  13 macro/source parameters re-fitted jointly with the mass (what published pipelines do per grid
  cell). Reported among lenses with joint Δχ² > 20.

Both report median error (dex), n underestimated, n within 0.5 and 0.25 dex. These are
diagnostics of a mechanism, not detection metrics: they use truth the scan never has.

## 3. Completeness (RQ4: mass, concentration)

For a fixed threshold, bin the **positive-population** lenses (those with a subhalo
truly present) into six 0.5-dex mass bins from 10⁸ to 10¹¹ M☉:

```
completeness(bin) = fraction of lenses in that bin with score ≥ threshold
```

computed once at c=60 (`test_fixed60`) and once at c=15 (`test_fixed15`) — same
threshold both times, since the threshold is calibrated on the concentration-
independent `no_subhalo` population. An empty bin (n=0) reports `completeness: null`,
never a silent 0.

## 4. Matched-pairs concentration-flip test (RQ4's paired evidence)

`test_fixed60` and `test_fixed15` are generated from the **same seed**, so index *i*
in both is the identical lens system — same macro-model, source, subhalo mass and
position — differing *only* in the subhalo's assumed concentration. (For Family
A/D, which subsample `n < len(population)` for speed, pairing is recovered by
matching each record's stored `index`/`idx` field back to the original population
position, not by array position in the subsample — subsampling and/or excluded
"unreliable" fits can otherwise silently misalign the pairing.)

For every lens with a subhalo present in both:
```
detected_at_c60_not_c15 = count(detected at c=60 AND NOT detected at c=15)
detected_at_c15_not_c60 = count(detected at c=15 AND NOT detected at c=60)
```
A large, asymmetric `detected_at_c60_not_c15 : detected_at_c15_not_c60` ratio is the
signature of a genuine concentration-driven completeness collapse (as opposed to
independent per-image noise, which would flip both directions roughly equally).

## 5. Confounder false-positive rate (RQ2)

Same threshold as above (calibrated on `no_subhalo`), applied to zero-subhalo
populations that instead carry a **multipole shape confounder** at two amplitudes
(`am/θ_E = 0.01, 0.03`, m=4 — O'Riordan+2025's two test values):

```
confounder_fpr(population) = fraction of that population's (all zero-subhalo) images with score ≥ threshold
```

A confounder-sensitive method inflates this well above the 10% baseline; a method
insensitive to this particular confounder stays near 10%.

## 6. Family A's "reliable fit" exclusion

Family A's smooth (no-subhalo) fit can fail to converge to a good macro-model even
when a subhalo is genuinely absent (a `least_squares` local-minimum failure, not a
physical false positive). A fit is **excluded from every denominator** (completeness,
FPR) when:

```
chi2_smooth / num_pix**2 >= CHI2_DOF_UNRELIABLE   (= 10.0)
```

Excluded counts are reported alongside every result (`n_unreliable_fits`), never
silently dropped — an optimizer failure must not be allowed to masquerade as either
a detection or a non-detection. No other family currently has an analogous failure
mode (the U-Net always emits a score; the NPE always returns a posterior), so this
exclusion is Family-A-specific, not part of the common protocol.

## 7. Multi-seed aggregation

Every *mean ± std* reported in `first_results.md` is `values.mean()` /
`values.std(ddof=1)` (sample std, Bessel-corrected) across independent training/fit
seeds — never a bootstrap or a fitted confidence interval. n is small by design (a
pilot: 4 seeds for the U-Net, 2 for Family A and for the Tier-1 U-Net) — the std is
indicative of seed-to-seed spread at this scale, not a formal power calculation.
Report the seed count next to every mean±std; never state one without the other.

## 8. Cost

Wall-clock seconds per lens, split into whatever stages that family actually has
(Family A: `t_fit_s` + `t_scan_s` per image; Family C/D: total training wall-clock
divided by n_train, since cost there is dominated by one training run amortized over
all future inferences, not per-image). Always wall-clock (`ELAPSED`), never
cumulative CPU-seconds (`TIME`) summed across threads — the two diverge sharply
under multi-core BLAS/PyTorch parallelism, a distinction this project got wrong once
mid-session and had to correct (see `plan.md`'s decisions log).

## 9. Reproducibility ledger fields (RQ6)

Per published detector reimplemented: architecture (as specified / as guessed),
training-set size (theirs vs. ours, and why they differ), loss function (specified
or our choice), detection statistic (specified or our choice), success criterion
(specified or our choice), and the resulting absolute number next to theirs — read
as a reproducibility gap or a genuine match, with an explicit note on which
differences (usually training-set scale) most plausibly explain any gap.

## What this doc does not cover yet

Population-level inference metrics (posterior bias/width vs. truth, `tarp`
coverage, proper scoring) are specified in `problem_statement.md` §4.3 but not yet
implemented in code — Family D's evaluation here is detection-statistic-based
(§1 above), not yet a population-inference test of Σ_sub or M_hm recovery. That
remains future work, tracked in `plan.md`.
