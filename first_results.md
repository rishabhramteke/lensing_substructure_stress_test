# First results — U-Net detector v0 (2026-09-10)

The first quantitative pass at the stress test, built on the Tier-0 simulator (`src/lensing/`)
and a from-scratch U-Net reimplementation of Ostdiek+2020/22 / Tsang+2024 (`src/detector/`,
RQ6 — neither paper released code). Everything below is reproducible from
`checkpoints/unet_v0/` + `results/detector_v0_best/` + `data/*/manifest.json`.

**Scope, stated up front:** one architecture, 8,000 training images per run
(Tsang+2024 used 5×10⁵ — a ~60x smaller budget, a deliberate compute/time choice, not an
attempt to match their scale), **n=4 independent training seeds** (0–3, same data/config,
different weight init and train/val split each time) so the numbers below carry an actual
seed-to-seed uncertainty rather than one run's luck. n=4 is a pilot, not a powered study —
read every std as indicative. Aggregation: `scripts/aggregate_seeds.py` -> `results/aggregate/`.

## Setup

- **Train:** `data/train_fixed60` — 8,000 images, Tsang+2024's operating point (EPL+shear,
  single-Sérsic source, TNFW subhalo at fixed c=60, 90% presence rate), HST/F160W/0.08″/Gaussian PSF.
- **Model:** small U-Net (`src/detector/unet.py`, base width 16), per-pixel BCE loss
  (pos_weight=150 for the ~1000:1 pixel imbalance), Adam, 25 epochs, batch 64, MPS (Apple GPU).
- **Checkpoints:** `model_best.pt` (lowest val loss, **epoch 16**, val_loss=0.368) and
  `model_last.pt` (epoch 25, val_loss=0.453 — visibly overfit; kept only as a robustness check).
- **Detection score:** max(sigmoid(logits)) over the frame — Tsang+2024's own convention.
  Threshold calibrated for 10% FPR on `data/no_subhalo` (1,000 clean negatives), then applied
  unchanged to every other population below.
- **Test sets:** `test_fixed60` / `test_fixed15` (1,000 each, **same seed → index-aligned,
  identical lens/source/subhalo mass & position, concentration is the only difference** — a
  matched-pairs design, not independent samples) for RQ4; `no_subhalo`, `multipole_m4_a1`,
  `multipole_m4_a3` (1,000 each, independent seeds) for RQ2.

## RQ4 — does completeness collapse at realistic concentrations? Yes, sharply, across every seed.

![completeness vs mass, 4-seed aggregate](results/aggregate/completeness_vs_mass_aggregate.png)

| mass bin (log₁₀ M☉) | completeness, c=60 (mean ± std, n=4) | completeness, c=15 (mean ± std, n=4) |
|---|---|---|
| 8.0–8.5 | 11.7% ± 5.0% | 9.3% ± 2.7% |
| 8.5–9.0 | 8.0% ± 1.1% | 8.7% ± 0.8% |
| 9.0–9.5 | 14.7% ± 3.8% | 12.7% ± 4.0% |
| 9.5–10.0 | 28.6% ± 7.8% | 11.0% ± 2.8% |
| 10.0–10.5 | 51.0% ± 10.1% | 15.4% ± 4.4% |
| 10.5–11.0 | 63.4% ± 4.2% | 13.2% ± 3.1% |

At c=60 completeness rises monotonically with mass in every one of the 4 seeds. At c=15 — the
concentration Tsang+2024 describe as "expected from ΛCDM simulations" — completeness stays
flat around 9–15% at *every* mass bin, with a std small enough that above 9.5 dex the two
curves are separated by several times their combined uncertainty (e.g. at 10.0–10.5: 51.0±10.1%
vs 15.4±4.4%). This robustly reproduces Tsang+2024's own description of their c=15 ablation:
"around 0.1 one would expect from random guessing."

**The matched-pairs test, averaged over all 4 seeds, is the cleanest evidence.** Of the 905
lenses in the paired test set that contain a subhalo, the *same* lens/source/subhalo (mass,
position identical) flips from detected to missed when concentration alone drops from 60 to
15 in **198.8 ± 41.2** cases, versus only **30.2 ± 16.8** flipping the other way — a **6.6:1**
ratio of means, in the same direction in every single seed, on identical images differing in
exactly one physical quantity.

**Reading it:** this is a real, independently-obtained, now properly-powered reproduction of
the field's own best-documented failure mode (Family C, §1 of `fulltext_findings.md`) — on a
detector we wrote from the paper's text, at 1.6% of their training-set size. That it
reproduces this consistently across 4 independent training runs at much smaller scale is
itself informative: the concentration sensitivity looks like a property of the *signal*, not
an artifact of a particular group's training budget or a lucky/unlucky training run.

### Scale-up check (self-review fix, 2026-09-11): 30,000 training images, 2 seeds

The referee objection: an 8,000-image detector at AUC 0.62 is too weak for its failures to
say anything about the published detector. So: `data/train_fixed60_30k` (n=30,000, seed 500,
3.75× the original), `checkpoints/unet_30k_seed{0,1}` (same recipe, 25 epochs, ~15 min/seed on
MPS, best val loss 0.280/0.277 vs 0.368 at 8k), evaluated on the identical Tier-0 test
populations; `results/aggregate_30k/` via `scripts/aggregate_unet_seeds.py`.

| | 8,000 images (n=4) | 30,000 images (n=2) |
|---|---|---|
| AUC (c=60 vs no subhalo) | 0.619 ± 0.018 | **0.666 ± 0.006** |
| completeness c=60, 9.0–9.5 / 9.5–10 / 10–10.5 / 10.5–11 | 14.7 / 28.6 / 51.0 / 63.4% | **20.6 / 50.8 / 69.7 / 79.1%** |
| completeness c=15, same bins | 12.7 / 11.0 / 15.4 / 13.2% | 9.9 / 16.7 / 13.2 / 27.8% |
| paired flip, c60→missed at c15 vs reverse | 198.8 ± 41.2 vs 30.2 ± 16.8 (6.6:1) | **284.5 ± 7.8 vs 42.0 ± 5.7 (6.8:1)** |
| multipole FPR, a=0.01 / 0.03 | 11.3 ± 2.1 / 10.5 ± 1.9% | **9.9 ± 0.3 / 10.1 ± 0.5%** |
| Tsang+2024's 33.6% at 10⁹–10⁹·⁵ | 11.8% | 20.6% |

**Reading it:** more data made a materially better detector *at its operating point* (the gap to
Tsang's published number roughly halved with a fraction of their data, consistent with the
residual gap being training scale) and **no more robust away from it**: the c=15 collapse is
unchanged (a stronger detector loses *more* detections when concentration drops — 6.8:1), and the
RQ2 null is even tighter. This is the pattern expected if the concentration collapse is a
property of the signal, not of an under-trained network — the referee's alternative explanation
is now tested and does not hold. The 8k model remains the primary throughout (n=4 seeds; the
localization, decoy and Tier-1 comparisons were all run with it); the 30k runs are reported as
the scale-up check (paper Fig. "Scale-up check").

## RQ2 — does a shape confounder inflate false positives? Not detectably, at n=4 seeds.

![confounder FPR, 4-seed aggregate](results/aggregate/confounder_fpr_aggregate.png)

| condition | mean FPR (n=4 seeds) | std | resolved from baseline at 2σ? |
|---|---|---|---|
| `no_subhalo` (the calibration set itself) | 10.0% | 0.0% | — |
| multipole m=4, a=0.01·θ_E | 11.3% | 2.1% | **no** |
| multipole m=4, a=0.03·θ_E | 10.5% | 1.9% | **no** |

With 4 independent training seeds, both multipole confounders sit within about 0.6–0.8 seed-
standard-deviations of the 10% baseline — nowhere near separated. **This is a genuine null
result at the precision this pilot can measure**, not an unresolved single-seed fluctuation:
no dramatic inflation like the Bayes-factor collapses Lange+2024 and O'Riordan+2025 report
**for parametric evidence-based scans (Family A/D)**, on the same style of confounder.

**The likely reason, still a hypothesis, now on firmer footing:** a segmentation U-Net is
trained to recognize a spatially *localized* dipole (see `data/sanity/panel_A_subhalo.png`),
while an m=4 multipole perturbs the arc *globally and symmetrically*
(`data/sanity/panel_C_multipole.png`) — a shape a max-pooled local detector may simply not
respond to the way a global-evidence comparison does. If that holds up under the Family-A/D
baselines still to come, it would be a genuine, non-obvious cross-family finding:
**confounder sensitivity may be method-family-specific**, not a universal property of "ML
detectors" or "strong lensing" — exactly the kind of result this whole stress test exists to
produce (RQ1). One caveat this pilot can't rule out: a *larger* multipole amplitude, or a
different order (m=3), might still confound this detector family — the two amplitudes tested
here are O'Riordan's own choices for a parametric-evidence method, not necessarily where a
spatially-local detector's sensitivity (if any) would peak.

## RQ1 — cross-family comparison: does a parametric scan (Family A) and a spatially-local detector (Family C) respond to the same confounder the same way? No — dramatically not.

**Method, in brief** (full design in `src/baseline_a/__init__.py`): fit a smooth EPL+shear+Sérsic
model near the true parameters via least-squares, then grid-scan a fixed-concentration
(c=15) NFW subhalo over position and mass, taking the largest Δχ² over the grid as the
detection statistic — the same fit → scan → threshold logic as Nightingale+2024 /
Despali+2022, run directly on `lenstronomy` rather than PyAutoLens's own nested-sampling
pipeline (which the papers describe as an O(days)-per-lens undertaking — intractable at
our population sizes). **Two idealizations are baked in and matter for how to read every
number below:** the smooth fit starts near the truth rather than from a blind global search,
and the scan always assumes concentration = 15 regardless of the subhalo's true value.

**n=2 independent seeds (0 and 200, 300 images each, fresh random subsamples and their own
per-seed calibration threshold), added 2026-09-11 specifically to check the single-run numbers
weren't a fluke of one particular 300-image draw** — `scripts/aggregate_baseline_a_seeds.py`.
Still far short of the U-Net's 4×1,000; read the std columns as "this wasn't a fluke," not as
a tight confidence interval.

**Unreliable fits** (chi2/dof ≥ 10, excluded, reported not hidden), seed 0 → seed 200:
test_fixed60 43→44/300, test_fixed15 16→14/300, no_subhalo 5→14/300, multipole a=0.01 17→20/300,
multipole a=0.03 36→31/300 — consistently in the 5–15% range across both draws.

### RQ4 revisited for Family A — a different mechanism, not the same collapse

![Family A completeness vs mass](results/baseline_a/summary/completeness_vs_mass.png)

| mass bin | completeness, true c=60 (mean ± std, n=2) | completeness, true c=15 (mean ± std, n=2) |
|---|---|---|
| 8.0–8.5 | 12.4% ± 0.9% | 10.3% ± 0.8% |
| 8.5–9.0 | 23.2% ± 8.2% | 19.8% ± 5.5% |
| 9.0–9.5 | 36.6% ± 1.3% | 23.0% ± 2.9% |
| 9.5–10.0 | 68.7% ± 3.9% | 56.6% ± 1.2% |
| 10.0–10.5 | 91.4% ± 8.5% | 79.4% ± 6.2% |
| 10.5–11.0 | 84.0% ± 0.9% | 90.2% ± 1.0% (small-n bin both seeds, n=13–33 — don't over-read the crossover) |

**Family A does *not* reproduce the U-Net's collapse to chance level at c=15, and this now holds
across both seeds, not just one.** Both curves rise together with mass in every draw; c=15
tracks a few points behind c=60 rather than flatlining at 10%. The reason is structural, not a
contradiction of the RQ4 finding: **Family A's scan always assumes c=15**, whatever the
subhalo's *true* concentration is. Against a true c=60 (compact) subhalo, the assumed-c=15
(diffuse) template is the wrong shape but still absorbs much of the total deflection near the
arc — lensing constrains projected mass within an aperture far more robustly than it constrains
the internal profile shape at HST pixel scale/noise, a real, known degeneracy, not a bug.
Against a true c=15 subhalo, the template is simply correct.

**This is itself the finding, not a null result:** a parametric method's concentration
sensitivity is a *modeling choice* — pick a diffuse assumed profile and you stay sensitive to
diffuse subhalos — while an ML detector's concentration sensitivity is *baked into its
training distribution* and can't be changed without retraining. Family A and Family C fail
*differently* at the same physical question — exactly the kind of family-specific behavior
RQ1 exists to surface.

**The matched-pairs test, both seeds:** of the ~228 lenses (paired test set, subhalo-present,
reliable fits) per seed, **21 flipped from detected to missed going c=60→c=15 in *both* seeds
exactly**, versus only 4 and 1 flipping the other way — a ~8.4:1 ratio, smaller than the
U-Net's 189:25 (~7.6:1 on 905 pairs, so actually a comparable ratio at 1/4 the sample), but in
the same direction both times.

### RQ2 revisited for Family A — the confounder that didn't touch the U-Net devastates this one, confirmed on a second, independent draw

| condition | Family A FPR, mean ± std (n=2 seeds) | Family C / U-Net FPR (mean ± std, n=4 seeds × 1,000) |
|---|---|---|
| `no_subhalo` (calibration set) | 10.15% ± 0.02% | 10.0% ± 0.0% |
| multipole m=4, a=0.01·θ_E | **32.7% ± 2.8%** | 11.3% ± 2.1% |
| multipole m=4, a=0.03·θ_E | **76.0% ± 1.4%** | 10.5% ± 1.9% |

**At the stronger amplitude, three in four zero-subhalo images get flagged as containing a
subhalo by the parametric scan — while the U-Net's false-positive rate never moved — and this
is now confirmed on a completely independent 300-image draw with a tight std (±1.4 points on a
66-point effect).** This was the single most important number from the first pass to
re-check, precisely because it was the headline result; it survives. This is the RQ1
hypothesis, confirmed twice: a global evidence-based comparison (fit the whole image with and
without an extra mass component, compare goodness-of-fit) is exactly the kind of statistic a
smooth, ring-wide angular distortion can fake, because "the fit got better everywhere" doesn't
distinguish "there's a compact extra mass" from "the galaxy isn't a perfect ellipse." A
max-pooled local detector, looking for one small compact feature, doesn't have that failure
mode available to it. This lines up with the literature's own reports for this same method
family — Lange+2024's Bayes factor falling from 60 to 11 once multipoles were fit, Nightingale
+2024 tracing 8 of 54 real candidate detections to an overly simple mass model — now measured
side by side against a method family that doesn't show it, on identical images, twice.

**Caveats specific to this number, stated plainly:**
- **This may still be a lower bound, not an upper bound, on the true confounder sensitivity.**
  The smooth fit is initialized *near the true (confounder-free macro-model) parameters* —
  a real pipeline doing a blind fit has more freedom to wander into a worse local optimum,
  which could make the confounder problem *worse* in practice, not better.
- **Excluded "unreliable" fits are an open question in either direction** — 10–15% of images
  per seed excluded as non-converged; some could plausibly be additional runaway false
  detections.
- **n=2 seeds** confirms the effect is real and roughly this size, not a single-draw fluke;
  it does not yet bound it tightly the way the U-Net's n=4 does.

**Bottom line:** on the concrete question this whole project exists to answer — does
confounder sensitivity depend on which method family you use — the answer, confirmed on two
independent draws, is an unambiguous **yes**.

### Cross-check: does Family A's forward model agree with real PyAutoLens? (2026-09-11)

Family A's whole RQ1 result rests on its `chi2_smooth`/`delta_chi2` statistic, which in turn
rests on lenstronomy's EPL+SHEAR+SERSIC_ELLIPSE forward model matching what the field's actual
standard tool (PyAutoLens) would compute for the same lens. That was asserted, not checked,
until now. `scripts/validate_pyautolens.py` builds the identical macro-lens model (same numeric
truth parameters) in both lenstronomy and real PyAutoLens (`PowerLaw`+`ExternalShear`+`Sersic`,
installed and working here — version 2026.9.8.1) and compares the rendered images directly.

**This is a forward-model consistency check, not a reproduction of PyAutoLens's own published
fit** — that still takes O(days)/lens by its own papers' account, which is the entire reason
Family A exists. It answers a narrower but still important question: is the physics the same?

Found one real thing worth knowing about along the way: lenstronomy's image array and
PyAutoLens's `.native` array are vertical mirror images of each other — a known axis-direction
convention difference between the two packages (confirmed by exact peak-position matching after
a row-flip), not a bug in either. Once that's accounted for, on 5 lenses from `test_fixed60`
(mix of has/no subhalo), the pre-PSF images agree at **Pearson r = 0.94–1.00 (mean 0.979)** —
see `results/pyautolens_validation/comparison_grid.png` for the visual (same ring geometry, same
brightness clumps at the same positions across all 5, residuals confined to sharp-edge
sub-pixel discretization noise). **This confirms Family A's forward model is physically
consistent with the field's standard tool**, which is what its detection statistic depends on.

Not checked: the TNFW subhalo profile's forward model. lenstronomy parameterizes truncated NFW
as (Rs, alpha_Rs, r_trunc); PyAutoLens's `NFWTruncatedSph` uses (kappa_s, scale_radius,
truncation_radius) — a genuine unit conversion between the two, not attempted in this pass.
The macro-model check above is still the more load-bearing one, since it's what the smooth fit
(and therefore every `delta_chi2`) is built on; the subhalo-scan forward model is a smaller,
flagged gap for anyone extending this validation.

**TNFW update (2026-09-11):** the gap in the paragraph above is now closed
(`scripts/validate_pyautolens_tnfw.py`, `results/pyautolens_validation/tnfw_check.json`,
`tnfw_comparison.png`). The conversion, derived from both codes' source and then pinned down
numerically, is **κ_s = α_Rs / [4·Rs·(1+ln½)]**, `scale_radius = Rs`, `truncation_radius =
r_trunc` (all arcsec; both codes use the Baltz+2009 truncation with τ = r_trunc/Rs, so nothing
further is needed). Lenstronomy's α_Rs is the *untruncated* NFW deflection at Rs
(`TNFW.alpha2rho0` "neglects the truncation"); PyAutoLens's κ_s = ρ_s r_s/Σ_crit. **Result: the
two codes' TNFW deflections are the same function** — ratio 0.99999–1.00000 from 10⁻³ Rs to
5 Rs for (M, c, τ) = (10¹⁰, 60, 20), (10⁹, 15, 20), (10¹⁰, 60, 5), and identical to 5 decimals
(max |Δα| = 2×10⁻¹⁷″) at the exact pixels where each subhalo's image imprint peaks. Convergence
agrees to <10⁻⁴ everywhere except the single grid point at R = Rs, where lenstronomy's `TNFW._F`
special-case branch returns a slightly negative κ — a lenstronomy numerical edge case, not physics,
and irrelevant to imaging (deflections make the image). At image level, comparing the
subhalo-only imprint (image with minus without, each as a fraction of its own code's total flux)
on 5 lenses whose macro models agree (r ≥ 0.97): residual r = 0.9997, 0.998, 0.986 for three, 0.88
for one whose macro agreement is weakest (r = 0.984), 0.64 for the weakest subhalo tried (10⁹·¹⁵ M☉,
imprint peaks on adjacent pixels); imprint RMS ratios 0.94–1.27 throughout — amplitude right to
~±25%, no order-of-magnitude issue. **The subhalo forward model is confirmed consistent.**
Two things this check surfaced that the earlier macro check could not: (1) the codes differ in
absolute flux units (raw peak ratios 25–160×, per lens — SimAPI applies the band's zero-point/
exposure scaling, PyAutoLens's `intensity` does not), invisible before because each image was
normalized by its own max; irrelevant to Family A, which lives entirely in lenstronomy's units.
(2) **The macro-model agreement above is tightest for near-round, near-isothermal lenses and is
not universal**: 4 of the first 9 subhalo lenses had to be excluded (idx 3, 4, 6, 8; pure-macro
r = 0.64, 0.30, 0.95, 0.96), and matching the pixel sampling on both sides does not change that.
A controlled test isolates the cause to a **uniform Einstein-radius normalization convention for
elliptical lenses** between lenstronomy's `EPL.theta_E` and PyAutoLens's
`PowerLaw.einstein_radius`: circular lenses agree to 1.0000 at every slope γ ∈ [1.5, 2.5], while
at γ = 2 the deflection ratio falls to 0.9986 (q = 0.9), 0.9843 (q = 0.7), 0.9428 (q = 0.5), and
becomes γ-dependent once q < 1 (1.076 at γ = 1.5, q = 0.7; 0.900 at γ = 2.5, q = 0.7) — spatially
uniform, shape correlation 1.00000 in every case. A uniform 2–10% deflection rescale moves the
ring radius, which is what collapses image correlation for source-on-arc configurations. This is a
*convention* difference (same physics, differently-normalized θ_E), not a bug in either code, and
it does not touch Family A's internal consistency — but the "r = 0.94–1.00" figure above should be
read as "identical physics, up to a documented θ_E-convention factor that grows with ellipticity,"
not as pixel-level equivalence at arbitrary q. Out of scope to resolve here; flagged for the write-up.

### Real-world sanity check: running Family A on an actual published lens, not a simulated one (2026-09-11)

Every result above uses our own simulated images. `scripts/validate_real_lens.py` asks a
different question: applied to a **real** telescope image of a **real, previously-published**
lens system, does Family A's fit+scan land anywhere near what an independent, peer-reviewed
analysis concluded? There is no ground truth for a real lens (see the "real lens catalogs"
discussion above) — this is not a grading exercise, it's a check on whether our code behaves
sensibly outside the synthetic world it was built and tuned in.

**Target:** JVAS B1938+666, via Şengül, Dvorkin, Ostdiek & Tsang 2022 ("Substructure Detection
Reanalyzed: Dark Perturber shown to be a Line-of-Sight Halo," arXiv:2112.00749, MNRAS 515,
4391) — a real system already in this project's own literature review
(`fulltext_findings.md`), whose whole point is that a previously-claimed subhalo detection is
more likely a line-of-sight halo, the same degeneracy flagged as this pilot's item 8 above.
Their code and the real HST/NICMOS F160W drizzled image are public
(`github.com/acagansengul/interlopers_with_lenstronomy`) — real data, not a proxy. Two
intermediate arrays their own script needs (`bckg.npy`, `pois.npy`) are not in that public repo,
so their exact pixel-level noise reduction could not be reproduced; everything else used here is
real and from their own script: z_lens=0.881, z_source=2.059, pixel scale 0.025″/px, PSF FWHM
0.14″. Background noise was instead estimated directly from this image's own corner
(background-dominated) pixels — simple, standard, and defensible, if not identical to their
pipeline.

**Result: the smooth EPL+shear+Sersic macro-fit does not fit this real system well.** χ²/dof
stayed at **4–5** even after substantially widening every parameter bound from this project's
synthetic-tuned defaults (two separate bound choices tried; both left several parameters pinned
at their limits, not converged to an interior optimum) — a real, meaningfully worse fit than the
~1 our synthetic single-Sersic-source images always achieve, since real single-Sersic sources are
correctly-specified for those images by construction, and B1938+666's real background source
is not remotely Sersic-shaped. **This is the same model-misspecification failure mode already
documented for Tier-1 COSMOS sources, now independently confirmed on an actual real observed
system rather than a synthetic proxy for one.** Built on that already-poor host fit, the
subsequent subhalo/interloper scan landed at (x, y) = (0.08″, 0.14″), log₁₀(M/M☉) = 8.0 —
inside Şengül+2022's own reported x-range (−0.15″ to 0.15″) but **not** their y-range (0.4″ to
0.6″) or mass range (up to 3.5×10¹⁰ M☉, i.e. our result is ~2.5 dex lower).

**Honest reading:** this is not a reproduction of their result, and doesn't claim to be — with
the smooth fit itself at χ²/dof~4-5, nothing built on top of it should be trusted quantitatively.
The real value here is what it confirms independently: Family A's single-Sersic-source
assumption, already shown to fail on synthetic COSMOS images (Tier 1), also visibly fails the
moment it meets one real, actual observed lens — not a hypothetical. A meaningful next step
(not attempted here) would need a properly flexible source model (a pixelated/shapelet source
reconstruction, as Şengül+2022 themselves use) before any real-data detection claim from this
pipeline could be trusted.

## Family D (population SBI) — a second, better-resourced attempt, still no working per-lens detector

**v0 recap** (2026-09-11, first pass): NPE + a 4-block/32-dim CNN embedding, 8,000 images,
single per-lens scalar target (log10 mass, floor=6.0 for absent). Posterior means clustered
near the training set's unconditional marginal (9.23/9.13/8.83 for a no-subhalo/huge-mass/
small-mass image respectively) — essentially no discrimination.

**v1 — three changes made before retrying, each independently motivated:**
1. **Floor de-duplication.** v0's 753 "no subhalo" targets were an *exact* repeated constant
   (6.0). Asking a continuous normalizing flow to represent a literal point mass mixed into a
   continuum is the wrong tool for the job; v1 jitters the floor (6.0 ± 0.3, seeded) into a
   narrow but finite bump instead.
2. **Deeper embedding net.** `src/baseline_d/embedding.py` v1: 5 conv blocks up to 256 channels
   and a 64-dim output (v0: 4 blocks, 128 channels, 32-dim) — closer to the U-Net's own
   effective capacity, on the theory that a global-summary task shouldn't need *less*
   representational power than a per-pixel one.
3. **More flow capacity and more data.** `hidden_features` 50→64, `num_transforms` 5→8;
   training set doubled to 16,000 images (`data/train_fixed60_big`).

Training took **113 minutes** (30 epochs, early-stopped — 6,796s), against v0's 26 minutes —
a real cost, worth weighing before a third attempt.

**Result: it didn't work, and by one measure it's slightly worse.** Best validation loss
reached was **1.723**, at epoch 10 — *higher* (worse) than v0's best of **1.6855**, despite 2x
the data and a substantially bigger network. The improvement from epoch 1 to the best epoch
was 1.729 → 1.723, under 0.4% relative — essentially no learning signal, before the same
overfitting-then-diverging pattern as v0 took over (val loss reached 2.72 by epoch 30).

**Completeness by mass bin is still flat and non-monotonic** — c=60: 19%, 15%, 13%, 21%, 5%,
10% across the six bins; c=15: 12%, 20%, 19%, 16%, 10%, 13%. **The highest-mass bin (the
easiest possible case — a huge, obvious subhalo) scores only 10% completeness, barely above
the 10% false-positive baseline.** The paired flip test is symmetric again (30 flip one way,
32 the other, of 275 pairs) — no systematic concentration effect either direction.

**The direct diagnostic confirms this isn't noise in the completeness metric — the posterior
genuinely still isn't conditioning on the image.** Sampling three of the *largest, easiest*
subhalos in the test set (true log₁₀ mass 10.87, 10.93, 10.82 — about as unambiguous a signal
as this dataset contains) gives posterior means of **8.81, 8.75, 9.24** — statistically
indistinguishable from a no-subhalo image's **8.87–9.15** across three separate draws. Whatever
the network is doing, it is not detecting even the most obvious cases.

**Conclusion, stated plainly:** more data and more capacity did not fix this, at real
additional cost (4.3× the training wall-clock for a worse best-validation-loss). That rules
out "just needed a bigger network" as the explanation and points at something more structural
— most likely that a single global scalar summarizing an entire 64×64 image, produced by an
image-conditioned normalizing flow, is a genuinely harder inference problem than the
segmentation task the U-Net solves (which gets a dense, spatially-localized training signal at
every pixel), and this pilot's setup isn't finding the optimization path to it. **A third
architecture tweak in this session is not warranted** — the honest next step is either (a) the
literature's actual approach: combine many individually-weak per-lens likelihood-ratio/
posterior estimates into one population-level constraint (Wagner-Carena+2023, Brehmer+2019),
rather than asking one network to nail a single lens, or (b) a properly resourced training run
(learning-rate search, batch-size sweep, many more epochs) that this pilot's time budget
doesn't cover. Either is future work, not a same-session fix.

**RQ1 status:** the three-way comparison remains two-way (Family A vs Family C, both now
seed-repeated and robust — see above). A working Family D baseline is open work, now with two
documented, honestly-diagnosed attempts behind it rather than one.

## RQ4 extension — does training on realistic concentration diversity fix the blind spot? Not at this budget: it broke training entirely.

**Motivation.** Before running this, we checked what the Dutton & Maccio (2014) ΛCDM
concentration–mass relation actually predicts for this mass range at z=0.5: **c ≈ 8–15**,
mostly *at or below* the literature's own "low concentration" ablation value of c=15
(c≈15 at 10⁸ M☉, falling to c≈8 by 10¹¹ M☉). So the field's "realistic" low-c test point is
already on the optimistic side of a proper mass-dependent relation — worth noting regardless
of what follows.

**Experiment.** Trained two fresh seeds on `data/train_cdm` — identical setup to the main
run, except every subhalo's concentration is drawn from the Dutton-Maccio relation with its
quoted 0.11 dex scatter, instead of a fixed c=60 — then evaluated on the *same*
`test_fixed60`/`test_fixed15` populations as everything above.

| | AUC (in-distribution) | detected @ c=60 (of 905 paired) | detected @ c=15 |
|---|---|---|---|
| cdm-trained, seed 0 | 0.487 | 80 | 84 |
| cdm-trained, seed 1 | 0.482 | 78 | 84 |
| *(for reference)* c60-only-trained, 4-seed mean | 0.619 ± 0.018 | ~276 | ~92 |

**Both cdm-trained seeds collapsed to chance level (AUC ≈ 0.48, i.e. no better than a coin
flip) — on *both* test sets, including the easy c=60 case it should have found trivial.**
This is not "generalizes evenly but weakly": it is training failure. The best validation loss
reached (0.41, 0.39) is also visibly worse than every c60-only run (0.34–0.37).

**Reading it, carefully — this is a negative result about this pilot's training recipe, not
established evidence that concentration diversity can't help.** The cdm training distribution
is genuinely harder (mean c≈10–12 across the mass range vs. always c=60), and every
hyperparameter here (pos_weight=150, learning rate, epoch budget, mask radius) was tuned
implicitly by trial on the easy c=60 case. At n=2 seeds and no hyperparameter search on the
harder distribution, the most defensible conclusion is: **at this data/compute budget
(8,000 images, 25 epochs, no curriculum), training directly on realistic concentration
diversity did not produce a working detector, where training on the easy fixed-c=60 case
reliably did.** That is itself a plausible, previously-unstated reason the literature trains
at easy, unrealistic concentrations: it may be a bootstrapping necessity at moderate data
budgets, not just an oversight. Testing this properly needs a curriculum (start at c=60,
anneal toward realistic concentrations), more data, or both — flagged as a next step, not
attempted here.

## Tier 1 — real COSMOS sources (2026-09-11): infrastructure validated, and the first result is a training-budget failure, not a robustness confirmation

**What this is.** Every result above used Tier 0: a single-Sersic source, the operating point
shared by Ostdiek+2020/22, Hughes+2024 and Campbell+2026. `problem_statement.md`'s Tier 1 calls
for real COSMOS/HST galaxy postage stamps instead (Tsang+2024's own baseline already used
these). Built `src/lensing/cosmos_source.py` + a `source_type="cosmos"` branch through
`config.py`/`simulate.py`: downloaded the `galsim` COSMOS 23.5-sample catalog (50,932 HST
ACS/F814W galaxies, Zenodo 3242143 — the same catalog `paltas` wraps, since `paltas` itself
still doesn't install on Python 3.14), draw a real galaxy per lens, deconvolve it from its
native ACS PSF, reconvolve with a regularizing Gaussian, and hand the resulting fine-pixel
stamp to lenstronomy as a source light profile (`INTERPOL`) — lenstronomy still does all the
actual ray-tracing, instrument PSF convolution and noise, exactly as in Tier 0.

**A real bug, caught by actually looking at the images, not by the code running cleanly.**
The first version used a 0.02" regularizing-Gaussian FWHM at a 0.03" output pixel scale —
narrower than the native ACS/F814W PSF (~0.1" FWHM). The code ran without error and produced
plausible-looking arrays, but the rendered stamps were pure noise/moiré with no galaxy visible
at all (`data/sanity/cosmos_raw_stamps.png`, first version) — deconvolving out the real PSF and
resharpening past the data's native resolution amplifies the correlated pixel noise in every
COSMOS cutout without bound. Fixed by raising the regularizing FWHM to 0.10" (≈ the native
resolution); re-rendered stamps now show clear disks, spiral arms and star-forming clumps
(`data/sanity/cosmos_raw_stamps.png`, current version). This is exactly the kind of failure
`../problem_statement.md`'s emphasis on direct sanity checks over trusting summary metrics is
for — a silent, confidently-wrong dataset would have been generated and possibly used
otherwise.

**Validated, then a first dataset generated.** `scripts/sanity_render_cosmos.py` renders the
identical macro-lens + subhalo system with a Sersic vs. a COSMOS source side by side
(`data/sanity/cosmos_vs_sersic.png`): same Einstein-ring geometry and same subhalo-residual
dipole location, as expected, with the arc itself now showing real internal clumpy structure
instead of a smooth Sersic profile. Generated `data/tier1_cosmos_fixed60/` (n=1,000,
Tsang+2024 macro/subhalo priors, c=60, `seed=0`) at 9.4 images/s once caches warmed up (vs. ~40
img/s for Tier 0 — the INTERPOL profile's per-pixel interpolation is the cost); spot-checked 8
images directly (`data/sanity/tier1_dataset_spotcheck.png`), all clean arcs, no NaN/Inf, flux
range sane. A faint residual moiré texture still shows in a minority of stamps (visible in one
of the eight spot-checked images) — some COSMOS galaxies apparently need more regularization
than others; not severe enough to obscure the arc, but noted here rather than glossed over.

**Retrained the U-Net (Family C) on Tier-1 COSMOS sources, n=2 seeds — it came back at chance
level.** Generated the full Tier-1 population battery mirroring Tier 0 exactly
(`data/tier1/{train_fixed60 (n=8,000), test_fixed60, test_fixed15, no_subhalo,
multipole_m4_a1, multipole_m4_a3}` (n=1,000 each), same seeds as their Tier-0 counterparts),
trained 2 independent seeds with the identical recipe (25 epochs, pos_weight=150, same
architecture) that gave Tier-0's U-Net an AUC of 0.619 ± 0.018, and evaluated with the
unmodified `evaluate_detector.py`/new `aggregate_tier1_seeds.py`.

**Result: ROC AUC 0.484 ± 0.009 — indistinguishable from a coin flip — and completeness flat
at 6-12% across every mass bin, at *both* c=60 and c=15, with the two concentration curves
overlapping within error bars** (`results/aggregate_tier1/completeness_vs_mass_aggregate.png`).
The confounder FPRs (9.7%/8.8%) sit at or slightly *below* the 10% baseline rather than
inflated — not evidence of confounder-robustness, just further confirmation the model isn't
discriminating anything. Both training runs show the same pathology: validation loss bottoms
out early (epoch 8-11, val_loss≈0.40 — already visibly worse than Tier-0's best of 0.368) then
climbs steadily to 0.74/0.73 by epoch 25 while training loss keeps falling (0.26/0.29) — textbook
overfitting, not a plateau.

**Direct diagnostic, mirroring the Family-D checks above:** scored the model on the 5 largest,
easiest subhalos in the test set (mass ≈ 10.996-10.999, essentially the top of the prior) —
max-pixel scores of 0.78-0.84 — then scored the *entire* `no_subhalo` population, which spans
0.71-0.87 (mean 0.79, std 0.037). **The easiest possible positive examples sit entirely inside
the negative population's own score range.** This is not "harder but still working" — there is
no separation at all, even in the best case. A second tell: mean per-image probability for these
"detections" is 0.12-0.14, not the near-zero-everywhere-except-one-small-peak a genuinely
localized 2-pixel subhalo detector should produce for a target that is ~0.3% of the frame — the
network is firing broadly across the image, consistent with having learned to respond to bright,
clumpy structure *in general* rather than to a localized subhalo perturbation specifically.

**Reading it, carefully — same posture as the RQ4 extension above.** Real COSMOS galaxies are
intrinsically clumpy: every image now has multiple genuine bright knots from the source's own
structure, not just the smooth single arc Tier 0 provided, so a "localized brightness
perturbation" is no longer a rare, subhalo-specific signature — it's what every image looks like
anyway. That is a real, literature-consistent difficulty (Adam 2026: source structure suppresses
detection sensitivity "across a broad range of scales"; Vegetti 2023 makes the same point for
analytic-source methods), and a plausible reason this pilot's recipe failed outright rather than
degrading gracefully. But at n=2 seeds, no hyperparameter search adapted to the harder task, and
the *same* 8,000-image/25-epoch budget that was tuned (implicitly, by trial) for the smooth Tier-0
case, **the most defensible conclusion is a training-recipe failure at this budget, not
established evidence that Tier-1 signal is undetectable in principle** — exactly the framing
already used for the concentration-diversity training failure above. A working Tier-1 detector
would need more data, source-aware preprocessing (e.g. subtracting a smooth source model first),
or both — flagged as a next step, not attempted further this session.

**What this means for RQ1/RQ2/RQ4's headline findings:** they remain Tier-0-only, unconfirmed at
Tier 1 — not because Tier 1 disproved them, but because no detector trained at this budget
produced a usable signal to test them with. The Family-A parametric scan (which fits an explicit
parametric model rather than learning from pixels) is the more promising next baseline to try on
these same Tier-1 populations, since it isn't vulnerable to the same "not enough training data to
learn past source clumpiness" failure mode.

### Family A on Tier 1 — ran, but the confounder-FPR result is noise-dominated at this sample size, not a clean finding either direction

Unlike the U-Net, Family A doesn't need training — it fits each image directly, so it ran on the
same Tier-1 populations without a data-budget problem. But its own smooth-model fit is
*genuinely misspecified* against a real COSMOS source (unlike Tier 0, where Sersic-fits-Sersic):
the optimizer still converges cleanly (verified directly: `scipy` `status=2`/`success=True` on
every case checked, not silently failing) but to a large, highly variable residual (χ²/dof
mean 57-67, **std 58-79 — larger than the mean**) simply reflecting how much a single smooth
blob can't capture a real galaxy's structure. This makes the Δχ² detection statistic extremely
heavy-tailed (population max 9,000-60,000 vs. a median of ~130-170), and the 10%-FPR threshold
ends up sitting deep in that tail.

**Two independent seeds gave opposite directions**: seed 1's confounder FPR *dropped* below
baseline (10.0% → 7.3% → 5.0%); seed 2's *rose* (10.0% → 14.3% → 12.0%). Averaged: 10.8% ± 4.9%
(a=0.01) and 8.5% ± 4.9% (a=0.03) — both consistent with the 10% baseline within 1 std, in
either direction. **Read honestly, this is not "Family A's confounder-sensitivity reverses on
real sources" — it's that the source-morphology noise floor is so large and heavy-tailed that
n=300/seed cannot resolve a confounder effect here at all**, in contrast to Tier 0's clean,
tight, 2-seed-confirmed 33%/76% result. Completeness also stayed low and flat at both
concentrations (comparable within noise), consistent with the same heavy-tail effect swamping
genuine subhalo signal too. A real test of Family A's Tier-1 confounder sensitivity needs either
many more seeds or a statistic more robust to this specific heavy tail (e.g. a rank-based or
median-referenced statistic) — not attempted here.

### A more important catch: does "detected" even mean "found in the right place"? For Family A, mostly no.

Every completeness/FPR number above (Tier 0 and Tier 1, both families) is a **pure classification
statistic** — score crosses a threshold — with no requirement that the method's own reported
subhalo *position* be anywhere near the truth. This matches Tsang+2024's own stated convention
("'maximum pixel probability' defines image score; threshold determines classification") but it
is silently permissive: a method can be scored as a "correct find" while pointing anywhere in the
frame.

**Directly checked, Tier-0 `test_fixed60`, seed 0** (`scripts/plot_localization_accuracy.py`,
`results/localization/`): of Family A's 108 nominal "correct finds," only **13.9%** land within
Ostdiek+2020's own 2-pixel localization criterion (median offset **0.30″** — a third of the
typical Einstein radius); **18 of the 108** have Δχ² < 0, meaning the subhalo scan *technically
made the fit worse* and still counted as "detected," because the calibrated threshold itself is
slightly negative (−10.6) — an artifact of how permissive a 10%-FPR bar is on this statistic's
distribution. Only **12/108 (11%)** are both well-localized *and* have a strong, unambiguous
Δχ² > 50. **The U-Net is materially better on this axis**: 56.7% of its 263 "correct finds" land
within 2 pixels (median offset 0.13″), and 43% are both well-localized and near-maximal
confidence (score > 0.99).

**Spatially, this is visually dramatic** (`results/localization/localization_accuracy.png`):
Family A's false alarms under the multipole confounder cluster into distinct **radial spokes** —
a direct fingerprint of its scan only checking 8 fixed angles × 3 radii, not a diffuse noise
pattern. It gets fooled *systematically*, at specific grid positions, not randomly. The U-Net's
false alarms show no such structure.

**Under the stricter, arguably more honest definition of "correct find" (detected *and*
localized within 2px) vs "false alarm" (detected, no subhalo present)**: Family A goes from 108
"correct finds" to just **15** true correct finds against **198** false alarms on the confounder
population alone — better than a **13:1** false-alarm-to-correct-find ratio. The U-Net: 149
correct finds vs. 95 false alarms — still imperfect, but not remotely as lopsided. An
interactive 3D comparison using exactly this stricter definition is at
`data/sanity/explainer/positions_for_viz.json` (rendered as the "Clump Atlas" artifact).

**This changes how RQ1's headline number should be read.** The 76%-under-confounder figure
reported above is a real, seed-confirmed *classification* false-positive rate — but Family A's
*true* localized-detection rate, even on the clean in-distribution population, is far lower than
"108/905" ever suggested. The honest summary is not "Family A finds subhalos and also panics
under confounders" — it's closer to "Family A's Δχ² statistic rarely pins down a real subhalo's
actual location even when it fires correctly, and fires constantly (and in a structured,
predictable spot) when there's nothing there at all." This is arguably the single most important
correction to make before this goes in front of a reviewer, and it should be checked for Family D
and for every mass bin, not just this one population — flagged as a next step.

**Second-seed confirmation (seed 200; same script with `--a-root results/baseline_a_seed1`,
`results/localization/summary_seed200.json`, 2026-09-11):** 113 nominal correct finds, **16.8%**
within 2 px (19/113; seed 0: 13.9%), median offset **0.31″** (seed 0: 0.30″), **26** with Δχ² < 0
still counted as detections (seed 0: 18; that seed's calibrated threshold was −16.1), 207 false
alarms on the confounder population (seed 0: 198). Two-seed localized fraction **15.4% ± 2.0%**.
Not a fluke of one 300-image draw.

### Re-scored at the literature's own kind of threshold (2026-09-11, self-review fix): the findings are threshold-independent

A fair referee objection to everything above: forcing a 10%-FPR calibration onto Δχ² puts the
threshold at −10.6 (seed 0) / −16.1 (seed 200), a bar no practitioner would use — parametric
scans report Δln Z > 10 (Nightingale+2024) or Δlog E ≥ 50 (Despali+2022). So
`scripts/evaluate_baseline_a_fixed_threshold.py` re-scores the *same* scan results at absolute
Δχ² > 20 and Δχ² > 100 (Δln L ≈ Δχ²/2, no Occam penalty — which would only make the evidence
thresholds stricter). No new compute; `results/baseline_a*/summary_fixed_threshold/`.

| operating point (seed 0 / seed 200) | FPR clean | FPR mp a=0.01 | FPR mp a=0.03 | detected (of subhalo-bearing) | localized ≤2 px | localized & mass low | median mass err |
|---|---|---|---|---|---|---|---|
| 10% FPR (Δχ² > −10.6 / −16.1) | 10.2% / 10.1% | 30.7% / 34.6% | 75.0% / 77.0% | 108/232 / 113/226 | 13.9% / 16.8% | 15/15 / 19/19 | −1.00 / −1.06 |
| Δχ² > 20 (≈ Δln Z > 10) | 0.7% / 0.0% | 11.7% / 5.4% | **57.6% / 61.0%** | 81/232 / 74/226 | 16.0% / 17.6% | 13/13 / 13/13 | −1.24 / −1.11 |
| Δχ² > 100 (≈ Δlog E ≥ 50) | 0.0% / 0.0% | 2.8% / 1.4% | **33.0% / 34.2%** | 58/232 / 56/226 | 15.5% / 16.1% | 9/9 / 9/9 | −1.24 / −1.38 |

**Reading it:** the permissive threshold *did* inflate the raw detection count and it alone
produced the 18/26 negative-Δχ² "detections" — that part of the objection stands and the
manuscript now says so. But the two substantive findings are threshold-independent: (1) the
confounder false-alarm result *sharpens* — even at Despali's strictest threshold a third of
subhalo-free multipole lenses are flagged while clean subhalo-free images give exactly 0%; (2)
localization stays flat at 15–18% at every operating point and every localized detection still
underestimates the mass (26/26 at Δχ²>20, 18/18 at Δχ²>100). A stricter threshold trades
completeness for false alarms without improving where the scan points. Written into the paper
as Table "Family A at three operating points" (Sect. 4.3) plus a sentence in Sect. 4.2.

### None of the three families reliably estimates mass — for three different reasons

"Correct find" above checks *position* only. Checking mass for those same 15 Family-A
location-correct cases (`test_fixed60`, seed 0) makes things worse, not better: **15/15 (100%)
have the wrong mass, always underestimated** — median error **−1.0 dex, range −0.47 to −1.94
dex** (i.e. the true clump is up to 87× more massive than Family A's own best fit says). Only
1 of the 15 lands within even 0.5 dex of truth. This is not scattered noise: 12 of the 15 land
on the *smallest* grid value tried (10⁸·⁵ M☉) regardless of the true mass. **[Superseded — see
"Concentration variants" below: the concentration explanation that follows was tested and is
wrong; the cause is the frozen macro-model.]** At the time this pointed at a
specific, already-documented mechanism — the scan assumes a fixed concentration (c=15) for every
trial subhalo, while this population's real subhalos are c=60 (much denser). A denser true clump
makes a sharper lensing signature than a puffy c=15 template can reproduce at the true mass; the
fit "resolves" the mismatch by shrinking the trial mass rather than matching it, since a bigger,
wrong-shape perturbation would create even more residual, not less. Right concentration
assumption → the fit's mass estimate would very plausibly be fine; wrong concentration
assumption → the mass estimate is wrong in a specific, predictable direction, even in the rare
cases where position is right.

**Zooming out, all three families fail at "what is the clump's mass," for three unrelated
reasons:**
- **Family C (U-Net) doesn't attempt it at all.** It's architecturally a location/presence
  classifier, matching Ostdiek+2020/Tsang+2024's own design — mass is simply out of scope, not a
  failure of execution.
- **Family A attempts it and gets it systematically wrong** — a concentration-assumption
  mismatch biases every mass estimate low, even among its rare correctly-located finds (above).
- **Family D attempts it as its whole purpose** (a full posterior over log10 mass) **and never
  produces an image-dependent answer at all**, in two independent tries — its posterior mean
  for the largest, easiest true subhalos in the test set was statistically indistinguishable
  from a no-subhalo image's (see the "Family D" section above). Not a biased estimator; no
  working estimator.

**None of the three published/reimplemented detection strategies tested in this pilot can be
trusted for a mass measurement** — one doesn't try, one is systematically biased by a
documented, mechanistic cause, and one never learned to condition on the image at all. That's a
strong, cross-family claim this pilot is well-positioned to make, and it matters beyond this
paper: a mass function is the actual observable that distinguishes dark matter models (CDM vs.
WDM vs. SIDM) — a detector that only says "yes/no" or gets the mass wrong by ~1 dex cannot
constrain that function, regardless of its completeness/FPR numbers.

**Second-seed confirmation of the mass bias (seed 200):** of its 19 location-correct finds,
**19/19 underestimate the mass** (median −1.06 dex, range −2.11 to −0.01), 2/19 within 0.5 dex —
the same picture as seed 0 (15/15, median −1.0 dex). **Across both seeds: 34 location-correct
cases, 34/34 biased low.**

### Concentration variants (self-review fix, 2026-09-11): the mass bias is NOT the concentration assumption

The paragraph above blamed the mass bias on the fixed c=15 template. The referee's version of the
same point was sharper: "you fixed c=15 against a c=60 population — the bias is self-inflicted."
Either way it is a testable claim, so the scan was re-run on the **identical 300-lens subsamples
(seeds 0/1/3)** with the concentration assumption changed (`run_baseline_a.py --concentration 60`
= oracle match to the population; `--concentration cm` = Dutton & Maccio 2014 relation at each
trial mass, c = 13.3 / 10.8 / 8.8 at 10^8.5 / 10^9.5 / 10^10.5). Results at the common 10%-FPR
threshold (`results/baseline_a_{c60,ccm}/summary_fixed_threshold/`, `results/localization/summary_{c60,cm}.json`):

| scan concentration | 15 (seed 0) | c–M | 60 (matched) |
|---|---|---|---|
| threshold Δχ² | −10.6 | −11.0 | −6.8 |
| completeness 9.5–10 / 10–10.5 | 71 / 85% | 74 / 85% | **83 / 94%** |
| multipole a=0.03 FPR | 75% | 74% | **89%** |
| detected of subhalo-bearing | 108/232 | 109/231 | 122/231 |
| localized within 2 px | 13.9% | 15.6% | 13.1% |
| localized & mass low | 15/15 | 16/17 | **16/16** |
| median mass error (dex) | −1.0 | −1.0 | −0.93 |
| on smallest hypothesis 10^8.5 | 13/15 | 15/17 | 13/16 |
| at Δχ²>20: low / localized | 13/13 | 14/14 | 14/14 |
| at Δχ²>100: low / localized | 9/9 | 9/9 | 11/11 |

**Three things follow.** (1) The matched concentration does what a better-matched template
should: completeness rises in every bin — and so does the confounder FPR (75→89%): a denser
template is more sensitive to *everything*, including unmodelled lens shape. (2) Localization
does not move (13–16%): it is set by the 3×8 position grid, not by the template. (3) **The mass
bias does not move.** 16/16 low at the true concentration, 16/17 at the ΛCDM relation, median
−0.93 to −1.0 dex either way, and the winner is still the smallest hypothesis in ~80–90% of
cases. So the explanation given above — "puffy c=15 template on a dense c=60 clump" — is
**wrong**, and the earlier text is superseded by this section. The paper (Sect. 4.3, Table
`tab:conc`, Fig. 5b) now says so explicitly.

### Template control (referee round 3, 2026-09-11): is the scan's non-collapse at c=15 just template matching?

On the c=15 population the default c=15 template is the *correct* one, so the scan's flat completeness
across concentration could have been a matched template hiding a weaker signal. Control: every template
on both populations (`results/baseline_a_{c60,ccm}/test_fixed15`, seed-0 subsample; `results/template_control.json`).

| template | c=60 pop: 9–9.5 / 9.5–10 / 10–10.5 / 10.5–11 | c=15 pop: same bins | flip (c60 lost : c15 gained) |
|---|---|---|---|
| c=15 (paper default) | 38 / 71 / 85 / 85% | 25 / 56 / 75 / 91% | 21 : 4 |
| c–M | 38 / 74 / 85 / 85% | 25 / 56 / 71 / 91% | 24 : 4 |
| **c=60 (held fixed on both)** | 44 / 83 / 94 / 100% | 28 / 63 / 80 / 94% | 24 : 3 |

With the c=60 template held fixed, going from the c=60 to the c=15 population costs 16/20/14/6 points —
a real loss (a diffuse clump of the same M200 is a weaker signal) but nothing like the U-Net (51→15) or
Family B (75→12.5). **The non-collapse is a property of the Δχ² statistic, not of a template that
happened to match.** Paper: Sect. 4.1 paragraph + Table `tab:conc` (c=15-population rows).

### Blind initialization (referee round 3, 2026-09-11): the near-truth init buys convergence, not results

`run_baseline_a.py --blind` (`fit._blind_init_vec`: θ_E from the flux-weighted radius of >5σ pixels,
γ=2, round, no shear, source at centre, generic Sérsic, amp from the peak; NO truth used) on the
seed-0 subsamples → `results/baseline_a_blind/`, `results/localization/summary_blind.json`.

| same lenses | near-truth init (paper) | **blind init** |
|---|---|---|
| unreliable macro fits: clean / fixed60 / multipole | 5 / 43 / 36 | **15 / 58 / 43** |
| median χ²/N on clean, fraction worse by >0.05 | 0.985 | 0.987, 5.7% |
| 10%-FPR threshold | −10.6 | −10.5 |
| nominal detections on fixed60 | 108 | 104 |
| localized ≤ 2 px | 13.9% (15) | 14.4% (15) |
| mass, localized | 15/15 low, −1.0 dex | 15/15 low, −1.0 dex |
| multipole FPR @ 20 / 100 | 58% / 33% | 59% / 35% |

Where the blind fit converges (95% of clean lenses, ~80% of subhalo lenses) it lands on the same
optimum; the idealization only changes *how often* the fit fails, and those failures are already
reported as exclusions. Paper: Sect. 3.2 "What the numbers mean".

### Joint-scan null control (referee round 3, point 1; 2026-09-12)

`run_baseline_a_joint.py` now continues the 13-parameter smooth fit from its solution with the 24 joint fits'
total budget (max_nfev 24×40×14, default tolerances) before scanning; Δχ² is measured against the polished smooth
fit. `results/baseline_a_joint_null_c15/{no_subhalo,multipole_m4_a3}` (100 lenses each; evaluated by
`evaluate_joint_vs_frozen.py --joint-root ... --tag _null`): **polish gain = 0.0 on every lens** (median and max,
97 clean + 92 multipole; optimizer stops after 2 evaluations) → the smooth fits were converged and the joint
numbers are unchanged (thr +7.2; multipole FPR 100 / 98 / 90% at 10%/Δχ²>20/>100; median Δχ² 737). The joint
improvement is the perturber's, not a local optimum's. Paper: Table `tab:joint` row + caption, Sect. 4.2.

### Full 1000-lens populations (referee round 3, point 4; 2026-09-12)

`results/baseline_a_full/` (frozen scan, c=15 template, all 5 Tier-0 populations, 1000 lenses each) and
`results/baseline_b_full/fitted/`; `scripts/evaluate_full_populations.py` → `results/full_populations.json`,
`paper/tables/full_populations.tex` (Wilson 68% intervals). The three 300-lens subsamples overlap (83–105 lenses
per pair, 647 distinct) — stated in the paper — so the full runs are the independent check:

| | A c60 (9–9.5 / 9.5–10 / 10–10.5 / 10.5–11) | A c15 | A FPR a1 / a3 | A flip | B c60 | B c15 | B FPR | B flip |
|---|---|---|---|---|---|---|---|---|
| 3 × 300 subsamples | 40±6 / 69±3 / 89±7 / 83±2 | 28 / 58 / 78 / 89 | 32.1 / 76.6 | 55:10 | 65±9 / 88±1 / 94±3 / 96±7 | 17 / 56 / 77 / 95 | 38.7 / 85.8 | 55:8 |
| **full 1000** | **36±4 / 69±4 / 89±3 / 81±5** | 25 / 59 / 78 / 84 | **31.5 [30–33] / 76.7 [75–78]** | 63:8 (732 pairs) | **58±4 / 87±3 / 96±2 / 93±3** | 23 / 52 / 79 / 94 | **39.6 [38–41] / 84.7 [83.5–86]** | 177:29 |

Everything within the subsample scatter; per-bin n = 124–163 (59 in A's top c60 bin). Paper: Sect. 4.1 paragraph +
Table `tab:fullpop`; Fig. 2 error bars are now binomial for A/B (full populations) and seed s.d. ⊕ binomial for the U-Net.

### Multipole in the macro-model (referee round 3, point 2; 2026-09-12)

`run_baseline_a.py --macro-multipole` (fit.py: `MULTIPOLE` m=4 term with free a_m, phi_m appended to the
vector, initialised at zero) on the seed-0 subsamples → `results/baseline_a_mpmacro/`. On the a=0.03 multipole
population the smooth fit recovers a_m to 4 digits and χ²/N → 0.99 (bare EPL: 4–7). At the common threshold the
multipole FPR falls **32→8.5%** (a=0.01) and **77→9.7%** (a=0.03); at Δχ²>20: 8.5→1.1% and 58→0%; at >100: 1.8→0.7%
and 33→0%. c=60 completeness 12/18/32/76/78/86% (bare: 12/17/38/71/85/85) — unchanged within scatter; localization
14%; mass bias unchanged (15/15 low). Macro-fit exclusions on the multipole populations 17/36 → 16/23; clean 5 → 10.
**The 77% headline is the cost of a bare EPL macro-model; the post-Lange+2024 fix removes it completely at no
completeness cost.** Paper: abstract, Sect. 4.2 + Table `tab:mpmacro`, Conclusion 2, Fig. 10 cell.

### Family B on Tier 1 (referee round 3, point 5; 2026-09-12)

`run_family_b.py --data-root data/tier1` (fitted variant, 300 lenses/population) → `results/baseline_b_tier1/`,
`scripts/evaluate_family_b_tier1.py`. The misspecified single-Sérsic source (inherited from Family A's fit)
drives 281/300 clean fits past the χ²/N<10 gate (median 43); gated numbers are meaningless (19 clean lenses left).
Gate lifted: threshold on max|δκ| **0.09 → 1.97** (20×), multipole FPR 7%, completeness 15/3/11/7/14/20% — chance
in every bin. Same failure as A on Tier 1: the source model, not the detector, is the limit. Paper: Sect. 4.5,
Fig. 10 cell.

### Grid localization ceiling (provenance, 2026-09-12)

`scripts/grid_localization_ceiling.py` → `results/grid_localization_ceiling.json`: with the scan's 3-radii × 8-angle
grid (nodes 0.45″ apart radially, ~0.8″ in angle) and the simulator's placement prior, only **21.4%** of subhalo
positions lie within 0.16″ (2 px) of any node; the median nearest-node distance is **0.23″**. The scan's 13–15%
localized fraction is therefore ~⅔ of what a perfect scan on this grid could report (Sect. 4.3).

### Completeness vs projected signal (referee round 3, point 6; 2026-09-12)

`scripts/completeness_vs_signal.py` → `results/completeness_vs_signal.json` (0.1″ aperture) and
`_0p2.json` (0.2″). TNFW mass projected within the aperture (lenstronomy `TNFW.mass_2d` × Σ_crit) as a
concentration-independent signal proxy; completeness of each family on both matched populations binned by it
(seed-0 A/B results; U-Net v0):

| log M_proj(<0.1″) bin | A c60 / c15 | B c60 / c15 | C c60 / c15 |
|---|---|---|---|
| 7.5–8.0 | 12 / 12% | 14 / 12% | 7 / 10% |
| 8.0–8.5 | 18 / 52% | 30 / 49% | 9 / 11% |
| 8.5–9.0 | 67 / 82% | 87 / 84% | **23 / 13%** |
| 9.0–9.5 | 85 / — | 95 / — | 55 / — |

Physical methods: the two populations coincide (within binomial scatter; the c15 template even gives the scan a
small edge on the c15 population) → the concentration collapse is the *signal* (less projected mass inside the
resolution element). U-Net: at equal projected signal the c=15 population is still missed (13% vs 23%, and flat
at chance everywhere) → a second, *learned* component (shape of the dense-perturber signature). Same at 0.2″
(U-Net 38 vs 13% at 9.0–9.5). Paper: Sect. 4.1 paragraph + Fig. `fig13_signal`, abstract, Conclusions 1, Sect. 5.1.

### Then what causes it? Oracle-position test: not the position grid either

Next suspect: the 3-radii × 8-angle position grid (median nearest-grid-point offset 0.24″). If
the winning mass is small because a big clump at a *wrong* position hurts more than it helps,
then evaluating the same Δχ² at the **true** position should recover the mass.
`scripts/oracle_position_mass_test.py` (same 300 lenses, same smooth fit, same c=60, subhalo
placed at the truth, mass on the scan's 3-point grid and on a fine 0.25-dex grid 8.0–11.0;
`results/oracle_position_mass/summary.json`). Among the 121 lenses with Δχ²>20 at the true position:

| subhalo at | mass grid | median error | low | within 0.5 dex |
|---|---|---|---|---|
| nearest grid point | 8.5 / 9.5 / 10.5 | −1.25 dex | 116/121 | 16 |
| **true position** | 8.5 / 9.5 / 10.5 | −1.06 dex | 116/121 | 22 |
| **true position** | fine, 0.25 dex | **−1.02 dex** | **121/121** | 11 (2 within 0.25) |

Oracle position and a fine mass grid change nothing. The bias also **grows with true mass**:
median error −0.3 dex at 10^8–10^8.5, −0.8 at 10^9, −1.15 at 10^10, −1.7 at 10^10.5. Looking at
the Δχ² curves themselves is decisive: for a true 10^10.95 clump *at its true position*, Δχ² is
**+712 at 10^9.0 and −38,741 at the true mass 10^10** — adding the true perturber on top of the
fitted smooth model makes the fit catastrophically *worse*. That only happens if the smooth model
has already absorbed most of the perturber's effect.

### The actual mechanism: the frozen macro-model of fit-then-scan

That is the last idealization standing. Family A fits EPL+shear+Sérsic **once, with the subhalo
in the data**, then holds all 13 parameters fixed while adding a perturber. Source position,
shear and ellipticity can absorb the large-scale part of a subhalo's deflection; the residual
left for the perturber to explain is then only the *unabsorbed* part, which a much smaller
clump reproduces best. The published pipelines this family stands in for do **not** freeze the
macro-model: Nightingale+2024 and Despali+2022 re-fit the smooth model jointly with the
perturber at every grid cell (which is exactly why they cost O(days)/lens). Our shortcut is the
fast version and it is the thing under test here. `scripts/joint_refit_mass_test.py` checks it
directly — subhalo at the true position and true c, (i) macro frozen + continuous free mass vs.
(ii) all 14 parameters re-fitted jointly — see the next entry for the outcome.

### Joint-refit test: the frozen macro-model IS the mechanism, and re-fitting removes the bias entirely

`scripts/joint_refit_mass_test.py --n 300 --seed 0` → `results/joint_refit_mass/summary.json`.
Same 226 reliable subhalo-bearing lenses, subhalo at its **true position** with its **true c=60**;
178 have Δχ²>20 under the joint fit.

| what is free | median error | low | within 0.5 dex | within 0.25 dex |
|---|---|---|---|---|
| (i) macro+source frozen at the smooth fit, only log M free (continuous) | **−1.08 dex** | **178/178** | 13 | 2 |
| (ii) all 13 macro+source params **re-fitted jointly** with log M | **+0.004 dex** | 74/178 | 175 | **174** |

(ii) is unbiased (74 low / 104 high is a coin flip) and accurate to 0.25 dex in 98% of cases.
The joint fit also improves the fit by a **median Δχ² = 321 more** than the frozen version — the
shortcut costs detection significance too, not only mass. And the macro-model shifts that absorb
a whole dex of subhalo mass are **tiny**: median |Δθ_E| = 0.004″, |Δγ| = 0.011, |Δe| = 0.005,
|Δshear| = 0.0025, source position 0.002″ — all well below what any lens-modelling paper would
quote as an uncertainty on those parameters.

**The chain of elimination, in order:** fixed c=15 → tested at c=60 and c–M, bias unchanged;
coarse position grid → tested at the oracle position, unchanged; three-point mass grid → tested
on a fine grid, unchanged; **frozen macro-model → tested by joint refit, bias gone.** That is one
cause, established by four experiments, with a clear physical reading: the smooth model fitted
with a subhalo *in* the data has already soaked up most of the subhalo's deflection through
sub-percent parameter shifts, so the residual left for the added perturber to explain is the
residual of a much smaller clump.

**What this means for the paper.** (a) The earlier "concentration" explanation is retracted;
Sect. 4.3 now presents the four-step elimination and Table `tab:mass_mechanism`. (b) This is a
*method-class* result, not a bug in our implementation: any fit-then-scan that freezes the macro
model — the fast variant one is tempted to run at Euclid scale — inherits a ≈ −1 dex mass bias
growing with mass; the published O(days)/lens pipelines avoid it precisely because they re-fit
jointly at every cell. (c) It gives a new wrong-assumption entry: "a macro-model fitted without
the perturber can be held fixed while the perturber is added". (d) It gives a concrete,
quantitative recommendation with a measured payoff (−1.08 → 0.00 dex).

### Joint-refit SCAN (referee Major 1, 2026-09-11): does the frozen macro-model also drive the false positives? No — it hides them

`scripts/run_baseline_a_joint.py` re-runs the scan with all 13 macro+source parameters **re-fitted
jointly with a free-mass perturber at every one of the 24 grid positions** (what Nightingale+2024 /
Despali+2022 do per cell; 24× the cost, ~30 s/lens on 1 core) on the **first 100 lenses of the
identical seed-0/1/3 subsamples**, c=15 (the headline configuration). `scripts/evaluate_joint_vs_frozen.py`
→ `results/baseline_a_joint_c15/summary_vs_frozen.json`.

| same 97–99 lenses | frozen macro (paper's scan) | **joint re-fit per cell** |
|---|---|---|
| 10%-FPR threshold on clean | Δχ² = −10.7 | **+7.2** (statistic now ≥ 0, as it should be) |
| multipole a=0.03 FPR @10% | 77% | **100%** |
| @ Δχ² > 20 | 64% | **98%** |
| @ Δχ² > 100 | 35% | **90%** |
| clean FPR @ 20 / 100 | 1% / 0% | 1% / 1% |
| median Δχ² on multipole lenses | 58 | **737** (joint − frozen: median +652, positive on 87/87) |
| median Δχ² gain on clean lenses | — | +65 (the smooth fit at maxiter=60 was not fully converged) |

**Reading:** the shortcut *understated* the confounder problem. With the macro-model free, a
perturber-plus-macro combination absorbs the m=4 structure far better, so the published-style
pipeline is *more* confounder-prone than our fast proxy, not less. The paper's 76% headline is
conservative; Sect. 4.2 and Table `tab:joint` now say so. (One multipole lens, index 32 — an
unreliable macro fit with a 27-s smooth fit — was killed after >10 min of joint fitting; it was
excluded from every denominator anyway.) **Subhalo population under the joint scan (`test_fixed60`, c=15, same first-100 lenses):**

| same lenses, 10% FPR each | frozen macro | **joint re-fit per cell** |
|---|---|---|
| detected of reliable subhalo-bearing | 40/77 (52%) | **62/75 (83%)** — the shortcut forfeits sensitivity too |
| completeness by bin (n≈8–18 each) | 23/20/25/62/88/88% | 42/67/86/100/100/100% |
| localized ≤ 2 px | 10% (4/40) | 15% (9/62) — the grid's ceiling, unchanged |
| median Δχ² of detections | 172 | 1389 |
| mass error, localized | −1.1 dex, 4/4 low | **+0.5 dex, 0/9 low** (4/9 within 0.5 dex) |
| mass error, all detected | −1.4 dex, 37/40 low | +0.4 dex, 3/62 low |

**Reading:** (i) localization does not move → it is the 3×8 grid, as claimed; (ii) with the
macro-model free the −1 dex bias is gone and a **+0.5 dex** bias appears: that is the fixed
c=15 template's own bias (a puffier profile needs more mass to mimic a dense c=60 perturber),
which the frozen macro-model had masked. At the true position *and* the true c it vanished
(joint-refit test above). So the two idealizations bias the mass in opposite directions and the
frozen-macro one dominates. (iii) The joint scan is also far more sensitive (52 → 83% detected at
the same FPR) — but also far more confounder-prone (77 → 100%). The c=60 joint scan on the grid
was skipped (the machine was oversubscribed; the true-position c=60 joint refit already answers
the mass question).

### Third Family A seed (referee Major 5, 2026-09-11): seed 300

`results/baseline_a_seed300/` (5 populations, same protocol), `results/localization/summary_seed300.json`,
aggregate now n=3 in `results/baseline_a/aggregate/summary.json`. Confounder FPR **32.1±2.2 /
76.6±1.5%** (75.0, 77.0, 77.9); c=60 completeness 15.1±4.7 / 22.4±5.9 / 40.0±5.9 / 69.0±2.8 /
89.0±7.3 / 83.1±1.7%; c=15 14.8 / 22.2 / 27.5 / 57.8 / 77.9 / 89.2%; flip 18±5 : 3±2 (21:4, 21:1,
13:5); localized 13.9 / 16.8 / **9.3%** → **13.3±3.8%**; mass **44/44 low** (medians −1.0, −1.06,
−1.12), 37/44 on the smallest hypothesis; fixed-threshold table (3-seed means): @20 → 34/34 low,
localized 15%, multipole FPR 58%; @100 → 25/25 low, 15%, 33%. Mass-function reweighting rerun with
n=3: Family A 53.1→21.4% (c60), 48.2→19.7% (c15). Every two-seed statement survives; the paper's
Family A numbers are now 3-seed means throughout.

### The unifying mechanism behind three separate-looking Family A findings: it isn't localizing, it's picking the best of ~72 fixed hypotheses

Poor localization (13.9% within 2px), systematic mass underestimation (15/15), and the false
alarms clustering into visible radial spokes in `results/localization/localization_accuracy.png`
and the Clump Atlas artifact all trace back to the same design fact: `scan_subhalo`
(`src/baseline_a/fit.py`) does not search continuously — it evaluates a fixed grid of **3 radii
× 8 angles × 3 masses = 72 candidate hypotheses** per image and reports whichever wins. A false
alarm is not "the method mis-locates something" in a continuous sense; it is "one specific
pre-defined hypothesis, out of 72, happened to fit slightly better than the rest" — which is
exactly why the false alarms in the confounder population land in a small number of discrete
angular positions rather than being smoothly scattered: **there are only 8 angles it is ever
capable of proposing.** The discreteness does *not* explain the mass bias, however — that was traced by
elimination to the frozen macro-model (see the three entries above; the concentration/grid
explanation is retracted) and the localization failure (with only 3 radii available, "close" is often the
best of a bad set of options, not a genuine fit to the true position). **This reframes Family A
not as "a detector that sometimes mislocalizes," but as a coarse hypothesis-selection procedure
whose apparent behavior is largely determined by the shape of its own search grid** — a
materially different, and more precise, characterization than "false-positive rate" alone
conveys, and one a reviewer familiar with the parametric-scan literature will likely find more
persuasive than the raw percentages. (A finer grid would presumably narrow all three gaps
somewhat, at proportionally higher compute cost — not tested here, flagged as a natural next
experiment.)

### Control: does either detector recognize a real subhalo's *lensing* signature, or just react to any bright anomaly? They answer oppositely.

**Question (user's, 2026-09-11):** every failure mode found so far is consistent with "reacts to
local anomalies in general" rather than "recognizes the specific gravitational distortion a real
NFW perturber produces." Direct test (`scripts/noise_decoy_control.py`,
`results/noise_decoy_control/`): inject a **purely non-physical Gaussian flux bump** — added
pixel brightness with no lensing physics whatsoever, ~1 pixel wide, placed in the same
0.6–1.3 θ_E annulus real subhalos occupy — into 100 clean `no_subhalo` Tier-0 images, at
three amplitudes (3σ, 6σ, 10σ of each image's own noise), and score both detectors at their
existing clean-image-calibrated thresholds (baseline FPR = 10% by construction). Family A's
smooth model is re-fit on each decoy image, as a real pipeline would have to; 93–96 of 100 fits
stayed reliable at every amplitude, so the numbers below are over essentially the full sample.

| decoy amplitude | U-Net FPR | Family A FPR |
|---|---|---|
| clean baseline | 10.0% | 10.0% |
| 3σ bump | 10.0% | 3.2% |
| 6σ bump | 13.0% | 3.2% |
| 10σ bump | **19.0%** | **2.1%** |

**The two methods respond in opposite directions.** The U-Net is mildly but monotonically
fooled as the non-physical blob gets brighter — a 10σ bump (very obvious to the eye) roughly
doubles its false-alarm rate. It is *not* fully fooled (a pure "any bright pixel = detection"
learner would fire on nearly every 10σ blob), so it has learned something shape-specific about
the real signature — but not enough to ignore a bright enough arbitrary bump. **Family A does
the opposite: its false-alarm rate falls *below* baseline, to 2–3%, at every amplitude, and does
not rise with decoy strength.** That is the correct behavior, and the reason is instructive: its
Δχ² asks specifically "does a *gravitational-lensing perturbation of the arc* improve the fit?"
An additive flux blob is not that kind of feature — no TNFW subhalo at any grid position helps
explain it — so the subhalo hypothesis is (correctly) not favored, and the added unmodelable
residual actually pushes Δχ² down.

**Why this matters for reading RQ1/RQ2 above:** it sharpens *what kind* of failure each family
has, and the two are exactly complementary to their designs:
- **Family A** recognizes genuine gravitational-lensing physics (it ignores the non-physical
  decoy) but **cannot tell *which* gravitational structure caused a distortion** — a real
  lens-shape multipole *is* a genuine gravitational distortion of the arc, so a subhalo
  hypothesis genuinely partly explains it, hence 76% false alarms there. Its problem is
  confusing one real gravitational source for another, not reacting to arbitrary anomalies.
- **The U-Net** does not "know" physics; it learned a pixel pattern. So it is robust to the
  global, smooth multipole distortion (which doesn't look like its learned localized pattern —
  RQ2's null result) but partly susceptible to a bright, localized, non-physical bump (which
  does). Its problem is the mirror image: it reacts to things that *look* locally like the
  pattern, whether or not they are gravitational.

Neither method is simply "fooled by noise"; each fails precisely where its own inductive bias
predicts. That is a cleaner, more mechanistic, and more useful characterization than the raw
false-positive rates alone, and it argues directly for the hybrid pipeline Tsang+2024 themselves
propose (ML "for flagging systems as a first pass... then... a more traditional fitting
procedure") — the two failure modes are nearly disjoint, so each could veto the other's
characteristic false alarm.

**Update (self-review fix, 2026-09-11): second seed + asymmetric dipole decoy**
(`--shape dipole`: a positive and a negative lobe 0.9 px apart along a random direction — the
shape a real subhalo residual has, with no gravitational deflection behind it).

| decoy | U-Net FPR 3σ / 6σ / 10σ | Family A FPR 3σ / 6σ / 10σ |
|---|---|---|
| Gaussian, seed 42 (original) | 10% / 13% / 19% | 3.2% / 3.2% / 2.1% |
| Gaussian, seed 43 | 7% / 9% / 12% | 3.2% / 3.3% / 4.2% |
| dipole, seed 42 | 11% / 16% / **21%** | 3.2% / 2.2% / 3.1% |

The direction is robust: the U-Net's false-alarm rate rises with decoy brightness in every run,
and the dipole — the fairer, subhalo-shaped test — fools it *most*, exactly as the "responds to
what looks like its learned pattern" reading predicts. The magnitude at 10σ has real seed
scatter (12–21%). Family A never exceeds ~4% and never rises with amplitude in any run.
Remaining caveats: n=100 per amplitude, Tier-0 sources only.

## Lens light (referee round 3, 2026-09-11): the realism axis every simulation omits — now tested once

Simulator: `LensLightConfig` (two-component truth: bulge n 3–5, R 0.25–0.5″, amp 0.5–2× the source's;
n=1 envelope R 0.7–1.3″ at 30–70% of the bulge; own q/PA offsets), `tier2_lens_light*` factories,
populations `data/lenslight_{no_subhalo,fixed60,multipole_m4_a3}` (300 each, seed 7). Family A fits ONE
Sérsic lens light jointly with the mass model (`fit.py`: 18-param vector, `_unpack3`,
`lens_light_image`), scan unchanged; `run_baseline_a.py` writes `lens_light_subtracted.npy`.
U-Net v0 (trained without lens light) scored on raw and on subtracted images.
`scripts/evaluate_lens_light.py` → `results/lens_light_experiment.json`, `results/baseline_a_lenslight/`.

| | no lens light | lens light, gated (χ²/N<10) | gate lifted |
|---|---|---|---|
| clean fits rejected | 5/300 | **141/300** (median χ²/N 9.4) | — |
| scan threshold @10% FPR | −10.6 | **+86** | +138 |
| scan FPR clean @ Δχ²>20 / >100 | 0.2 / 0% | **25 / 9%** | 31 / 15% |
| scan FPR multipole @ 20 / 100 | 58 / 33% | 53 / 25% | 55 / 35% |
| scan completeness 9–9.5/9.5–10/10–10.5/10.5–11 | 38/71/85/85% | 12/42/53/25% (n=17/12/17/8) | 24/39/48/68% |
| U-Net AUC c60 vs clean | 0.62 | 0.55 subtracted (0.53 raw) | |
| U-Net clean FPR at Tier-0 threshold | 10% | **86%** subtracted (77% raw) | |
| U-Net completeness, recalibrated (thr saturates 0.999) | 15/29/51/63% | 5/18/13/36% | |

**Upper bound (referee item 7, 2026-09-12): correctly specified double-Sérsic fit**
(`run_baseline_a.py --lens-light-components 2` → `results/baseline_a_lenslight2/`; U-Net scored on its
subtracted images): clean fits rejected 21/300 (vs 141 single, 5 none); median χ²/N 1.0; scan threshold
−7.5; **clean FPR @20/100: 0 / 0%**; multipole FPR 47 / 23% (vs 58/33 none, 53/25 single); completeness
37/56/79/83% (vs 38/71/85/85 none, 12/42/53/25 single); **U-Net AUC 0.67** (vs 0.62 none, 0.55 single),
clean FPR at Tier-0 threshold 21%, completeness 13/18/46/66% (vs 15/29/51/63 none). → Lens light itself
is not the problem; an under-flexible lens-light model is — Nightingale+2024's diagnosis for 16/34 real
candidates, reproduced.

**Family B with single-Sérsic lens light (referee item 9)** (`run_family_b.py` now subtracts the fitted
lens light and masks on the subtracted image; `results/baseline_b_lenslight/fitted/`): threshold on
max|δκ| **0.092 → 0.93** (10×; 1.47 gate-lifted); completeness **15/18/12/8/6/0%** gated, 11/10/13/13/13/26%
lifted — chance in every bin; multipole FPR 5% (everything swamped by the residual). The free-form
correction absorbs the lens-light residual everywhere, exactly as its mechanism predicts — it is the
family most damaged by imperfect subtraction.

**Reading.** (1) The first casualty is the fit: a single Sérsic on a two-component galaxy leaves
χ²/N ≈ 9 and the reliability gate throws away half the sample — you cannot run fit-then-scan at all
without a flexible lens-light model. (2) The residual is detected as a perturber: clean-lens FPR at
the literature's own Δχ²>20 goes 0.2 → 25%. (3) Completeness halves. (4) The multipole stops
mattering on top (58→53%): the lens-light residual is now the dominant unmodelled structure.
(5) The U-Net is at chance raw and after single-Sérsic subtraction — the misfit ring looks like a
perturbation (86% of clean lenses above its old threshold). (6) Family B is destroyed by the residual.
(7) The double-Sérsic fit recovers A and C almost entirely → the culprit is model flexibility.
Not done: real lens-light morphologies, colour-dependent subtraction, pixelized light models, a second
seed. Paper: Sect. 4.6 + Table `tab:lenslight` (none/single gated/single lifted/double) + Fig. 11
(`fig11_lenslight`: observed / single-subtracted / double-subtracted / true removed), Sect. 2.1
lens-light paragraph, Fig. 1 Tier-2 box, assumption (x), Conclusions (6).

## Family B (free-form potential correction) — prototype works; population runs done (2026-09-11)

The self-review named the absence of Family B — gravitational imaging / pixelized potential
corrections, the family behind essentially every *real* claimed detection — as the paper's
biggest gap. PyAutoLens ships the published family's own implementation
(`autolens.potential_correction`: linear δψ inversion on a `RegularDpsiMesh`, `CurvatureMask`
regularization, δκ = ½∇²δψ), so this is the real method's code, not a stand-in.

**Single-lens prototype** (`scripts/family_b_prototype.py`, `results/family_b/`, verified by
reading the JSON and looking at the maps): with an *oracle* macro model (lenstronomy's own
noiseless no-subhalo render — an idealization stronger than Family A's near-truth init, stated
as such) the inversion recovers a smooth δψ well centred on the subhalo and a compact positive δκ
lump on it, while the noise-sharing subhalo-free twin gives featureless δκ:

| idx | log₁₀M | M_sub / M(<θ_E) | peak \|δκ\| subhalo / twin | δκ-peak offset from truth | aperture mass recovered / true |
|---|---|---|---|---|---|
| 332 | 10.28 | 6.2% | 0.57 / 0.07 | **0.06″** (< 1 mesh px) | 4.1e9 / 4.35e9 |
| 188 | 9.00 | 0.3% | 0.097 / 0.078 | 0.11″ | 5.3e8 / 3.9e8 |
| 451 | 9.56 | 0.9% (on a cusp) | 0.25 / 0.09 | 0.55″ (δκ dipole/ringing) | 2.2e9 / 1.1e9 |
| 104 | 10.67 | 12.2% | 2.5 / 0.10 | 0.39″, checkerboard, corr 0 | fails |

Two things worth knowing already: (1) **the first-order linear inversion breaks for very massive
perturbers** (idx 104, 12% of the lens mass shifts the whole ring by ~0.7 px; the literature uses
the iterative re-ray-traced variant there) — a real Family-B characteristic that will show up as
a top-mass-bin completeness deficit; (2) **regularization strength matters exactly as
Galan+2022 warn**: the twin's peak |δκ| spans two decades over λ = 10–10⁶, the lump only
localizes for λ ≥ 10⁴, and per-lens max-evidence λ is "no correction" on every twin and 10² on
the failure case — so λ must be fixed by rule (1e5 chosen), not selected per lens. Cost: 0.1–0.7 s
per inversion, 2–4 s per lens including a 6-λ sweep — population runs are cheap.

**Population runs (done, 2026-09-11; full write-up in `results/baseline_b/NOTES.md`):** same
300-image subsamples and seeds as Family A (so A and B are compared on the *identical* lenses and
the identical macro fit); `fitted` variant (macro model = Family A's own near-truth-initialized
smooth fit) as primary, `oracle` (exact macro model) as the inversion's ceiling; statistic =
max |δκ| at fixed λ=1e5 calibrated at 10% FPR on `no_subhalo` (λ=1e4/1e6 stored as a band);
position = δκ peak; mass = aperture δκ mass in 0.32″. 0 errors in 3000 inversions; 1.4 s macro
fit + 0.4 s inversion per lens. Output mirrors Family A's `scan_results.jsonl`, so
`evaluate_baseline_a.py`, `evaluate_baseline_a_fixed_threshold.py` and
`plot_localization_accuracy.py --b-root` ran unchanged.

**Headline, same images, same protocol (10% FPR on `no_subhalo`, seed set 0; oracle in brackets):**

| family | c60 9.0–9.5 | 9.5–10 | 10–10.5 | 10.5–11 | c15 9.0–9.5 | paired flip | FPR a=0.01 / 0.03 | loc. ≤ 2 px | mass (localized) |
|---|---|---|---|---|---|---|---|---|---|
| **B potential correction (fitted, λ=1e5)** | **75.0** [90.6] | **88.1** [97.7] | **93.8** [100] | 100 (13/13) | 12.5 | 58:10 (5.8:1) | **39 / 86%** [62 / 92] | **51.5%** (median 0.14″) | −0.64 dex, 67/69 low, MAD 0.23 |
| A parametric scan (seed 0) | 37.5 | 71.4 | 85.4 | 84.6 | 35.7 | 21:4 (5.3:1) | 31 / 75% | 13.9% | −1.0 dex, 15/15 low |
| C U-Net (4-seed mean) | 14.7 | 28.6 | 51.0 | 63.4 | 12.7 | 199:30 (6.6:1) | 11 / 11% | 56.7% | not estimated |

**Reading it (paper Sect. 3.3, 4.1–4.3, Fig. "Family B", Discussion 5.1):**
1. **Most sensitive detector.** Given the identical macro fit, B doubles A's completeness at
   10^9–10^9.5 (75 vs 38%) and beats the U-Net 2–5× above 10^9. Below 10^9 all three sit at the
   10% floor. The oracle macro model adds 10–20 points per bin — the ceiling of the linear method.
2. **Not immune to concentration.** c=15: 75 → 12.5% at 10^9–10^9.5, 88 → 65% at 10^9.5–10^10;
   paired flip 58:10, same sign and similar ratio as C (6.6:1) and A (5.3:1). Mechanism is
   physical: a diffuse clump gives a lower, broader δκ bump and a *peak* statistic is intrinsically
   concentration-dependent. **Three families, three routes, same qualitative result** — the
   strongest form of the RQ4 claim the paper now makes.
3. **Worst confounder response of the three:** 39 / 86% (oracle 62 / 92%). A free-form δψ
   absorbs *any* smooth-model misspecification; an m=4 multipole is a legitimate potential
   perturbation to the inversion; a peak-height statistic can't tell a compact lump from an
   extended m=4 pattern. **No λ escapes it**: λ=1e4 → 22 / 50% FPR but −40 points of completeness
   at 10^9–10^9.5; λ=1e6 → 64 / 97%. A shape/extent criterion on δκ is the obvious fix and is not
   in the published recipe. (Decoy control for B: see "Round 5" — 67–76% at 10σ, the most decoy-prone family.)
4. **Localizes like the U-Net:** 51.5% within 0.16″ (median 0.14″ = one mesh pixel), 3.7× Family
   A on the same macro fit. B's detections are *positions*; A's are *hypotheses*. Find-and-localize
   ledger: B 69 correct vs 228 false alarms; A 15 vs 198; C 149 vs 95.
5. **Mass:** aperture mass is projected mass in 0.32″, not M200 → tight −0.64 dex offset
   (MAD 0.23, oracle −0.47) — *calibratable*, unlike A's frozen-macro-model bias. Still: **no family
   recovers the subhalo mass function as published** — one doesn't try, one is biased by a fixed
   aperture, one by a frozen macro-model.
6. **Oracle is a ceiling for detection, not localization** (13.7% localized!): with the exact macro
   model the residual still holds the subhalo's *long-range* deflection, which the first-order
   inversion smears across the mesh; *re-fitting* the macro model (the `fitted` pipeline) absorbs
   that part and leaves the compact residual → localization improves to 51.5%. Same physics as
   the Family A joint-refit finding, seen from the other side: the macro model absorbs the
   perturber's smooth part whether you want it to or not.
7. **Top mass bin is out of the linear regime**: 27/40 lenses with log M ≥ 10.5 are excluded by
   the reliability rule in `fitted`, and the oracle mislocalizes 100% of them (residual χ²/N up to
   220). The iterative `IterFitDpsiSrcImaging` is the published fix; left as future work, stated.
8. Published-style absolute operating points are fit-limited, not noise-limited: the null's top two
   values are two macro-fit failures that passed the χ²/N<10 rule (0.26, 0.69 vs oracle max 0.13).

**Replicates (seed sets 200 and 300, `results/baseline_b_seed{200,300}/fitted`; 3-seed aggregate in
`results/baseline_b/aggregate_fitted_3seeds.json`, 0 errors / 6000 inversions total):** c=60
completeness 13.1±2.5 / 19.4±3.8 / **65.3±9.2** / **87.6±1.4** / **94.4±2.7** / 95.8±7.2%; c=15
9.3±3.5 / 9.1±1.3 / 16.8±9.7 / 55.8±8.4 / 77.2±6.0 / 94.6±2.4%; flip **55±10 : 8±3** (58:10, 63:9,
43:4); confounder FPR **38.7±3.4 / 85.8±1.2%**; localized ≤ 0.16″ **54.7±5.3%** (51.5, 51.8, 60.9);
aperture-mass error −0.65±0.01 dex. Seed 300 has more macro-fit exclusions (62/276 vs 42–44) — see
the conservative-completeness note below. Every seed-0 statement above holds at all three seeds;
the paper reports the 3-seed means.

### Conservative completeness (referee fix, 2026-09-11): excluded fits counted as misses

`scripts/conservative_completeness.py` → `results/conservative_completeness.json`. The χ²/N ≥ 10
exclusion removes 42–62 of 268–276 subhalo-bearing lenses per seed set, **27–35 of them in the
10^10.5–10^11 bin** (a perturber of several % of the lens mass breaks the smooth fit). Counting
every excluded lens as a miss:

| family (c=60) | 10.0–10.5 reported → conservative | 10.5–11.0 reported → conservative | bins < 10^10 |
|---|---|---|---|
| A c=15, seeds 0/200 | 85→66%, 97→84% | 85→28%, 83→33% | ≤ 4 pts |
| A c=60 / c–M | 94→73%, 85→66% | 100→32%, 85→28% | ≤ 2 pts |
| B fitted, seeds 0/200/300 | 94→73%, 97→84%, 92→62% | 100→32%, 100→39%, 88→27% | ≤ 7 pts |

Neither bound is "the" completeness: a smooth fit that fails at χ²/N ≥ 10 on a correctly specified
source is itself a substructure signal (it happens to 5–18 of 300 subhalo-free lenses) and would be
followed up, not discarded. The paper states both bounds (Sect. 2.3, 4.1); figures show the
reported bound; cross-family comparisons are unaffected (A and B share the exclusion on the same
lenses; the U-Net has none).

## Round 5 (2026-09-12): the last "not run" cells filled

**Family B on the non-physical decoys** (`scripts/make_decoy_populations.py` materialises the
bit-identical decoys of `noise_decoy_control.py` as `data/decoy_{gaussian,dipole}_s42_a{3,6,10}`;
`run_family_b.py` fitted variant; `scripts/evaluate_decoy_family_b.py` →
`results/noise_decoy_control_familyB/results.json`, threshold 0.0919 = B's own 10%-FPR value):

| decoy amplitude | 3σ | 6σ | 10σ |
|---|---|---|---|
| Gaussian bump — B FPR | 12.5% | 28% | **67%** |
| dipole — B FPR | 17% | 49% | **76%** |
| Gaussian, **seed 43** (round 8) — B FPR | 14% | 50% | **63%** |
| (U-Net, same decoys) | 10–11% | 13–16% | 19–21% |
| (Family A, same decoys) | 3% | 2–3% | 2–4% |

B is the family most fooled by a non-physical feature: a free-form δψ times the source gradient
reproduces an additive blob wherever the arc has a gradient. **Correction to the paper's earlier
framing:** what ignores a non-physical feature is not "a physical statistic" but a *parametric*
one — only the rigid TNFW template has no way to represent a blob. Sect. 4.2, 5.1, Fig. 4b (green
curves), Fig. 11 cell, Conclusions (2) updated.

**Family B with the double-Sérsic (correctly specified) lens-light fit** (`run_family_b.py
--lens-light-components 2`, `results/baseline_b_lenslight2/`): gate rejects 21/300 clean; threshold
back to 0.094 (0.93 single); completeness 9/5/49/72/94/97% (none: 14/18/75/88/94/100; single:
15/18/12/8/6/0); multipole FPR 69% (86% none). Recovered like A and C. Table 5 "double" column filled.

**Second lens-light seed set** (`data/lenslight_*_s8`, seed 8; `results/baseline_a_lenslight_s8/`,
`results/lens_light_experiment_s8.json`): single Sérsic — gate 148/300 clean (141), median χ²/N 9.9
(9.4), threshold +99 (+86), clean FPR @20/100 26/10% (25/9), multipole 57/33% (53/25), completeness
10/35/64/60% (12/42/53/25; n=10–20 per bin), U-Net raw AUC 0.49 (0.53), subtracted 0.52 (0.55),
clean FPR at Tier-0 threshold 83% (86%). Reproduces seed 7 within the small-n scatter; the paper
quotes seed-set ranges. **Double-Sérsic seed 8** (`results/baseline_a_lenslight2_s8/`): gate 12/300 clean (21 at seed 7), threshold −9.3 (−7.5), clean FPR @20/100 0.3/0% (0/0), multipole 49/26% (47/23), completeness 25/55/89/81% (37/56/79/83), U-Net AUC 0.63 (0.67), completeness 17/28/51/56% (13/18/46/66). Both seed sets agree: the correctly specified light model recovers A and C to within ~10 points of Tier 0.

## Reference audit (2026-09-12)

`paper/fetch_refs.py` downloaded every cited arXiv paper into `paper/refs/` and `paper/refs/README.md`
records the claim-by-claim check. Two numbers the manuscript attributed to Nightingale+2024 ("8 of 54",
"16 of 34") do **not** appear in that paper and were replaced with its actual statements (54 lenses, five
candidates, four power-law false positives, lens-light residuals removed by hand). Everything else
attributed to a source was found verbatim or in a table. Lesson: numbers quoted from the literature
must come from the downloaded text, not from memory.

## Synthesis (2026-09-11) — robustness, generalization, and which assumptions are actually wrong

Pulling every result above into one place. A spatial view of this section's central claim (905
real simulated clumps vs. both methods' correct finds/false alarms, stricter localization-aware
definition, same 2D sky-plane coordinates lensing actually measures) is at
`data/sanity/explainer/positions_for_viz.json`, generated by `scripts/extract_positions_for_viz.py`,
rendered as the "Clump Atlas" artifact.

**Family A (classical parametric scan) — accurate-looking by a permissive classification
statistic, but rarely actually correctly localized, and brittle to anything its model doesn't
include.** At Tier 0, with a correctly-specified smooth source, its *classification* completeness
reaches ~91% at high mass, and its forward model is cross-checked directly against real
PyAutoLens (r=0.94-1.00). But the localization check above is the more important correction:
only 13.9% of its nominal "correct finds" are actually within 2 pixels of the truth, and under a
find-*and*-localize definition it's outnumbered by false alarms more than 13:1 on the confounder
population. Its detection statistic (Δχ²: does adding a subhalo improve the fit) cannot
distinguish "there's a real subhalo" from "my model is missing some other physical effect" — so
it inflates 3-7.6x under an ordinary lens-shape confounder that has nothing to do with dark
matter (and does so in a structured, systematic way — false alarms cluster at its scan's fixed
grid angles, not randomly). On Tier 1 (real COSMOS source), its own smooth fit becomes so
misspecified that the whole statistic turns heavy-tailed and noise-dominated (two seeds gave
opposite-direction confounder-FPR results) rather than cleanly inflated. **This is a
statistical-framework problem: the method's own logic conflates "unmodeled complexity" with
"dark matter," so it fails wherever real galaxies are more
complicated than its model** — which is everywhere, to some degree.

**Family C (U-Net) — robust to the one confounder tested, completely brittle to distribution
shift in its own training assumptions.** It barely notices the same lens-shape wrinkle that
fools Family A (10.0% → 11.3%/10.5%, not resolved from baseline) — a spatially-local pattern
detector doesn't respond to a global, symmetric shape change the way a global-evidence
statistic does. But push on either of the two realism axes actually tested and it collapses
completely, not gracefully: realistic subhalo concentration (AUC 0.62 → chance, same images)
and real source structure (AUC 0.62 → 0.48, chance, even on the easiest, biggest true
subhalos). **This is a classic ML generalization problem: it learned a narrow pixel pattern
that only matches the narrow training distribution it saw**, and training directly on the
harder, more realistic distribution didn't produce a working detector either at this budget —
it broke training outright (see the RQ4 extension above).

**Family D (population SBI) never worked at all**, at either attempt, even on the easiest
Tier-0 setup — a different failure again (the specific per-lens-scalar framing tried here
seems to be the wrong tool, not a robustness problem in the RQ1/RQ4 sense).

**The meta-finding, and the more general claim than any single result above: every method that
"worked" in this pilot only worked at its own author's convenient operating point, and the two
methods that did work fail for *different underlying reasons*** — one a statistical/framework
problem (can't separate "unmodeled" from "dark matter"), the other a distribution-shift problem
(learned pattern doesn't transfer). That two independently-motivated failure mechanisms both
converge on "published performance is contingent on assumptions that don't hold" is a more
robust claim than either family's result in isolation.

**Which specific assumptions are actually wrong, concretely:**

1. **"Subhalos are dense" (c≈60).** Wrong per the field's own ΛCDM concentration-mass relation
   (Dutton & Maccio 2014 predicts c≈8-15 for this mass range — at or below the literature's own
   "pessimistic" c=15 ablation). This single assumption alone explains most of the reported
   detection completeness in the published literature (RQ4).
2. **"The background source is a smooth blob" (single Sersic).** Wrong for real galaxies, which
   are clumpy and asymmetric (COSMOS/HST imaging). Correcting it breaks the ML detector outright
   and destabilizes the classical scan's statistic (Tier 1).
3. **"A fit improvement from adding a subhalo means a subhalo is there."** Wrong whenever any
   other unmodeled complexity exists — a lens-shape wrinkle, source clumpiness, anything the
   smooth model doesn't include (RQ1, RQ2).
4. **"The lens galaxy's shape is a simple ellipse + external shear."** Wrong in detail for real
   galaxies (there's always some higher-order shape term) — exactly what inflates Family A's
   false-positive rate.
5. **"Test images will look statistically like training images"** — the general, usually-implicit
   ML assumption behind every detector in this literature, and the one silently violated by both
   concentration realism and source realism, with nobody checking it explicitly before this pilot.
6. **"A classification score crossing threshold means the position was found."** Wrong, at least
   for Family A: 86% of its Tier-0 "correct finds" are not within 2 pixels of the true subhalo.
   Reported completeness numbers across this literature (this pilot included, until this check)
   routinely conflate "confidently said yes" with "correctly localized" — the two are not the
   same claim, and treating them as interchangeable is itself an unstated assumption worth
   naming.
7. **"If a parametric scan locates a subhalo, its fitted mass is a reasonable estimate."**
   Wrong: even the 15 cases where Family A's position guess was correct all had the wrong mass
   (100%, median −1.0 dex), traced to the same fixed-concentration assumption that drives its
   confounder false-positive rate and its lack of a completeness collapse under RQ4. One wrong
   assumption (concentration) contaminates position-detection statistics, mass estimates, and
   confounder robustness simultaneously — it is not three independent failure modes, it is one
   assumption with three visible symptoms.
8. **"Only subhalos physically inside the lens galaxy can mimic a subhalo signal."** Untested
   here, but not hypothetical: real line-of-sight halos at a different redshift than the lens can
   produce a similar lensing signature (a known degeneracy; Şengül 2022's widely-discussed "first
   dark perturber" claim was later argued to likely be exactly this). Not built in this pilot
   (`problem_statement.md`'s Tier 2), but a strong candidate next confounder given how much of
   Family A's failure mode above already comes from a wrong assumption about *one* physical
   parameter (concentration) of an otherwise correctly-placed subhalo.
9. **"A completeness curve averaged evenly across mass bins describes real-world performance."**
   Wrong — see the next section. Our test populations sample mass uniformly-in-log purely so
   every 0.5-dex bin gets enough images to measure; nature does not. Weighting the *same*
   per-bin numbers by the real CDM subhalo mass function collapses every headline completeness
   figure by more than half.

### The subhalo mass function itself is a simulation assumption, not (yet) a real measurement — and re-weighting by it changes every completeness number

Every mass-dependent result above (RQ4, RQ1) reports completeness **per 0.5-dex mass bin**,
which silently treats each bin as equally represented. It isn't. The subhalo mass function used
throughout this literature (dN/dM ∝ M^−α, α≈1.9 — Dhanasingham+2025's own prior, cited in
`fulltext_findings.md`) is a *prediction of N-body ΛCDM simulations*, not yet an independent
real-data measurement at this mass range (see the discussion below) — but it is the field's own
best working assumption, and it says the population is **overwhelmingly dominated by the
smallest masses**: of six 0.5-dex bins from 10⁸–10¹¹ M☉, the smallest single bin (10⁸–10⁸·⁵)
contains **64.6%** of all subhalos; the top two bins combined contain **1.4%**.

Re-weighting the already-computed per-bin completeness numbers by this realistic mix
(`scripts/reweight_by_mass_function.py`, no new simulation or training required) instead of a
flat bin average:

| | naive bin-average completeness | CDM-mass-function-weighted completeness |
|---|---|---|
| U-Net, c=60 | 29.6% | **12.2%** |
| U-Net, c=15 | 11.7% | **9.6%** (≈ chance) |
| Family A, c=60 | 52.7% | **19.5%** |
| Family A, c=15 | 46.5% | **15.9%** |

**Every method's population-level completeness drops by more than half once the mass function's
real shape is accounted for**, because the easy, high-mass bins that drive the impressive
headline numbers (Family A: 91% at 10.0–10.5, U-Net: 63% at 10.5–11.0) are also the *rarest*
subhalos in nature. A detector that looks strong "at high mass" is, in a real ΛCDM-abundance
population, mostly being asked to find the low-mass subhalos it's worst at. This is a cheap,
honest correction every completeness-vs-mass plot in this literature (and this pilot, until now)
should carry alongside the per-bin curve, not instead of it.

**On whether the mass function itself is safe to assume:** it rests on N-body gravity plus
ΛCDM initial conditions — a framework independently well-supported by unrelated data (CMB,
large-scale structure) — but the specific slope at this exact mass range is not yet an
independent real measurement; that measurement is the long-term scientific goal this whole
subfield (and, at one remove, this pilot) works toward. The available real, if indirect,
supporting evidence: Milky Way satellite counts (only at the higher, galaxy-hosting mass end),
a handful of individually claimed real detections (each individually uncertain — see item 8
above), and Nightingale+2024's real 54-lens sample, whose simple-model false-candidate rate
(0.35–0.44/lens) was **10–20x above the CDM-predicted rate** (0.025–0.1/lens) — itself evidence
of a too-simple model manufacturing detections, on real data, echoing this pilot's own synthetic
RQ1 finding.

## RQ6 — reproducibility ledger

| | Tsang+2024 specified | What we did | Why |
|---|---|---|---|
| Architecture | "U-Net image segmentation" | U-Net, 4-level encoder/decoder, base width 16 | No further detail given in the text |
| Training images | 5×10⁵ | 8,000 | Compute/time budget for a first pass |
| Loss | not specified | per-pixel BCE, pos_weight=150 | Standard choice for the class imbalance |
| Detection score | max-pixel probability | same | Directly stated in their text |
| Success mask | "within 2 pixels of true center" | 2-px-radius disk target | Directly stated |
| Reported completeness, 10⁹–10⁹·⁵ M☉, c=60, coarsest (80 mas) pixel scale | 33.6% | 11.8% | At ~1.6% of their training-set size; the *direction and shape* of the mass-dependence and the concentration collapse both reproduce; the *absolute level* does not yet, as expected |
| Code released | no | — | This reimplementation is the first public one, as far as the methods survey found |

**Bottom line for RQ6:** the paper's architecture and evaluation protocol were reproducible
from the text alone. Its absolute numbers were not reproduced at 1/60th of its training scale —
an expected and honest gap, not a contradiction of their result.

## Caveats (what would firm each finding up) — refreshed 2026-09-11 after all runs

1. **Pilot-scale statistics.** n=4 training seeds for the Tier-0 U-Net, n=3 for Family A at Tier 0
   and for Family B, n=2 for Family A at Tier 1, the Tier-1 U-Net, the 30k scale-up and the
   concentration-diversity training; n=2 (Gaussian) + 1 (dipole) for the decoy control; the joint-refit
   scan is 100 lenses at one seed. Enough to separate the large effects (RQ4 collapse, RQ1's
   7x FPR gap, the 44/44 mass bias) from seed noise; not enough to bound small effects (RQ2's
   null, Family A's Tier-1 confounder FPR, which two seeds put on opposite sides of baseline).
2. **Training sets are 1.6–6% of the literature's** (8,000 / 30,000 vs 5×10⁵). Explains the
   absolute completeness gap in the RQ6 ledger (11.8 → 20.6% vs 33.6% as data grows); the 30k run
   shows more data does not change any qualitative result.
3. **Family A carries three stated idealizations**: smooth fit initialized within 15% of the
   truth (a real blind fit is harder); a fixed scan concentration (c=15 by default — now also run
   at c=60 and at the c–M relation, which changes completeness and confounder FPR but *not*
   localization or the mass bias); and a **frozen macro-model** during the scan, which is the
   cause of the −1 dex mass bias (joint refit removes it) and is a shortcut the published
   pipelines do not take. The 72-hypothesis grid caps localization and produces the radial-spoke
   false-alarm pattern; a finer grid would narrow that gap at proportional cost.
4. **The PyAutoLens cross-check is exact for the TNFW subhalo and for near-circular macro
   lenses**; for elliptical lenses the two codes' θ_E conventions differ by a slope-dependent
   factor (0.98 at q=0.7, 0.94 at q=0.5), so the r=0.94–1.00 macro agreement is stated for
   lenses whose macro-models agree, not for all nine.
5. **Family D is a per-lens stand-in**, not the hierarchical population method the literature
   actually uses; its two failures rule out "more data/capacity" for *this* framing only.
6. **Tier 1 is a training-recipe result.** The U-Net's collapse used the Tier-0 recipe unchanged;
   Family A's Tier-1 confounder numbers are noise-dominated at n=300 (heavy-tailed Δχ² from a
   misspecified source). Neither is evidence the Tier-1 signal is unlearnable.
7. **The decoy control is Tier 0 only, n=100 per amplitude** — two Gaussian seeds plus one
   dipole run; Family B was run on the identical decoys in round 5 (one seed set of 100 lenses, a second Gaussian seed added in round 8).
8. **The real-lens test used our own corner-pixel noise estimate** (two arrays of Şengül+2022's
   reduction are not public) and a single-Sérsic source that the χ²/dof≈4 fit shows is inadequate;
   it demonstrates the misspecification, not a reproduction of their result.
9. **The CDM mass function used for re-weighting is a simulation prediction**, not yet an
   independent real-data measurement at 10⁸–10¹¹ M☉ — it is the field's working assumption,
   used here on its own terms.
10. **Family B idealizations**: analytic Sérsic source held fixed during the inversion (the joint
    source+δψ inversion is the full method), one mesh scale, one fixed λ (1e4/1e6 band reported),
    first-order linear inversion only (top mass bin out of regime: 27/40 excluded), macro model =
    Family A's near-truth-initialized fit. Two seed sets. Not run on the decoys or on Tier 1.
11. **Not tested at all**: Family E (field-level statistics), line-of-sight halo populations,
    lens light and its subtraction residuals, instrument transfer (RQ5), posterior
    coverage/calibration metrics (RQ3). All are in the released suite's design and none in this
    paper.

## Next steps

**Toward submission (paper 1, A&A):** swap `aa.cls` v9.0 → current v9.4 from aanda.org; fill the
`\todo{}` facts in `paper/main.tex` (e-mail, repository URL, acknowledgements + AI-assistance
disclosure, dates); tidy bibliography journal strings; one outside read by someone in the
subfield; arXiv endorser for astro-ph; publish the suite (top-level README, environment spec,
seeds table, "regenerate every figure" path) at the personal GitHub repository.

**Cheap robustness additions (hours each):** a finer Family-A scan grid (does localization
improve as predicted? — the mass bias will not, that is the frozen macro-model); a joint-refit
variant of the full scan (re-fit the macro model at each of the 72 cells) to measure what the
shortcut costs in *detection*, not only mass; a third/fourth Family-A seed at Tier 0; Family B
on the decoy control and on Tier 1; a δκ shape/extent criterion for Family B's confounder FPR;
the iterative `IterFitDpsiSrcImaging` for Family B's top mass bin.

**Substantive extensions (each a real piece of work, candidates for paper 2):** a flexible
source model (pixelated / shapelet) for Family A so Tier 1 and real data can be fit properly —
the single biggest unlock, since every source-misspecification failure traces to it; a
line-of-sight halo population (multi-plane) as the next confounder, directly relevant given
B1938+666; the hierarchical multi-lens Family-D baseline the literature actually uses; curriculum
training at realistic concentrations (start at c=60, anneal toward the ΛCDM relation) and
source-aware preprocessing for a Tier-1 U-Net; posterior coverage (`tarp`/SBC) once any
population method works; instrument transfer to Euclid/Rubin resolution; scaling training to
~10⁵ images to see whether the RQ6 completeness gap closes.

## 2026-09-12 (late): A&A class upgraded to v9.4

- `paper/aa.cls` + `paper/aa.bst` replaced by the official macro package v9.4 (aa.cls dated
  2025-11-27; `macro-latex-aa.zip` downloaded by hand from aanda.org — the site returns 403 to
  curl). The 2016 v9.0 files were deleted at the user's request (no rollback copy).
- Consequences under 9.4: `\email{}` in `\institute` is ignored at compile time (reserved for
  metadata extraction) → e-mail moved to `\corrauth{}` in the `\author` line, rendered as the
  "Corresponding author" footnote; margin line numbers are hard-coded (referee copy, intended);
  the 28 natbib "multiply defined" warnings of v9.0 disappear (9.2 "fix hyperlinks").
- 9.4 counts the abstract: it was **461 words**, not the ~300 previously logged (the earlier
  count evidently missed the structured-abstract fields). Rewritten to ≤300 by the class's own
  count (all headline numbers kept; 'customary' and hedging words dropped; conclusions shortened).
- `\titlerunning{Assumptions under stress: a common-suite test of substructure detectors}` and
  `\authorrunning{R. Ramteke}` added (without them 9.4 prints a request into the body text).
- Build: 17 pages, 0 errors, 0 undefined, 0 overfull, 0 class warnings.

## 2026-09-12 (night): round 10 — fourth external report (7/10, "minor bordering on moderate")

Report's two gating points, both about the newest material, plus defects the revision introduced.

### Signal variable (point 2) — recomputed on the FULL 1 000-lens populations, three variables
`scripts/completeness_vs_signal.py` now records, per subhalo lens, log10 M_proj(<0.1"), log10 M_proj(<0.2")
and the perturbation S/N (sqrt Σ((I_full − I_no_sub)/σ)² from the noiseless twins + lenstronomy's noise
model), with detections from `results/baseline_a_full`, `results/baseline_b_full/fitted` and the U-Net (every
lens). Bins need ≥20 lenses of each population. Output `results/completeness_vs_signal.json`,
`paper/tables/signal_variables.tex`, Fig. `fig13_signal` (2×3: S/N top, M_proj(<0.2") bottom).

The referee was right about 0.1": at equal aperture mass the c=15 population is detected MORE often
(A +16 pts mean, z=6.7 in 8.0–8.5; B +5, z=2.6) — most of a diffuse perturber's deflection sits outside 0.1".
The honest picture (c15 − c60 at equal signal, mean over bins / largest |z|):

| family | M_proj(<0.1") | M_proj(<0.2") | S/N_pert |
|---|---|---|---|
| A scan | +16 / 6.7 | +6 / 4.0 | −3 / 2.0 (10 bins) |
| B pot. corr. | +5 / 2.6 | −5 / 1.8 (4 bins) | −15 / 5.2 |
| C U-Net | −2 / 2.7 | −5 / 6.5 | −14 / 7.2 |

So: A's curves coincide against total perturbation S/N (its statistic is a total χ²); B's coincide against the
compact projected mass within 0.2" but NOT against S/N (its statistic is a δκ peak, i.e. compact mass); the
U-Net's coincide against neither (53% vs 8% at log S/N 2.75–3.0). The round-9 wording "the two physical
methods coincide at equal M_proj(<0.1")" was wrong in detail and read off 300-lens subsamples; replaced in
abstract, Sect. 4.1, Fig. 13 caption, Table tab:signal, Conclusion 1, Sect. 5.1.

### Macro-basin control (point 1) — `run_baseline_a_joint.py` rewritten
The round-3 "polish" control (continue the converged smooth fit) only tests local convergence. New control:
every one of the 24 cells' joint macro solutions is taken, the perturber removed, and the 13 macro parameters
re-optimised from there (max_nfev 40×14). `delta_chi2` is now min(smooth χ² over all starts) − min_cell χ²_joint;
the raw form is kept as `delta_chi2_vs_unpolished_smooth`, every cell's solution in `cells`. Runs on the same
first-100 lenses (seeds 1/3/0; index 32 of the multipole population skipped as before; 900 s per-lens wall-clock
guard) → `results/baseline_a_joint_basin_c15/`. Evaluator `evaluate_joint_vs_frozen.py --joint-root ...` adds
`c15_raw_statistic` and a `basin_control` block. Results appended below when the runs finish.

### m=4 fix caveat — `fit.py` generalised to several multipole orders
`MULTIPOLE_ORDERS` / `set_multipole_orders()`, layout inferred from the vector length (13 + 5 n_ll + 2 n_mp,
all nine combinations distinct). `run_baseline_a.py --macro-multipole --macro-multipole-orders 3,4`.
New population `data/multipole_m3_a3` (n=1000, generation seed 2 = same macro-lenses and sources as
`multipole_m4_a3`). Runs (300-lens seed-matched subsamples): bare EPL on m3 (`results/baseline_a_m3truth`),
EPL+m4 on m3 (`results/baseline_a_mpmacro/multipole_m3_a3`), EPL+m3+m4 on clean / m4 a1 / m4 a3 / m3 a3 / c60
(`results/baseline_a_mp34`). `scripts/evaluate_multipole_orders.py` → `paper/tables/mpmacro.tex` (Table
tab:mpmacro is now generated; includes the gate-rejection row the referee asked for: clean 5 → 10 of 300 with
the m=4 term). Results appended below when the runs finish.

### Defects fixed
Fig. 11 (summary grid) B/Tier-1 cell was still "not run" (the earlier edit had hit a different string) → "281/300
gated; null ×20, chance". Fig. 3 subtitles shortened ("full populations, n = 1 000") so neighbouring panels no
longer overprint; caption's "n=3" for A and B replaced. Sect. 5.1 duplicated sentence merged. Sect. 4.2 paragraph
2 (~750 words) split into four. Page-1 running-head request and the natbib warnings were already gone with aa.cls
9.4; abstract is counted by the class (≤300). Tsang+2024 is still arXiv-only (Crossref has no DOI other than the
arXiv one; checked again tonight) — the bib entry is correct as an e-print.

### Multipole-order results (runs finished 22:50)
`results/multipole_orders_summary.json`, Table tab:mpmacro (generated). Same 300-lens seed-matched subsamples; each
column's 10%-FPR threshold from its own clean scan (bare −10.6, +m4 −12.0, +m3+4 −11.4).

| | bare EPL | +m=4 | +m=3+4 |
|---|---|---|---|
| χ²/dof median, m=4 truth a=0.03 | 2.20 | 0.99 | 0.99 |
| χ²/dof median, m=3 truth a=0.03 | 3.35 | 2.91 | 0.99 |
| FPR m=4 a=0.01, cal / >20 | 30.7 / 11.7% | 8.5 / 1.1% | 8.5 / 0.7% |
| FPR m=4 a=0.03, cal / >20 | 75.0 / 57.6% | 9.7 / 0.0% | 8.6 / 0.0% |
| **FPR m=3 a=0.03, cal / >20** | **86.1 / 75.8%** | **82.7 / 68.4%** | **8.5 / 0.0%** |
| fits rejected: clean / m4 / m3 | 5 / 36 / 69 | 10 / 23 / 63 | 10 / 20 / 16 |
| compl. c=60: 9–9.5 / 9.5–10 / 10–10.5 / 10.5–11 | 38 / 71 / 85 / 85% | 32 / 76 / 78 / 86% | 27 / 51 / 68 / 82% |
| localized ≤2 px | 14% | 14% | 13% |

Reading: the m=4 fix is form-specific (matched truth and template) — it does nothing against an m=3 truth (83% vs
86%). General azimuthal freedom (m=3+m=4) removes both confounders (8.5–8.6%, 0% at >20) but costs 11–20 points
of c=60 completeness between 10^9 and 10^10.5 (a free m=3 term absorbs part of the subhalo's asymmetric signature)
and doubles clean-lens gate rejections (5 → 10 of 300, same as m=4 alone). Written into Sect. 4.2 ¶1, abstract,
Conclusion 2. The round-9 sentence "removes the false positives entirely at no cost in completeness" is now
qualified accordingly.

### Macro-basin control results (joint runs finished 23:00; `results/baseline_a_joint_basin_c15/`, 1489–2938 s per population at 3 workers)
Reliable fits: 97 clean, 92 multipole, 86 c=60 (3 lenses hit the 900 s per-lens guard with incomplete cell scans; their Δχ² is a lower bound).
The referee's scenario does occur, rarely: a joint solution leads to a BETTER smooth optimum in 1/97 clean, 1/92 multipole and
3/86 c=60 lenses, with gains of 1.3–2.5×10⁴ in χ² (the original smooth fit was stuck). Every other lens returns to its
original optimum to within 10⁻⁴. The round-3 "polish" control (continue the smooth fit from its own solution) caught only
one of these five — the referee was right that it tested only local convergence.
Basin-corrected statistic (min smooth χ² over 25 starts − best joint χ²): threshold +7.1 (was +7.2); multipole FPR 100 / 97.8 /
90.2 % at 10% cal / >20 / >100 (unchanged); median Δχ² 725 (was 737); clean FPR at >20 0 % (raw 1 %: the one clean basin lens
was the false positive); c=60 detected 61/75 (raw 62/75), localized 15 % (9), mass error of localized +0.5 dex (0/9 low).
Paired joint − frozen Δχ² on the multipole lenses: median +652, positive for 87/87. Table tab:joint, Sect. 4.2 ¶3, the
summary-grid cell and the abstract clause ("raises the scan's rate to 100 %") stand, now with the proper control behind them.
Gate observation from the same control: of the 24 lenses the three joint runs reject for χ²/dof ≥ 10, five have a joint
solution whose smooth re-optimisation lands at χ²/dof 1.0–2.1 (clean 278: 40.7 → 0.97; multipole 60: 103 → 2.05; c=60 56: 55 → 1.07,
227: 29.7 → 1.07, 347: 150 → 1.48). Part of what the paper counts as "macro-fit failure" is optimiser failure that a wider search
rescues — one more reason the conservative completeness (Sect. metric) counts gate-rejected lenses as misses. Sentence added to
Sect. 4.2 ¶3. Bound for "all other lenses" is ≤10⁻³ (multipole population max 1.0e-3), not 10⁻⁴.

## 2026-09-12 (late night): round 11 — fifth external report (8/10, "accept after minor revisions")
No new experiments requested. Fixed: Sect. 4.3 detection count 62/75 → 61/75 (basin-corrected, matches Table 5); Sect. 4.1 now
states that the scan's coincidence against S/N_pert is expected by construction (S/N² is the perturbation χ², the scan's statistic
is a χ² difference) and that the non-trivial results are B and the U-Net; the U-Net's non-coincidence is given its two
inseparable readings (c=60-only training set vs 2-pixel-disc target; the c–M-trained network reached chance so they cannot be
separated); Fig. 13 and Table tab:signal captions name the U-Net seed (seed 0 of four, every lens scored); "at the widest bin" →
"at the bin with the largest gap"; gate loop closed (5/24 rescuable ≈ one in five of the excluded lenses per seed set; the reported
and conservative completeness bounds already bracket it); main.tex header comment updated to aa.cls 9.4.
Referee's placement advice: A&A section "Numerical methods and codes" (alternative: Cosmology).

### 2026-09-12 (late night): release repository made public; subtitle dropped
Full-history secret scan clean (token patterns, private keys, e-mails, "tomtom"/"password"/"secret": 0 hits in 496 tracked files);
`gh repo edit --visibility public` on rishabhramteke/lensing_substructure_stress_test. Paper's Data availability now gives the
URL alone. The 13-word subtitle was removed at the user's request (the abstract carries its content); title unchanged.
Margin line numbers are aa.cls 9.4's referee mode (`\linenumbers` set by the class), removed in the typeset article.

## 2026-09-13: round 12 — sixth external report (6.5/10, moderate-to-major, one round)

### Point 1 (gating): the scan statistic had no null floor
Family A's grid holds 72 hypotheses, all containing a subhalo, so Δχ² can be negative and the 10%-FPR
threshold sits below zero. Adding a **zero-mass hypothesis** is exact and needs no refitting: a zero-mass
TNFW *is* the smooth model, so the floored statistic is max(Δχ², 0). `scripts/evaluate_null_floor.py`
→ `results/null_floor.json`, `paper/tables/null_floor.tex`, Table tab:nullfloor, new paragraph in Sect. 4.3.

Full 1000-lens populations, floored vs raw:

| | 72 hypotheses | + zero-mass |
|---|---|---|
| threshold at 10% FPR | −13.4 | **does not exist** (99.6% of clean lenses sit at exactly 0) |
| clean lenses flagged | 10% (by construction) | 0.42% |
| detections with Δχ²<0 | 75/351 | 0 |
| compl. c=60, 4 bins >10⁹ | 36/69/89/81% | 28/64/84/68% |
| compl. c=15, same | 25/59/78/84% | 12/48/74/76% |
| paired flip lost:gained | 63:8 | **71:1** |
| FPR multipole a=0.01 / 0.03 | 31% / 77% | 15% / **69%** |
| localized ≤2 px | 13% | 16% |

Three consequences, written into the paper: (i) the 10%-FPR operating point ceases to exist; (ii) the
completeness reported in the two LOWEST mass bins (17%, 18% at 10⁸–10⁹) was almost entirely threshold
artefact → 1%, 3%; (iii) **every stressed conclusion survives or strengthens** — the multipole ratio goes
from 7.7:1 (77%/10%) to **165:1** (69.5%/0.42%). The common 10%-FPR calibration is kept as the cross-family
axis (it is Tsang+2024's convention and B/C have non-negative statistics), with Family A's sub-10^9.5
numbers to be read from the floored column.

### Point 2: c=15 is not "the ΛCDM value for a subhalo"
Dutton & Maccio is a FIELD-halo relation; subhalos are tidally stripped and denser at fixed M200 by ~2–3×
near the host centre (Moliné+2017, MNRAS 466, 4974 — added to the bib, Crossref-verified). Added
`fixed30` concentration mode + `data/test_fixed30` (n=1000, **seed 101 = matched lens-by-lens to
test_fixed60/test_fixed15**, verified 1000/1000). `scripts/evaluate_intermediate_concentration.py`
→ `results/intermediate_concentration.json`, `paper/tables/c30.tex`, Table tab:c30.

| bin (log10 M200) | A: c60/c30/c15 | U-Net (4 seeds): c60/c30/c15 |
|---|---|---|
| 9.0–9.5 | 38 / 32 / 25% | 15 / 11 / 13% |
| 9.5–10.0 | 71 / 67 / 56% | 29 / 12 / 11% |
| 10.0–10.5 | 85 / 77 / 75% | 51 / 28 / 15% |
| 10.5–11.0 | 85 / 96 / 91% | 63 / 43 / 13% |
| paired flip vs c60 | 9:2 / 21:4 | 130:29 / 199:30 |

**The U-Net has already lost half its total c60→c15 loss by c=30** — the collapse does not depend on the
extreme value. This strengthens the paper against the objection. Language softened everywhere
("ΛCDM-motivated c=15" → "the literature's low-concentration ablation").

### Point 3: confounder FPR conditional on a goodness-of-fit gate
`evaluate_null_floor.py::gate_analysis` → `paper/tables/gate_fpr.tex`, Table tab:gate. Both the confounder
population AND the clean calibration set are restricted to fits passing the gate; threshold recalibrated on
the survivors. At χ²/dof<1.2, 93% of clean lenses pass but only 11% of a=0.03 lenses do — and **77% of those
survivors are still flagged** (47% floored, 17% at Δχ²>20). At <2.0: 45% pass, 80%/67%/49%. For the joint
re-fit, 15 of 100 pass and **100%** are flagged (87% at Δχ²>20). A gate thins the sample; it does not protect
what it keeps.

### Points 4–8 and minors
- Abstract: scoped ("fixed-source variant", "on 100 lenses at 3% amplitude"), results paragraph cut ~a third,
  now ≤300 by the class count.
- Sect. 2: exposure 5400 s, zero point 25.96, sky 22.3 mag/arcsec², read noise 4e⁻, gain 2.5, PSF FWHM 0.08″,
  lenstronomy noise model, **arc S/N median 1.6×10³ (10–90% 0.6–3.0×10³), peak pixel ~250σ**, source AB mag
  19.5–24.7 (median 21.4). Multipole written as eq. (1) with κ_m = a_m cos[m(φ−φ_m)]/(2r), so a_m/θ_E is
  exactly the fractional convergence perturbation at θ_E; cited to Xu+2015 MNRAS 447, 3189 (lenstronomy's
  docstring labels it by the 2013 preprint year — added to bib).
- **Population sharing stated truthfully**: the referee assumed the clean control and multipole populations
  share lenses. They do NOT (seeds 1, 2, 3 — verified 0/1000 macro matches). Only c60/c30/c15 (seed 101) and
  m4_a3/m3_a3 (seed 2) are matched. Text now says so and quotes the ±1.5 pt sampling term at n≈900.
- Exclusion rate reconciled: Sect. 2.3 "2–15%" was wrong → **6–19%** (6% clean, 8% c=15, 19% c=60).
- Table tab:fullpop now carries reported / conservative / Δχ²>0 row blocks (conservative bound was prose only).
- Fig. 6b DID omit Family B — the committed PDF was a **stale render** predating the Family B rows; regenerated,
  and the label wrapping fixed so the six row labels no longer collide with panel (a).
- Localization wording no longer conflates the 21.5% grid ceiling with the 13% achieved.
- Conclusion 1 reworded (the 6–20 pts is for the c=60 template held fixed on both populations).
- max|z| null expectation (~2 for ten bins) stated in the Table 1 caption.
- "near-maximal confidence" defined (score >0.99, sigmoid saturated).
- B1938+666 line-of-sight reading attributed to Şengül+2022 as their reanalysis, not settled.
- Top mass bin flagged as a companion galaxy halo rather than a dark subhalo.
- Family B's single seed set in the lens-light table explained (~40 s/lens vs ~1 s).
- Zenodo DOI promised at acceptance in Data availability.
- Sect. 4.5 (Tier 1) compressed 2155→1620 chars; the c–M training paragraph cut to one sentence with the
  speculation dropped.
- Length: macro-basin control block + Table tab:mass_mechanism moved to new **Appendix A** (`app:controls`);
  duplicated basin prose in Sect. 4.2 cut 1643→383 chars. Two unused figure PDFs deleted; their generators
  marked SUPPLEMENTARY. Three stray blank-line runs collapsed. Tables tab:conc and tab:c30 moved beside their
  discussion (they had drifted 5 pages).
- Build: 19 pages, 0 errors / 0 overfull / 0 undefined / 0 class warnings.

## 2026-09-13: round 13 — sixth external report (6.5/10, major revision)

Two new experiments, both of which changed the paper.

### M6 — survey depth (`data/test_shallow_*`, `scripts/evaluate_depth.py` → `results/depth.json`)
Same lenses, sources and subhalos re-rendered with exposure 5400 s → 135 s (new
`InstrumentConfig.exposure_time`, `tier0_shallow`). Verified 1000/1000 identical truths.
Arc S/N median 1329 → 242, peak pixel 237σ → 41σ. Each family recalibrated on its own arm's clean control.

| | deep | shallow |
|---|---|---|
| A clean FPR at Δχ²>0 | 0.68% | **23%** |
| A macro fits rejected (c=60) | 43/300 | **0/300** |
| A compl. c=60 @ matched 0.68% FPR | 28/69/81/85% | 9/44/68/80% |
| A multipole FPR (Δχ²>0) | 67% | **68%** |
| U-Net clean FPR at its deep threshold | 10% | **72%** |
| U-Net compl. c=60, recalibrated to 10% | 15/29/51/63% | **11/10/16/22%** (chance) |

Three things break (null floor, reliability gate, U-Net transfer), one does not (the multipole
confounder). The result the paper most wants to transfer is the one least sensitive to depth,
while the reliability machinery behind the deep numbers is the part that does not survive.

### M7 — line-of-sight halos (`data/los_halo_z0{25,75}`, `scripts/evaluate_los.py`)
Multi-plane ray tracing via `lens_redshift_list` in ModelAPI (note: `multi_plane` is NOT a
ModelAPI kwarg; supplying the redshift list is what switches it on). New `LOSHaloConfig`,
`tier0_los_halo`, `no_los` ablation control. No subhalo at the lens plane; one field halo
(Dutton–Macciò c, unboosted) at z=0.25 or 0.75, same annulus and mass range.

Foreground z=0.25: flagged 20% (vs 0.68% clean), 65% in the top mass bin; localized ≤2 px only
**5%** (median offset 1.59″ vs 0.30″ for subhalos); inferred mass **−2.0 dex**. Background z=0.75:
4% flagged. U-Net: 11.9±1.5% and 11.0±1.3% — at its baseline, blind to them. Caveat written into
the text: ~1 dex of the mass error is the known frozen-macro-model bias and some is the low field
concentration, so it is not a clean measurement of the redshift error alone.

### M2 — matched denominators (`scripts/evaluate_matched_denominators.py`)
A and B exclude macro-fit failures; the U-Net did not. Intersecting the lenses A and B both retain
(944/826/922 of 1000) and scoring all three there: the U-Net's c=60 top-bin completeness falls
63% → **36%**, and 51% → 45% in the bin below; everything else moves ≤1 point. The family ordering
B > A > C is then **stable under all three accountings in every bin below 10^10.5**; only the top bin
(companion galaxy halos, 0.36% of a CDM population) reverses, so it is kept out of headline statements.

### M3 — null floor made primary for Family A
Defined in Sect 2.3; abstract, conclusions, localization (13% → 16%) and multipole (32–77% → 15–69%
against 0.4% clean) now quote the floored statistic. Bootstrapping the clean control shows the
calibrated threshold carries a realized-FPR range of 9.1–11.1% (full) / 8.5–11.9% (300-lens), moving
completeness by 2 and 5 points — a third argument for the floor, which has no threshold to estimate.

### Other
- Signal test now uses a pooled χ² = Σz² (one dof per bin) instead of max|z|, whose null expectation
  grows with bin count. A coincides ONLY against S/N (χ²/dof 0.9, p=0.57), B only against M_proj(<0.2″)
  (1.4, p=0.22), C against none (p≤0.03). This also explains the |z|=4.0 the referee flagged.
- U-Net saturation checked: threshold 0.958, clean median 0.897, only 1.0% of clean above 0.99 — the
  percentile is not on a saturated tail.
- Bibliography: 10 entries gained volume/pages. **The OVER dict had duplicate keys** — my new entries were
  silently shadowed by older, less complete ones later in the same dict literal. Removed 10 duplicates.
  Only 5 entries remain without volume/pages and all 5 are genuine preprints (verified).
- Tier 1 demoted to Appendix B with a short pointer in the results; Limitations rewritten into four
  paragraphs, now saying plainly that three training runs did not converge so all U-Net claims rest on
  one working configuration.
- Fig 12 promoted to the head of Results (p6, was p15) with two new rows (LOS, depth); two stale cells
  fixed to the floored values. Title cut 116 → 103 chars. τ=20 caveat, COSMOS cuts, λ band, and the
  Table 7/8 clean-FPR difference all stated.
- Build: 23 pages (was 18), 0 errors / overfull / undefined / class warnings. Growth is the three new
  tables the referee asked for plus two new experiments.
