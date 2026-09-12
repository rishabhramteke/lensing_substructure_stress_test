# 05 · Methods survey — dark-matter substructure inference from strong lenses

**Step 0 of the stress test. Version 0 — compiled 2026-09-10** from arXiv abstract pages, GitHub search, and the arXiv listings in `README.md`. **Full-text pass completed the same day: see [`fulltext_findings.md`](fulltext_findings.md) for verified setups, exact limitation quotes and the assumption matrix; it supersedes any FT-marked cell below.** The problem statement derived from it is in [`problem_statement.md`](problem_statement.md).

Purpose: know precisely what each published approach assumes, claims, and released — so the stress test measures what has *never* been measured, and so baseline choice is driven by what can actually be reproduced.

New to the field? Start with [`learning_guide.md`](learning_guide.md) — concepts in plain language, each paper's exact wording, and verified lectures/videos.

---

## 1. Taxonomy — six method families

| Family | What it does | Lineage / key papers | Open code? |
|---|---|---|---|
| **A. Parametric perturber search** | Add one NFW/PJ subhalo to a smooth lens model; scan position/mass; accept by Bayesian evidence threshold. Produces sensitivity maps. | Vegetti & Koopmans 2009 → Despali+ 2022 (sensitivity), Nightingale+ 2024 (54 HST lenses), Minor+ 2024, Lange+ 2024 (JWST), Amvrosiadis+ 2026 (ALMA), Powell+ 2025 (VLBI) | **PyAutoLens** (open, active); Vegetti-group code private |
| **B. Free-form potential corrections** | Reconstruct deviations from smoothness in ψ(θ) on a grid / wavelet / neural-field basis. | Vegetti & Koopmans 2009 (pixelated), Galan+ 2022 (wavelets), Biggio+ 2022 (neural field), Adam 2026 (spectral basis, Fisher) | **herculens** (open); Biggio's neural field **not found** in herculens index |
| **C. ML detection of individual subhalos** | CNN/UNet classifies or segments perturbations directly from the image. | Ostdiek+ 2020/2022 (segmentation), Tsang+ 2024 (UNet, COSMOS sources), Hughes+ 2024 (ResNet50), Campbell+ 2026 (polar CNN) | **None public** for any of the four |
| **D. Population-level simulation-based inference** | Neural ratio/posterior estimators infer SHMF normalization, WDM mass, density slopes, or DM model class from one or many lenses. | Brehmer+ 2019 (NRE), Coogan+ 2020 & Anau Montel+ 2022 (TMNRE), Wagner-Carena+ 2023/2024 (NPE/SNPE), Zhang+ 2022/2023 (NLRE slope, real HST), Dhanasingham+ 2025 (NPE, LOS+sub), Filipp+ 2026 (NRE, LSST), Alexander+ 2020 / DeepLense (DM-model classification) | **paltas/paltax**, **swyft** (engine), **mining-for-substructure-lens**, **neural-subhalo-slope**, **DeepLense** open; Anau Montel lensing sims, Dhanasingham, Filipp **not found** |
| **E. Statistical / field-level** | Infer the power spectrum of potential perturbations (GRF) rather than individual objects. | Vernardos+ 2020 (uncertainty-aware CNN), Fagin+ 2024 (23 SLACS), Adam 2026 (information geometry) | Fagin+ 2024 **public** (GRF_ML); Vernardos 2020 data on request; Adam 2026 "upon publication" |
| **F. Forecast / sensitivity infrastructure** | Fisher/Asimov or injection-recovery to predict detectability for a survey or instrument. | Despali+ 2022, O'Riordan 2025 (hidden population), Wedig+ 2025 (Roman), Filipp+ 2026 (LSST), **hwo-slaps** (JPL, HWO) | **hwo-slaps** (open, Apache-2.0, PyAutoLens + HCIPy PSFs); others n/s |

---

## 2. Methods matrix

Columns: **Target** · **Source model** · **Lens complexity** · **LOS halos** · **Instrument** · **Train size** · **Claimed result** · **Stated limitation** · **Code**.

### Family A — parametric perturber search

| Paper | Target | Source | Lens complexity | LOS | Instrument | Train | Claimed | Limitation | Code |
|---|---|---|---|---|---|---|---|---|---|
| Nightingale+ 2024 [2209.10566](https://arxiv.org/abs/2209.10566) | Individual subhalos in 54 HST lenses | Pixelized (PyAutoLens) | Power law vs stars+DM decomposition; "varied azimuthal freedom" | No (subhalo at lens z) | HST | — (real data) | 5 candidates, 2 compelling, 45 non-detections | "Detectability depends upon the assumed parametric form for the lens galaxy's mass distribution" | [autolens_subhalo](https://github.com/Jammy2211/autolens_subhalo) + [PyAutoLens](https://github.com/PyAutoLabs/PyAutoLens) |
| Despali+ 2022 [2111.08718](https://arxiv.org/abs/2111.08718) | Lowest detectable mass (sensitivity maps) | Mock | NFW perturber on smooth lens | FT | Varied S/N & resolution | — | Min detectable mass 1.5×10⁸–3×10⁹ M☉ depending on S/N, resolution, geometry | Sensitivity is configuration-dependent | n/s |
| Lange+ 2024 [2410.12987](https://arxiv.org/abs/2410.12987) | Subhalo vs angular complexity in 2 JWST lenses | Pixelized | Multipoles m=1,3,4 | No | JWST NIRCam multi-band | — | Bayes factor for substructure 60 → 11 once multipoles included; still "5σ" in SPT2147-50 | "further analysis is needed to confirm that the signal is not due to systematics associated with the lens mass model" | n/s (PyAutoLens-based) |
| O'Riordan 2025 [2509.02660](https://arxiv.org/abs/2509.02660) | Effect of undetectable population on detections | FT | "varying amounts of angular structure" | FT | HST-like | — | Hidden population causes **+40% excess** in detected subhalo counts | Multipoles "degenerate with the population signal" | n/s |
| Minor+ 2024 [2408.11090](https://arxiv.org/abs/2408.11090) | J0946+1006 substructure significance | — | — | — | HST | — | ~17σ via pixel supersampling; (2.2–3.4)×10⁹ M☉ within 1 kpc | — | n/s |

### Family B — free-form potential corrections

| Paper | Target | Source | Lens complexity | LOS | Instrument | Train | Claimed | Limitation | Code |
|---|---|---|---|---|---|---|---|---|---|
| Galan+ 2022 [2207.05763](https://arxiv.org/abs/2207.05763) | Non-smooth potential structure | FT | Tested: localized subhalo; GRF (LOS-like); multipoles | Via GRF | HST-like | — (per-lens fit) | "wavelets are able to recover all of these structures accurately" | n/s | [herculens](https://github.com/Herculens/herculens) |
| Biggio+ 2022 [2210.09169](https://arxiv.org/abs/2210.09169) | Full potential or perturbations only | **Sérsic source fixed to the truth** — "we keep the source fixed to the truth and do not attempt to reconstruct it simultaneously" (FT-verified) | SIE; perturbations: single subhalo, GRF population, external shear; no lens light | Via GRF | HST-like, PSF known | none (per-lens fit) | Perturbation parameters recovered well *given perfect knowledge of the main potential*; artifacts when the full potential is fitted; GRF power spectrum recovered statistically | Fourier-feature σ set "by trial and error"; source fixed; PSF known | implemented in **herculens** ([austinpeel/herculens](https://github.com/austinpeel/herculens)); no separate release |
| Adam 2026 [2608.18224](https://arxiv.org/abs/2608.18224) | Information geometry of unresolved population | Flexible source at varying expressivity | Macro-model complexity analysis | n/s | n/s | — | Macro-model degeneracies confined to low-order perturbations; **source-model degeneracies "strongly suppress sensitivity across a broad range of scales"** | n/s | n/s ([caustics](https://github.com/Ciela-Institute/caustics) is the group's simulator) |

### Family C — ML detection of individual subhalos

| Paper | Target | Source | Lens complexity | LOS | Instrument | Train | Claimed | Limitation | Code |
|---|---|---|---|---|---|---|---|---|---|
| Ostdiek+ 2020/22 [2009.06663](https://arxiv.org/abs/2009.06663) | Segmentation; subhalo detection/mass | Sources at mag 17–25 (FT model) | Ellipticity, quadrupole, octopole | No | FT | 1 lens, 0–1 subhalo per image | Lens area recovered to ~1%; subhalos to 10⁸·⁵ M☉ "in bright pixels" | Trained on 1 lens + ≤1 subhalo | **none** |
| Tsang+ 2024 [2401.16624](https://arxiv.org/abs/2401.16624) | Segmentation of subhalos | **COSMOS real galaxies** | Power law + multipoles + shear | n/s | HST-like, realistic noise | n/s | **TPR 71% @ FPR 10%** for 10⁹–10⁹·⁵ M☉ | **"fails at detecting subhalos with lower concentrations (expected from ΛCDM simulations)"** | **none** |
| Hughes+ 2024 [2403.04349](https://arxiv.org/abs/2403.04349) | Binary perturbed/unperturbed | Sérsic clumps (1–5) | Single subhalo 10⁷·⁵–10¹¹ M☉ | No | Natural seeing (ground) | **800,000** images | 74% accuracy at highest S/N; source complexity matters little beyond 3 clumps | Performance "heavily dominated by high mass subhalos" | **none** |
| Campbell+ 2026 [2607.02663](https://arxiv.org/abs/2607.02663) | Subhalo mass regression + uncertainty | n/s | n/s | n/s | Simulated HST, varying noise; c=30 and c=60 | n/s | Polar input → **+15% detection** for 10⁹–10⁹·⁵ M☉ | Gains mostly in low-S/N, low-c regime | **none** |

### Family D — population-level SBI

| Paper | Target | Source | Lens complexity | LOS | Instrument | Train | Claimed | Limitation | Code |
|---|---|---|---|---|---|---|---|---|---|
| Brehmer+ 2019 [1909.02005](https://arxiv.org/abs/1909.02005) | SHMF population parameters (NRE) | FT | FT | FT | FT | FT | Proof of principle | "proof-of-principle application to simulated data" | [mining-for-substructure-lens](https://github.com/smsharma/mining-for-substructure-lens) |
| Coogan+ 2020 [2010.07032](https://arxiv.org/abs/2010.07032) | Substructure distribution (NRE + variational source) | GP source | FT | FT | FT | FT | "preliminary results" | — | engine: [swyft](https://github.com/undark-lab/swyft) |
| Anau Montel+ 2022 [2205.09126](https://arxiv.org/abs/2205.09126) | WDM half-mode mass (TMNRE) | FT | Parametric | **Yes — perturbers below detection threshold marginalized** | HST resolution, multiple lenses | FT | "empirically testable inference of the dark matter cutoff mass" | Proof of concept on sims | engine: swyft; lensing sims **not found** |
| Wagner-Carena+ 2023 [2203.00690](https://arxiv.org/abs/2203.00690) | SHMF from hundreds of lenses (NPE + hierarchical) | **COSMOS** | Realistic low-mass halos <10¹⁰ M☉ | **Yes** | HST | large (FT) | "Reliably infer the SHMF … scale to hundreds of lenses" | Simulated only | [paltas](https://github.com/swagnercarena/paltas) |
| Wagner-Carena+ 2024 [2404.14487](https://arxiv.org/abs/2404.14487) | Same, sequential NPE | COSMOS (FT) | as above | Yes | HST | 5× fewer sims than NPE | Same constraints with 1/5 the mocks; else >3 orders of magnitude more GPU-hours | **"constraints are limited primarily by methodology and not the data itself"** | [paltax](https://github.com/swagnercarena/paltax) |
| Zhang+ 2022/23 (NLRE slope) | Subhalo effective density slope, incl. on real HST | FT | FT | FT | HST (real) | FT | Slope constraints on real lenses | FT | [neural-subhalo-slope](https://github.com/gemyxzhang/neural-subhalo-slope), [-data](https://github.com/gemyxzhang/neural-subhalo-slope-data) |
| Dhanasingham+ 2025 [2511.17732](https://arxiv.org/abs/2511.17732) | LOS + subhalo mass functions, m–c, multipoles (NPE) | n/s | Multipoles (power-law parameterization) | **Yes** | n/s | n/s | — | **"remains challenging"; LOS amplitude ↔ subhalo normalization degeneracy; multipole params poorly recovered; training data "may inadequately represent the true underlying structure"** | **none** |
| Filipp+ 2026 [2604.07438](https://arxiv.org/abs/2604.07438) | HMF parameters (NRE), LSST forecast | FT | Subhalos + LOS to ~10⁷ M☉ | **Yes** | LSST 10-yr | up to 2,500 lenses | 2,500 lenses exclude 74%/36% of prior at 3σ/5σ | **"assumes perfect knowledge of the data-generating process; cannot be directly applied to data"** | **none** |
| Alexander+ 2020 [1909.07346](https://arxiv.org/abs/1909.07346) | Classify DM model (CDM / none / vortex) | n/s | n/s | n/s | n/s | n/s | "reliably distinguish" | n/s | successor: [ML4SCI/DeepLense](https://github.com/ML4SCI/DeepLense) (64×64 toy sims) |

### Family E — statistical / field-level

| Paper | Target | Source | Lens complexity | LOS | Instrument | Train | Claimed | Limitation | Code |
|---|---|---|---|---|---|---|---|---|---|
| Vernardos+ 2020 [2010.07314](https://arxiv.org/abs/2010.07314) | GRF power-spectrum of potential perturbations, uncertainty-aware CNN | **Real galaxy sources** | GRF perturbations | via GRF | HST-like (FT) | FT | Confidence intervals reduced 10% "in an unsupervised manner" | FT | **none** |
| Fagin+ 2024 [2403.13881](https://arxiv.org/abs/2403.13881) | GRF power-law statistics on 23 SLACS lenses | Shapelets + Sérsic from Shajib+ 2021 fits; augmented | SIE + shear | via GRF | HST/ACS 0.05″, Tiny Tim PSF, correlated noise | 250k (110×110 px) | Population: log σ²=−2.94, β=4.67; "significant substructure perturbation favoring a high frequency power spectrum"; coverage calibrated | GRF breaks if few massive subhalos; single-Sérsic lens light; sim→real uncertainty mismatch | **public:** [JFagin/GRF_ML](https://github.com/JFagin/GRF_ML) (sims on request) |

### Family F — forecast / sensitivity infrastructure

| Item | What | Instrument | Method | Code |
|---|---|---|---|---|
| **hwo-slaps** (NASA JPL, Sep 2025) | End-to-end sim + Fisher/Asimov subhalo detectability vs PSF stability | HWO (segmented aperture, HCIPy PSFs) | PyAutoLens-based; not ML; "study layer" (sweeps, aggregation) still TODO | [nasa-jpl/hwo-slaps](https://github.com/nasa-jpl/hwo-slaps), Apache-2.0, 102 commits, sparse docs |
| Wedig+ 2025 [2506.03390](https://arxiv.org/abs/2506.03390) | Roman lens yield; ~500 substructure-grade lenses; sims "designed to support neural network training" | Roman WFI | Yield simulation | n/s |
| Kollmann+ 2025 [2510.17956](https://arxiv.org/abs/2510.17956) | Inner slope β=2.2 subhalos detectable 10× lower in mass than NFW | HST / Euclid / JWST mocks | Parametric | n/s |
| **slsim** (LSST DESC) | Strong-lens simulation pipeline for Rubin | LSST | Simulation | [LSST-strong-lensing/slsim](https://github.com/LSST-strong-lensing/slsim) |

### Supporting tools (open)

| Tool | Use in the stress test |
|---|---|
| [paltas](https://github.com/swagnercarena/paltas) / [paltax](https://github.com/swagnercarena/paltax) | Simulation with subhalo + LOS populations, COSMOS sources, HST realism; NPE training sets. **Primary Tier 0–2 engine.** |
| [lenstronomy](https://github.com/lenstronomy/lenstronomy) | Underlying lens modelling; multipoles, shear, arbitrary profiles. |
| [caustics](https://github.com/Ciela-Institute/caustics) | Differentiable PyTorch simulator (fast batched sims; Adam 2026's basis). |
| [herculens](https://github.com/Herculens/herculens) | Family B baseline (wavelets). |
| [PyAutoLens](https://github.com/PyAutoLabs/PyAutoLens) | Family A baseline; also used by hwo-slaps. |
| [swyft](https://github.com/undark-lab/swyft), [sbi](https://github.com/sbi-dev/sbi) | Family D engines (TMNRE, NPE). |
| [tarp](https://github.com/Ciela-Institute/tarp) | **Coverage tests for posterior calibration — the calibration metric.** |
| [slsim](https://github.com/LSST-strong-lensing/slsim), [deeplenstronomy](https://github.com/deepskies/deeplenstronomy) | Rubin/DES-style survey realism (Tier 3). |
| [COOLEST](https://github.com/aymgal/COOLEST) | Standard format for lens-model outputs — adopt for reporting substructure claims uniformly. |
| [DeepLense](https://github.com/ML4SCI/DeepLense) | Toy-tier reference (what many ML papers effectively train on). |

---

## 2b. Exact wording — the sentences the stress test is built on

Verbatim from the arXiv abstracts (automated extraction, 2026-09-10; re-verify against PDFs before citing). Grouped by what they establish.

**Methods fail under realism**
- Tsang+ 2024: "Our algorithm fails at detecting subhalos with lower concentrations (expected from ΛCDM simulations)."
- Hughes+ 2024: performance "heavily dominated by high mass subhalos."
- Dhanasingham+ 2025: "Recovering the dark matter substructure mass functions and mass-concentration parameters remains challenging"; training data "may inadequately represent the true underlying structure."
- Wagner-Carena+ 2024: "Current constraints are limited primarily by methodology and not the data itself."

**Confounders mimic substructure**
- Lange+ 2024: Bayes factor for substructure ~60 without multipoles → ~11 with; "further analysis is needed to confirm that the signal is not due to systematics associated with the lens mass model."
- O'Riordan 2025: "This population causes an excess of 40 per cent in the number of detected subhaloes for HST-like strong lens observations"; multipoles are "degenerate with the population signal."
- Adam 2026: "Degeneracies with the source model can strongly suppress sensitivity across a broad range of scales."
- Şengül+ 2022: "first dark perturber shown to be a line-of-sight halo with a gravitational lensing method."
- Nightingale+ 2024: "Detectability of subhalos depends upon the assumed parametric form for the lens galaxy's mass distribution."

**Validation is sim-to-sim**
- Filipp+ 2026: "assumes perfect knowledge of the data-generating process"; "cannot be directly applied to data analysis."
- Brehmer+ 2019: "proof-of-principle application to simulated data."
- Anau Montel+ 2022: proof of concept on simulated data (abstract wording: "empirically testable inference of the dark matter cutoff mass").

**Claimed capabilities (what the stress test re-measures)**
- Tsang+ 2024: "true positive rate of 71%" at 10% false-positive rate for 10⁹–10⁹·⁵ M☉.
- Hughes+ 2024: "74% accuracy in highest peak signal-to-noise datasets."
- Campbell+ 2026: "For subhalos with mass 10⁹M☉ ≤ M ≤ 10⁹·⁵M☉, detection fraction increases by ~15 per cent."
- Ostdiek+ 2020/22: lens area recovered with "1.3% of true area missed and 1.2% added elsewhere."
- Galan+ 2022: "wavelets are able to recover all of these structures accurately."
- Biggio+ 2022: "explicitly retains lensing physics (i.e., the lens equation)."
- Wagner-Carena+ 2023: "Reliably infer the SHMF across a variety of configurations and scale efficiently to populations with hundreds of lenses."
- Despali+ 2022: minimum detectable mass "lies between 1.5×10⁸ and 3×10⁹ M☉."
- Fagin+ 2024: "significant substructure perturbation favoring a high frequency power spectrum."
- Vernardos+ 2020: confidence intervals reduced by 10% "in an unsupervised manner."
- Kollmann+ 2025: steep-slope subhalos detectable "an order-of-magnitude lower in mass than NFW."

---

## 3. Cross-cutting assumptions (what the field takes for granted)

1. **Source realism is the exception, not the rule.** Real (COSMOS) sources: Tsang 2024, Wagner-Carena 2023/24, Vernardos 2020. Sérsic/clumps: Hughes 2024, Ostdiek, DeepLense. Biggio 2022 *assumes the source is known.* Adam 2026 shows source degeneracy is the dominant sensitivity killer — so this choice decides results.
2. **Angular complexity of the lens is inconsistently modelled.** Multipoles included: Tsang, Ostdiek (m=2,3/4), Nightingale (azimuthal freedom), Lange (m=1,3,4), Dhanasingham, Galan (test only). Absent: Hughes (single subhalo), Alexander/DeepLense, most forecasts. Lange and O'Riordan both show multipoles are *degenerate with substructure.*
3. **Line-of-sight halos are treated only by the SBI family.** Wagner-Carena, Anau Montel, Filipp, Dhanasingham include LOS; the ML detectors (C) and most of A do not — yet Şengül 2022 showed the "first dark perturber" was a LOS halo.
4. **Subhalo concentration is unrealistic in training sets.** Detectors are trained/tested at high c (Campbell tests c=30/60); Tsang and Hughes both report failure at ΛCDM-like concentrations; Kollmann shows the inner slope alone moves detectability by 10×. Nobody evaluates on a ΛCDM c–M relation *with scatter.*
5. **HST is the default instrument.** Euclid VIS realism is essentially absent from the detector/SBI literature despite DR1; Roman, LSST, HWO exist only as forecasts.
6. **Metrics are incomparable.** TPR@FPR (Tsang), accuracy (Hughes), Bayes factors (Nightingale, Lange), exclusion volume (Filipp), sensitivity maps (Despali), confidence-interval width (Vernardos). Posterior coverage is almost never reported although `tarp` exists.
7. **Sim-to-sim evaluation is universal.** Every SBI paper trains and tests on the same simulator; Filipp states the perfect-knowledge assumption explicitly; Dhanasingham admits training data may not represent reality. Real-data contact: Nightingale (54 HST), Fagin (23 SLACS), Zhang (HST), Lange (2 JWST). The ML detectors (C) have never been run on real lenses in a controlled way.
8. **Code is the exception.** Of ~25 method papers surveyed, ~11 have public implementations. **Family C (ML detectors) has zero** — Ostdiek 2020/22, Tsang 2024, Hughes 2024, Campbell 2026 all release nothing or "on request". Family E: Fagin 2024 public, Vernardos 2020 on request.

---

## 4. What is missing — preliminary gap list

To be sharpened during reproduction; each gap is stated so it can be *measured*, not argued.

| # | Gap | Evidence it's real | Why nobody closed it |
|---|---|---|---|
| **G1** | **No side-by-side evaluation of families A–E on the same data.** | Every row in §2 uses its own sims and metric; numbers are not comparable. | Each group owns one method; evaluation across groups has no owner. |
| **G2** | **Confounder-induced false positives are never measured across methods.** Multipoles, bars, LOS halos, unresolved populations, source complexity — each shown to mimic substructure by *one* paper for *one* method. | Lange (multipoles, BF 60→11), Shan (bars), Şengül (LOS), O'Riordan (+40%), Adam (source). | Requires a common simulator with confounders switchable one at a time — exists as parts (`paltas`, `lenstronomy` multipoles), never assembled. |
| **G3** | **Posterior calibration under misspecification is untested.** Train on simulator/tier X, test on Y; report coverage. | Filipp's perfect-knowledge caveat; Dhanasingham's admission; Wagner-Carena's "methodology-limited". `tarp` is available and unused across methods. | Sim-to-sim is cheaper and looks better. |
| **G4** | **Concentration realism.** No published evaluation grid in (mass, concentration) with a ΛCDM c–M relation + scatter. | Tsang/Hughes failures; Campbell's c=30/60; Kollmann's 10×. | Training sets were built for detectability, not realism. |
| **G5** | **Instrument transfer, esp. Euclid VIS.** | HST-only detectors; DR1 imminent. | Euclid-realistic sims exist (Kollmann's mocks, Euclid SL SWG) but not in ML papers. |
| **G6** | **Reproducibility of family C and E is zero.** A faithful reimplementation from the text, with reported numbers reproduced or not, is itself a result. | §2 code column. | Nobody is rewarded for reproducing. |
| **G7** | **No uniform reporting of substructure claims.** | Metric heterogeneity (§3.6); COOLEST covers lens models, not substructure results. | Standards emerge from evaluations, which don't exist (G1). |
| **G8** | **Controlled sim-to-real for ML detectors.** Run C-family detectors on the same real lenses A-family analysed (Nightingale's 54; SLACS) and compare claims. | Detectors never touched real data in a controlled comparison. | Detectors weren't built to be trusted on real data yet. |

**Working hypothesis for paper 1:** G1 + G2 + G3 together — *one suite, all families, confounders on/off, coverage reported* — is the paper. G4 and G5 are dimensions of that suite. G6 is a reproducibility section. G7/G8 are follow-ups or paper 2 material.

**What the stress test is not:** not a new detector, not a new SBI method, not a benchmark/challenge for others to enter. It is a measurement.

---

## 5. Baseline selection (decision, v0)

| Family | Baseline | Source | Effort | Notes |
|---|---|---|---|---|
| A | PyAutoLens subhalo scan | open + Nightingale scripts | Medium; nested sampling per lens is slow → run on a *subset* (tens of lenses per tier) | Also gives the "classical" reference the community trusts |
| B | herculens wavelet corrections | open | Medium | Per-lens fit; compare recovered perturbation power |
| C | **Reimplement** a UNet segmenter (Ostdiek/Tsang recipe) | **no code** | ~1 week + training on CPU/small GPU | Reproduction attempt is a finding either way |
| D | paltas + NPE (Wagner-Carena recipe) | open | Low–medium | Primary SBI baseline; hierarchical population inference. Optional: swyft TMNRE with the same simulator |
| E | *skip in v0* (GRF CNN; no code) | — | — | Add only if time; note the reproducibility gap |
| Metrics | `tarp` coverage; ROC/PR; confounder FPR; runtime | open | Low | Standardize outputs (COOLEST-style JSON) |

Tools to install first: `paltas`, `lenstronomy`, `galsim` (COSMOS), `sbi`, `tarp`, `herculens` (JAX), `PyAutoLens`. Check `hwo-slaps` for reusable PSF/noise modules and its Fisher detectability code as a cross-check on Tier 0.

---

## 6. Full-text verification checklist

Read the PDF and fill the FT cells for: Tsang 2024 (training size, image size, LOS?), Hughes 2024 (PSF, pixel scale), Ostdiek 2020/22 (both papers; image size, noise), Wagner-Carena 2023 (training size, LOS model, mass ranges), Anau Montel 2022 (simulator details, code link in text?), Dhanasingham 2025 (entire simulation setup), Filipp 2026 (simulator, metric), Biggio 2022 (is the source really assumed known? code link?), Galan 2022 (limitations), Vernardos 2020, Fagin 2024, Brehmer 2019 (simulator), Adam 2026 (instrument, code), Despali 2022, O'Riordan 2025 (method, code), Zhang 2022/23 (add rows properly). Also skim: Ostdiek+ 2020 *Extracting the subhalo mass function…* (ApJ) — companion to the A&A Letter.

---

## 7. Sources consulted

arXiv (abstract pages / API): 2209.10566 · 2111.08718 · 2410.12987 · 2509.02660 · 2408.11090 · 2207.05763 · 2210.09169 · 2608.18224 · 2009.06663 · 2401.16624 · 2403.04349 · 2607.02663 · 1909.02005 · 2010.07032 · 2205.09126 · 2203.00690 · 2404.14487 · 2511.17732 · 2604.07438 · 1909.07346 · 2010.07314 · 2403.13881 · 2112.00749 · 2506.03390 · 2510.17956 · 2508.14624.

GitHub: swagnercarena/{paltas,paltax} · PyAutoLabs/PyAutoLens · Jammy2211/autolens_subhalo · Herculens/herculens · undark-lab/swyft · Ciela-Institute/{caustics,tarp} · smsharma/{mining-for-substructure-lens,lensing-neural-fields} · acagansengul/interlopers_with_lenstronomy · gemyxzhang/neural-subhalo-slope · ML4SCI/DeepLense · nasa-jpl/hwo-slaps · LSST-strong-lensing/slsim · deepskies/deeplenstronomy · aymgal/COOLEST · giga-lens/gigalens. Code search `neural` in Herculens/herculens: 0 files.
