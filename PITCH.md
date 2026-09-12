# 05 · Dark-matter substructure from strong lensing — a systematic stress test of published methods

**Status:** **PRIMARY — solo, Sep 2026 → ~Jan 2027** · Step 0: methods survey → tiered simulation suite → reproduce open methods → stress test → gap analysis → paper 1; paper 2 chosen from what the stress test exposes · **No challenge, no community organizing.** · **Compute:** CPU (simulation + running others' open baselines); small GPU rental for one or two baselines

## One-paragraph pitch

Dark matter is not observed directly; it is *mapped* through gravity, and the sharpest map is strong gravitational lensing — a massive galaxy bends light from a background source into arcs and rings whose shape encodes the lens's mass distribution, including small dark-matter substructure that no other probe reaches. This is an **image problem**: find lenses in survey images, then infer mass (and dark-matter model parameters) from the arcs. Euclid's Q1 release (March 2025) delivered hundreds of new strong-lens candidates and the full survey will find ~10⁵; simulators (`lenstronomy`) provide unlimited labeled training data. Dozens of methods now claim to detect or constrain dark-matter substructure in lens images — CNN detectors, neural posterior/ratio estimators, gravitational imaging, wavelet potentials — **and each is validated on its own simulations.** Meanwhile the literature keeps finding things that mimic substructure (angular mass complexity, bars, unresolved low-mass populations) and detectors that collapse at realistic subhalo concentrations. Euclid DR1 (late 2026) brings ~15,000 lenses and a predicted ~2,500 subhalo detections. **Nobody has tested these methods side by side under the same realistic conditions.** This project does that: a proper study of what the published approaches actually assume and claim, a tiered-realism simulation suite, the open methods reproduced and stress-tested on it, and a gap analysis that decides the next paper. Solo work; the suite is released as a by-product, not run as a challenge. It also carries the cleanest theory hook in the portfolio: the **mass-sheet degeneracy** — a proven statement that lensing images alone cannot determine the mass normalization — the rigorous "uncertainty principle" analog for this domain.

## Why this fits Rishabh

- It is computer vision on images of the sky, with ground truth from simulators — the same shape as an industry CV project.
- Flows + Bayesian inference + a real theorem (mass-sheet degeneracy) satisfy the theory appetite.
- CPU-feasible: lens images are 64–128 px; posteriors are low-dimensional; `lenstronomy` renders thousands per hour.
- Fresh data (Euclid Q1; Rubin DP1) means results are timely.
- The CV angle (below) is real, if not as dominant as in 01. Combined with motivation and speed-to-result, that made this the first project.

## The computer-vision angle

Lens modelling is inverse rendering. A lens maps source-plane coordinates β to image-plane coordinates θ through a **deflection field**: β = θ − α(θ). Recovering the mass map means recovering a smooth 2D warp from the distortion of a background galaxy — the same problem family as image registration and optical flow, solved by differentiable ray-shooting instead of differentiable rasterization. Free-form (pixelated or neural-field) parameterizations of the lens potential ψ(θ), with α = ∇ψ and convergence κ = ½∇²ψ, are exactly where geometry-aware CV experience pays off: regularization of deformation fields, coarse-to-fine optimization, degeneracy handling. Recent work already uses continuous neural fields for lens potentials (Biggio et al. 2023) and score-based priors for pixelated reconstructions (Adam et al. 2022) — the seams to work at are calibrated *uncertainty* on the recovered κ map and robustness to source complexity.

## Visualizing dark matter — what you will actually be able to render

Dark matter is only seen through gravity, so every "image" of it is a reconstruction. Concretely, within the first two weeks:

| What | Source | Effort | Where it goes |
|---|---|---|---|
| **Dark-matter map of the sky** — weak-lensing convergence (κ) maps over thousands of deg² | DES Y3 mass maps (Jeffrey et al. 2021), KiDS-1000, HSC Y3 — public FITS/HEALPix | An evening: `healpy` + matplotlib | Website hero image; paper intro figure |
| **The cosmic web in 3D** — dark-matter density from simulation | IllustrisTNG / CAMELS snapshots (public) | A weekend: subsample particles → point cloud → three.js | Interactive 3D on the website |
| **Real lensed arcs** | Euclid Q1 strong-lens candidate cutouts; SLACS HST lenses | An evening | Gallery; paper figure |
| **Your own reconstructions** — κ map of each lens (total projected mass, mostly dark), posterior uncertainty map, subhalo perturbation/residual maps | This project's model | The project | Paper's main figures; interactive lens demo (drag a source, see arcs; slide a mass sheet, watch the degeneracy) |

Honest framing: κ maps from parametric lens models are smooth ellipsoidal blobs — scientifically central, visually modest. The striking visuals are the sky-scale weak-lensing maps, the cosmic web, and substructure residuals. Use them on the website; keep the paper's claims on the inference.

## Prior-art check — verified 2026-09-10 (arXiv API, 2023–2026 window)

| Theme | What exists | Consequence |
|---|---|---|
| Neural-field lens potential | Biggio et al. — [2210.09169](https://arxiv.org/abs/2210.09169) (A&A 2023): continuous neural field for the potential incl. subhalo perturbations. Mishra-Sharma & Yang — [2206.14820](https://arxiv.org/abs/2206.14820): neural-field *source*. No follow-ups found 2023–26. | Not saturated, but not new either. |
| Learned-prior posteriors over mass maps | **Mainstream in weak lensing:** Boruah et al. [2502.04158](https://arxiv.org/abs/2502.04158) and [2511.14667](https://arxiv.org/abs/2511.14667) (diffusion posterior sampling on DES-Y3, uncertainty validated); Whitney et al. [2410.24197](https://arxiv.org/abs/2410.24197); Remy et al. [2606.31988](https://arxiv.org/abs/2606.31988) (joint κ + cosmology, calibration scored); Aoyama [2505.00345](https://arxiv.org/abs/2505.00345); Flöss [2405.05484](https://arxiv.org/abs/2405.05484) (CMB). **Cluster-scale strong+weak lensing:** Royo et al. [2603.14503](https://arxiv.org/abs/2603.14503) — diffusion prior on mass, posterior samples, validated on MACS 1206. | The original "contribution 1" (calibrated neural-field κ maps with a learned prior, galaxy scale) would be a **port of an established technique**. Publishable, ~10–30 citations, incremental. **Demoted to a baseline.** |
| Fisher / information geometry for substructure | Adam et al. [2608.18224](https://arxiv.org/abs/2608.18224): differentiable simulator + Fisher analysis of degeneracies between subhalos and lens/source components. | The CRLB/Fisher theory angle is taken. Cite it; don't redo it. |
| Population inference with SBI | Filipp [2604.07438](https://arxiv.org/abs/2604.07438) (NRE, LSST forecast); Dhanasingham [2511.17732](https://arxiv.org/abs/2511.17732) (NPE; **reports degeneracies and limited accuracy**); Wagner-Carena [2404.14487](https://arxiv.org/abs/2404.14487) (sequential NPE; **"methodological constraints dominate data limits"**); Lonergan [2504.15468](https://arxiv.org/abs/2504.15468) (flows emulating subhalo populations). | Active, crowded, and openly struggling with validation. |
| Individual detection with ML | Tsang [2401.16624](https://arxiv.org/abs/2401.16624), Hughes [2403.04349](https://arxiv.org/abs/2403.04349): UNet/ResNet reach 71% TPR at 10% FPR for 10⁹–10⁹·⁵ M☉ **but fail at realistic ΛCDM concentrations**; Campbell [2607.02663](https://arxiv.org/abs/2607.02663) (polar coords, +15%); Fagin [2403.13881](https://arxiv.org/abs/2403.13881) (power spectrum on 23 SLACS). | Toy-sim performance does not transfer. |
| Confounders that mimic substructure | Lange [2410.12987](https://arxiv.org/abs/2410.12987) (JWST: multipoles vs subhalo); Shan [2510.02805](https://arxiv.org/abs/2510.02805) (bars → flux-ratio anomalies); O'Riordan [2509.02660](https://arxiv.org/abs/2509.02660) (hidden low-mass population inflates detections by ~40%); Kollmann [2510.17956](https://arxiv.org/abs/2510.17956) (inner slope shifts detectability 10×). | A systematic, common-yardstick treatment of these does not exist. |
| Real detections | Powell [2510.07382](https://arxiv.org/abs/2510.07382) (10⁶ M☉ via VLBI gravitational imaging); Minor [2408.11090](https://arxiv.org/abs/2408.11090); Amvrosiadis [2605.21212](https://arxiv.org/abs/2605.21212); Gilman [2606.05277](https://arxiv.org/abs/2606.05277) (28 JWST lenses). | The science is real and high-profile; the validation is not shared. |
| Building blocks | Wedig [2506.03390](https://arxiv.org/abs/2506.03390) (Roman sims built for NN training); SIDM Concerto [2503.10748](https://arxiv.org/abs/2503.10748) (SIDM subhalo populations, public); DREAMS [2409.02980](https://arxiv.org/abs/2409.02980) (subhalo emulator); differentiable simulators: `caustics`, `herculens`, `paltas`, `lenstronomy`. | Everything needed to build a benchmark exists as parts. |
| Benchmarks / challenges | **None for modelling or substructure inference.** Only lens *finding*: Metcalf et al. 2019; Euclid Q1 AgileLens [2604.06648](https://arxiv.org/abs/2604.06648), AstroVink [2604.21977](https://arxiv.org/abs/2604.21977). DeepLense (ML4Sci GSoC) offers 64×64 toy sims, no hidden test set or leaderboard. Precedent for a *modelling* challenge: TDLMC (Ding et al. 2021, H₀). | **This is the gap.** |
| What's coming | Euclid DR1 public late 2026: ~15,000 lenses, ~2,500 predicted subhalo detections ([Nature Astronomy 2025](https://arxiv.org/abs/2508.14624)); Roman ~500 substructure-grade lenses. | Demand for a yardstick peaks in 2027. |

**Verdict.** The field has many methods and no common test. Every group validates on its own simulations; the confounders are documented but never stress-tested side by side; and the largest lens sample in history arrives within months. For a solo author the highest-leverage paper here is not another method — it is **the systematic evaluation that every later method paper has to cite and compare against.** Evaluation papers with a released suite get cited by construction (Lueckmann et al. 2021 for SBI; Zeghal et al. [2409.17975](https://arxiv.org/abs/2409.17975) for weak lensing; Metcalf 2019 for lens finding). No organizing required — the suite just has to be good and public.

## The project: a systematic stress test (solo)

### Step 0 — Methods survey: the proper study (weeks 1–3)
For every published substructure-inference approach since ~2017, record: inference target (individual subhalos / population parameters / free-form map / statistics), the simulation realism it was trained and validated on (source model, lens complexity, line-of-sight halos, instrument/PSF/noise, image size), training-set size, claimed performance, stated limitations, and **code availability**. Output: [`methods_survey.md`](methods_survey.md) → full-text pass [`fulltext_findings.md`](fulltext_findings.md) → [`problem_statement.md`](problem_statement.md) (all done 2026-09-10; primer: [`learning_guide.md`](learning_guide.md)) — the matrix that becomes the paper's Section 2 and decides which baselines are reproducible. The goal of this step is to be able to say precisely *what has never been tested.*

### Step 1 — Tiered-realism simulation suite (weeks 3–6)
- **Tier 0** toy: power-law lens + Sérsic source — what most ML papers train on.
- **Tier 1** realistic sources: real HST/COSMOS galaxies via `galsim`.
- **Tier 2** lens complexity — the confounders: multipoles (m=3,4), disks/bars, external shear, line-of-sight halos, unresolved low-mass populations.
- **Tier 3** instrument realism: HST/ACS, JWST/NIRCam, Euclid/VIS, Roman/WFI, Rubin — PSF, pixel scale, depth.
- **Subhalo populations** from CDM / WDM / SIDM with hidden truth per image.

### Step 2 — Reproduce and run the open methods as published (weeks 6–11)
Three or four of: a UNet detector (Tsang/Hughes-style); NPE and/or TMNRE population inference (`sbi`, `swyft`); gravitational imaging (`PyAutoLens`); wavelet potential corrections (`herculens`). Do **not** improve them in this paper; reproduce first, then run on every tier.

### Step 3 — Stress test (weeks 11–14)
Metrics below. The headline table: how often each method reports substructure when a Tier 2 confounder is on and there is none.

### Step 4 — Gap analysis → paper 2 (weeks 14–16)
The stress test will show which failure is largest and unaddressed — confounder-induced false positives, calibration collapse under LOS degeneracy, sim-to-real transfer, or something unexpected. **Paper 2 is the method that fixes the biggest one**, now with proof that it matters. If nothing there is compelling, paper 2 is project 01.

**Paper 1:** "How robust is dark-matter substructure inference from strong lenses? A controlled stress test of published methods." Single author is fine; credibility comes from released code, fixed seeds, published configs, and conservative claims. The suite goes public as a by-product — no leaderboard, no organizing. If others adopt it, good; not required.

### Where the industry edge lands
Reproducing other people's research code, building controlled evaluation pipelines, dataset engineering, honest metrics — industry ML's daily work and academia's chronic weak spot.

### Theory content
Proper scoring rules and calibration metrics for posteriors; a degeneracy taxonomy (mass-sheet; multipole–substructure; LOS–subhalo; source complexity; unresolved population). Cite Adam et al. 2026 for the information-geometry treatment rather than redoing it.

### People, minimized
None required for the work. Two unavoidable touch-points, both one email each: an arXiv endorser (or skip arXiv until the journal accepts), and any baseline author whose code won't run (ask once; if silent, document and drop).

## Prior art to read first

1. **Metcalf et al. (2019), *The Strong Gravitational Lens Finding Challenge*** — the template for how a lensing benchmark paper is structured, run, and cited. Read it as a blueprint, not for content.
2. **Tsang et al. 2024** ([2401.16624](https://arxiv.org/abs/2401.16624)) and **Hughes et al. 2024** ([2403.04349](https://arxiv.org/abs/2403.04349)) — ML subhalo detectors and exactly how they fail at realistic concentrations. Your Tier 0→1 gap in one paper.
3. **Dhanasingham et al. 2025** ([2511.17732](https://arxiv.org/abs/2511.17732)) and **Wagner-Carena et al. 2024** ([2404.14487](https://arxiv.org/abs/2404.14487)) — SBI population inference and its acknowledged degeneracies.
4. **Lange et al. 2024** ([2410.12987](https://arxiv.org/abs/2410.12987)), **Shan et al. 2025** ([2510.02805](https://arxiv.org/abs/2510.02805)), **O'Riordan et al. 2025** ([2509.02660](https://arxiv.org/abs/2509.02660)) — the confounders that become Tier 2.
5. **Adam et al. 2026** ([2608.18224](https://arxiv.org/abs/2608.18224)) — information geometry of substructure inference; cite for the theory, use `caustics` for differentiable sims.
6. **Zeghal et al. 2024** ([2409.17975](https://arxiv.org/abs/2409.17975)) — an SBI benchmark for weak-lensing cosmology; a recent, well-received example of the paper type.
7. Brehmer et al. 2019; Vegetti & Koopmans 2009; Hezaveh et al. 2017 — the method lineages your baselines come from.
8. Schneider & Sluse 2013 — the mass-sheet degeneracy, for the degeneracy taxonomy.

## Data

### Strong lensing
| Source | What | Link |
|---|---|---|
| **lenstronomy** | Open-source lens modelling/simulation — unlimited labeled images (mass profile, substructure, source, PSF, noise). | https://github.com/lenstronomy/lenstronomy |
| **deeplenstronomy** | Survey-realistic simulation wrapper (DES/LSST/Euclid-like). | https://github.com/deepskies/deeplenstronomy |
| **DeepLense datasets** | Simulated lenses labeled by DM substructure model. | https://github.com/ML4SCI/DeepLense |
| **Euclid Q1** | First public data (Mar 2025) incl. strong-lens candidate catalogs. | ESA Euclid Science Archive — https://eas.esac.esa.int |
| **Strong Lens Finding Challenge** (Bologna Lens Factory) | Benchmark simulated Euclid/ground images. | Metcalf et al. 2019; challenge site |
| **HSC-SSP / DES / KiDS** | Wide-field imaging + published lens candidate lists. | Survey archives |
| **SLACS / BELLS / HST** | Confirmed lenses with HST imaging (real-data test set). | MAST |

### Cosmological SBI (alternative or extension)
| Source | What | Link |
|---|---|---|
| **CAMELS** | Thousands of hydrodynamic sims varying cosmology + feedback; the **CAMELS Multifield Dataset (CMD)**: ~10⁵ 2D maps, 256×256, 13 fields, with labels (Ω_m, σ₈, feedback params). Built for exactly this. | https://camels.readthedocs.io |
| **Quijote** | N-body suite for cosmology inference. | https://quijote-simulations.readthedocs.io |
| **Weak-lensing maps** | DES Y3, KiDS-1000, HSC Y3 public products. | Survey archives |
| **SPARC** | 175 galaxy rotation curves — classic dark-matter evidence; tiny, theory-friendly (halo profile fitting, radial acceleration relation). | http://astroweb.cwru.edu/SPARC/ |

## Method — building the suite and running the baselines

### Simulation suite
- **Engine:** `lenstronomy` via `paltas` (Wagner-Carena's pipeline for SBI training sets — already handles subhalo populations, LOS halos, and survey realism); `caustics`/`herculens` where differentiability is needed for a baseline.
- **Tier 0 → 3** as defined above. Each tier is a config file; every image carries a hidden-truth record (macro-model, every subhalo's mass/position/concentration, LOS halos, confounder flags).
- **Subhalo populations:** CDM and WDM from semi-analytic prescriptions in `paltas`; SIDM from the SIDM Concerto tables; optionally the DREAMS emulator for realistic spatial distributions.
- **Sources:** COSMOS/HST galaxies through `galsim` (Tier 1+). **Instruments:** PSF, pixel scale, depth for HST/ACS, JWST/NIRCam, Euclid/VIS, Roman/WFI, Rubin.
- **Scale:** ~10⁴ images per tier per instrument is enough for detector training and for population inference; all CPU.

### Baselines (run as published; do not "improve" them in Phase 0)
1. **UNet detector** — Tsang-style segmentation of individual subhalos.
2. **NPE / NRE population inference** — `sbi` (NPE) and `swyft` (TMNRE) on SHMF normalization / WDM half-mode mass.
3. **Gravitational imaging** — `PyAutoLens` pixelated potential corrections.
4. **Wavelet potential corrections** — `herculens`.
5. *(optional)* **Neural-field κ posterior with a learned prior** — the earlier plan, now one baseline among five.

### Protocol
Train (where applicable) on Tier 0; evaluate on every tier; then train on Tier 3 and evaluate again. Report per-tier, per-instrument, per-confounder. Fix random seeds; publish configs, weights, and evaluation scripts.

## Evaluation (the metrics *are* the contribution)

- **Detection:** ROC / precision-recall vs subhalo mass **and concentration**; completeness maps in (mass, projected distance from arc) space.
- **Population inference:** posterior coverage (expected vs empirical), SBC rank histograms, bias vs truth, proper scoring rules (e.g. log score / energy score) — reported per tier.
- **Confounder-induced false positives:** for each Tier 2 confounder switched on with *no* subhalos, the rate at which each method reports substructure. This is the headline table.
- **Realism degradation:** every metric as a function of tier; sim-to-sim transfer (train Tier 0 → test Tier 3).
- **Cost:** wall-clock and hardware per lens per method.

## Theory component

- **Proper scoring rules and calibration** for posterior evaluation — what "a good posterior" means when truth is a population parameter and the model is misspecified.
- **Degeneracy taxonomy:** mass-sheet (Schneider & Sluse 2013), multipole–substructure (Lange 2024), LOS–subhalo (Dhanasingham 2025), source-complexity–substructure (Hughes 2024), unresolved population (O'Riordan 2025). Organized, with the observables that break each. Cite Adam et al. 2026 for the Fisher treatment rather than redoing it.

## Compute plan

CPU for simulation and small flows (64–128 px images). A small rented GPU speeds up training on 10⁵ images but isn't required. CMD is a few GB per field.

## Milestones

| Month | Deliverable |
|---|---|
| 1 (Sep) | **Step 0 methods survey** (`methods_survey.md`). Tier 0–1 sims with COSMOS sources; subhalo population generators; metric definitions. Week-one blog visuals. |
| 2 (Oct) | Tiers 2–3. UNet + NPE/TMNRE baselines reproduced and running. |
| 3 (Nov) | Gravitational imaging + wavelet baselines. Full result grid; calibration and confounder-FPR analysis. |
| 4 (Dec) | Gap analysis. Paper 1 draft; suite v0 public. |
| 5 (Jan 2027) | Submit paper 1 (A&A / MNRAS / RASTI) + arXiv. **Choose paper 2** from the gap analysis, or start 01. |

## Risks

- **Single-author credibility.** Mitigation: everything reproducible (code, seeds, configs, weights); conservative claims; frame as "we measured", not "they are wrong".
- **Baselines are other people's code with sharp edges.** Budget one week each; accept partial coverage; a method that cannot be reproduced from its release is itself a result worth one honest paragraph.
- **Realism disputes.** Referees will say the sims aren't realistic enough. Mitigation: tiered realism, everything open, and explicit statement of what each tier does and does not include.
- **Someone publishes a similar evaluation first.** A stress test doesn't need exclusivity — two independent evaluations strengthen each other. Publish fast; don't over-scope.
- **Scope.** Keep Tier 2 to 2–3 confounders and Tier 3 to 3 instruments before the first submission.

## Website by-product

- **The benchmark explorer:** browse simulated lenses, toggle subhalos on/off and watch the arcs change, flip confounders on to see how bars/multipoles mimic substructure. Outreach and a genuinely useful research tool in one page.
An interactive lens: drag a source galaxy behind a mass profile and watch arcs form; slider for a mass sheet to *show* the degeneracy.

## Venues

- **Paper 1:** A&A or MNRAS; RASTI as fallback. ML4PS workshop version if the timing fits (late Aug).
- **arXiv:** astro-ph.CO + astro-ph.IM + cs.LG. Needs one endorser for a first astro-ph submission — or post after journal acceptance.

## Key references

- Brehmer et al. 2019, ApJ — SBI for subhalo populations
- Hezaveh, Perreault Levasseur & Marshall 2017, Nature — CNN lens modelling
- Metcalf et al. 2019 — Strong Lens Finding Challenge
- Alexander et al. 2020 — DeepLense
- Euclid Collaboration 2025 — Q1 strong-lens papers
- Schneider & Sluse 2013 — mass-sheet degeneracy
- Birrer & Amara 2018 — lenstronomy
- Villaescusa-Navarro et al. 2021, 2022 — CAMELS, CMD
- Lelli, McGaugh & Schombert 2016 — SPARC
- Metcalf et al. 2019 — Strong Lens Finding Challenge (the citation-pattern precedent); Ding et al. 2021 — TDLMC; Lueckmann et al. 2021 — SBI Benchmark
- Adam et al. 2026 — information geometry of substructure inference ([2608.18224](https://arxiv.org/abs/2608.18224))
- Tsang 2024, Hughes 2024, Campbell 2026 — ML subhalo detectors; Dhanasingham 2025, Filipp 2026, Wagner-Carena 2024 — SBI population inference
- Lange 2024, Shan 2025, O'Riordan 2025, Kollmann 2025 — confounders and detectability
- Boruah 2025, Remy 2026, Royo 2026 — learned-prior mass-map posteriors (weak lensing / clusters)
- Vegetti & Koopmans 2009, MNRAS — gravitational imaging (pixelated potential corrections)
- Biggio et al. 2023, A&A — continuous neural fields for lens potentials
- Mishra-Sharma & Yang 2022 — neural-field source reconstruction
- Legin et al. 2021 — SBI of lens parameters; Wagner-Carena et al. 2023, ApJ — end-to-end substructure inference from hundreds of lenses
- Anau Montel et al. 2023, MNRAS — WDM mass from lensing with truncated marginal neural ratio estimation (`swyft`)
- Despali et al. 2022 — sensitivity function for substructure detection
- Adam et al. 2022 — pixelated source/lens reconstruction with score-based priors
- Jeffrey et al. 2021 — DES Y3 mass maps
- AION-1 (Polymathic 2025) — https://arxiv.org/abs/2510.17960
