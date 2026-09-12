# Problem statement — v1 (2026-09-10)

Built on [`fulltext_findings.md`](fulltext_findings.md). Everything below is designed so that the paper's claims are *measurements*, its tables are *reference material*, and its suite is *reused* — the three properties that make an evaluation paper citable for a decade.

---

## 1. The problem in one paragraph

Strong-lensing substructure inference is how the community proposes to test the particle nature of dark matter with the ~10⁴–10⁵ lenses Euclid, Rubin and Roman will deliver. At least five method families now claim to detect or constrain dark-matter substructure from lens images. **Every one of them has been validated only on its own simulations, and every one of those simulations omits at least one physical effect that a different paper has shown can mimic or hide substructure**: angular complexity of the lens (Lange 2024: Bayes factor 60→11; O'Riordan 2025: a ≤3% multipole amplitude erases the population signal; Nightingale 2024: 8 of 54 false positives from an overly simple mass model), line-of-sight halos (Şengül 2022: the field's "first dark perturber" was one), lens light (Nightingale: 16 of 34 candidate detections), source structure (Adam 2026: sensitivity suppressed "across a broad range of scales"; Vegetti 2023: analytic-source methods "biased towards models that are colder"), and subhalo concentration (Tsang 2024: at ΛCDM concentrations the detector performs "around 0.1 one would expect from random guessing"). Their metrics are mutually incomparable — TPR at 10% FPR, 95%-purity completeness, Δln Z thresholds of 10 or 50, exclusion volumes, Pearson correlations. Posterior calibration, where tested at all, is tested sim-to-same-sim. The field's own review states that "a detailed investigation of this issue as a function of data type and quality is still lacking," and its leading practitioners disagree about whether the science is limited by data (Vegetti+ 2023) or by methodology (Wagner-Carena+ 2024). **Nobody has put the methods on one simulation suite, switched the confounders on one at a time, and measured what happens.** That measurement is this paper.

---

## 2. Title candidates

1. *Substructure or systematics? A controlled stress test of dark-matter substructure inference from strong gravitational lenses*
2. *How robust is dark-matter substructure inference from strong lenses? Confounders, concentrations and calibration on a common simulation suite*
3. *One suite, five methods: measuring the false-positive rates and calibration of strong-lensing substructure inference under realistic lens complexity*

Option 1 for a journal; option 3 if a workshop version is cut.

---

## 3. Research questions and falsifiable hypotheses

| RQ | Question | Hypothesis (direction predicted) | Headline output |
|---|---|---|---|
| **RQ1 — Comparability** | On identical images and metrics, how do families A–E compare in detection completeness vs (mass, concentration) and in population-parameter bias? | Rankings from the literature do not survive a common metric; the parametric scan (A) and population SBI (D) dominate in different regimes. | Completeness surfaces in the (mass, concentration) plane, one per method. |
| **RQ2 — Confounder false positives** | With **zero** subhalos present, what fraction of lenses does each method flag as containing substructure when one confounder is switched on: multipoles ≤1% / ≤3%, a disk or bar, a line-of-sight population, an unresolved sub-threshold population, a complex (real) source, imperfect lens-light subtraction? | Every method exceeds its nominal FPR by ≥2× for at least one confounder; multipoles and source complexity dominate; SBI population posteriors shift by more than their reported 68% widths. | **The confounder-FPR table** — methods × confounders. The paper's most-quoted object. |
| **RQ3 — Calibration under realism shift** | Train/tune on realism tier *k*, test on tier *k+1*: is posterior coverage still nominal? By how much do credible intervals become overconfident? | Coverage collapses for NPE/NRE when angular complexity or LOS halos are added post-training; SNPE-style targeting does not fix it. | Coverage-vs-tier curves (`tarp`), one per SBI method. |
| **RQ4 — Concentration realism** | How much of the published detector performance is a concentration artifact? Evaluate on fixed c=60, fixed c=15, and a ΛCDM c–M relation with scatter. | Completeness at c–M+scatter is at most half the c=60 numbers for 10⁹–10⁹·⁵ M☉; the mass threshold moves up by ≥0.5 dex. | Completeness vs mass at three concentration models. |
| **RQ5 — Instrument transfer** | HST → Euclid VIS (0.1″, 0.17″) → Rubin (0.2″, seeing): what survives? | Individual detection is near chance below 10⁹·⁵ at Euclid resolution (consistent with O'Riordan 2023's one detection per ~70 lenses); population inference retains signal but with the RQ2/RQ3 pathologies amplified. | Same tables at three instruments. |
| **RQ6 — Reproducibility** | Can the four ML detectors be reimplemented from the text and reproduce their reported numbers? | At least one cannot be reproduced to within its reported completeness without undocumented choices. | A reproducibility ledger (what was specified, what had to be guessed, what number resulted). |

If the hypotheses are *wrong* — methods survive the confounders — that is an equally publishable, more reassuring result. The design does not depend on the answer.

---

## 4. Experimental design

### 4.1 Suite (hidden truth per image; every knob a config flag)
- **Tier 0 — as published:** Sérsic source, EPL + shear, no lens light, tNFW at fixed c (60 and 15), 0–1 subhalo, HST-like Gaussian PSF. This reproduces the literature's operating point.
- **Tier 1 — realistic source:** real COSMOS/HDF galaxies (via `paltas`/`galsim`), redshift-appropriate.
- **Tier 2 — lens complexity (the confounders, individually switchable):** multipoles m=1,3,4 at ≤1% and ≤3%; disk/bar component; external shear; **line-of-sight population** (Sheth-Tormen, two-halo term, multi-plane); **unresolved sub-threshold population** (f_sub = 0.01 down to 10⁶ M☉); lens light with imperfect subtraction (residual at 1–5%).
- **Tier 3 — instruments:** HST/ACS (0.05″, Tiny Tim PSF, drizzling), JWST/NIRCam (0.031″), Euclid/VIS (0.1″, 0.17″), Rubin (0.2″, 0.7″ seeing), each with correlated noise where appropriate.
- **Concentration grid:** fixed c=60; fixed c=15; ΛCDM c–M (Diemer & Joyce / Ludlow) with lognormal scatter; optional steep-slope gNFW (Kollmann) as a stress case.
- **Populations:** CDM; WDM (half-mode mass grid); SIDM profiles from public tables. Hidden truth recorded per image in a COOLEST-style JSON.
- **Scale:** ~10⁴ images per (tier, instrument) for detectors and population inference; ~30–50 lenses per cell for the slow parametric scan.

### 4.2 Methods (run as published — no improvement in this paper)
| Family | Baseline | Source | Protocol |
|---|---|---|---|
| A | PyAutoLens subhalo scan (Nightingale recipe), Δln Z thresholds 10 *and* 50 | open | Subset per cell (compute-bound). Also yields sensitivity maps à la Despali. |
| B | herculens wavelet potential corrections (Galan recipe) | open | Per-lens; report recovered perturbation power. |
| C | U-Net detector reimplemented from Ostdiek/Tsang; ResNet binary from Hughes | **none public → reimplement** | Train at Tier 0 exactly as described; report reproduction gap (RQ6); then evaluate on all tiers. |
| D | paltas NPE + hierarchical (Wagner-Carena 2023); optional swyft TMNRE (Anau Montel) | open engines | Train at Tier 0/1 as published; evaluate Tiers 2–3 (RQ3); coverage with `tarp`. |
| E | Fagin GRF_ML (public) | open | Run on all tiers; test the "GRF breaks with few massive subhalos" caveat directly. |

### 4.3 Metrics (fixed before any result is looked at)
- **Detection:** completeness vs (mass, concentration, distance from arc) at a **common** FPR of 10% *and* at each method's published operating point; ROC/PR curves.
- **False positives under confounders:** fraction of zero-subhalo lenses flagged, per confounder, per method; for parametric scans the Δln Z distribution of the best spurious subhalo.
- **Population inference:** bias and 68% width of Σ_sub / M_hm posteriors vs truth; **coverage** (`tarp`, SBC rank histograms); proper scoring (log score).
- **Cost:** wall-clock and hardware per lens per method.
- **Reproducibility:** ledger fields — specified / inferred / guessed; reported vs reproduced number.

### 4.4 Controls
Fixed seeds; identical noise realisations across methods within a cell; every config, weight and evaluation script released; the assumption matrix (findings §1) reproduced as the paper's Table 1 with the suite's own row added.

---

## 5. Deliverables — and why each gets cited

| Deliverable | Why it is reused |
|---|---|
| **Table 1 — the assumption matrix** (25 papers × 11 realism dimensions) | Every future method paper must place itself on it. Reference table → citation by default. |
| **Table 2 — the confounder-FPR table** | The quotable numbers ("method X reports substructure in Y% of lenses with a 2% m=4 multipole and no subhalos"). |
| **Figure — calibration vs realism tier** | The first cross-method coverage test under misspecification in this field. |
| **Figure — completeness at realistic concentrations** | Corrects the operating point the detector literature has been reporting at. |
| **The suite** (configs, generators, hidden-truth format, evaluation scripts) | Zero-cost adoption for anyone validating a new method; every adoption is a citation. |
| **Reproducibility ledger** | Rarely done, widely appreciated; establishes the author as careful. |

Precedents for the citation pattern: SBI Benchmark (Lueckmann+ 2021), Strong Lens Finding Challenge (Metcalf+ 2019), TDLMC (Ding+ 2021), Zeghal+ 2024 weak-lensing SBI benchmark. None required organising a community; all released a suite and a result table.

---

## 6. Framing sentences for the abstract (draft)

> Strong-lensing substructure inference will move from tens of lenses to tens of thousands with Euclid, Rubin and Roman, yet the methods proposed to do it have been validated only on their authors' own simulations. We assemble a tiered-realism simulation suite with hidden truth and evaluate five families of published methods — parametric perturber scans, free-form potential corrections, machine-learning detectors, population-level simulation-based inference, and field-level statistics — on identical images and metrics. Switching realistic lens complexity on one effect at a time with no substructure present, we measure each method's false-positive rate under multipoles, disks, line-of-sight halos, unresolved populations, source structure and lens-light residuals; we test posterior calibration when the test data are more realistic than the training data; and we re-measure detector completeness at ΛCDM concentrations. We find [numbers]. We release the suite, configurations and a reproducibility ledger so that future methods can be evaluated at a common operating point.

---

## 7. Scope guards

**In:** measurement, comparison, calibration, reproduction, released suite.
**Out:** any new method; any real-data detection claim (an appendix running detectors on Nightingale's 54 lenses for *consistency*, not discovery, is optional); a leaderboard or challenge; anything requiring collaborators.
**Compute:** CPU for simulation, U-Net/ResNet training (small images), NPE; a rented GPU for the final training runs; PyAutoLens scans limited to subsets.
**Timeline:** unchanged from `plan.md` — submit ~Jan 2027.

---

## 8. Risks specific to this framing

- **"The sims aren't realistic enough."** Always true. Answer: tiers are explicit, everything is open, the Tier 0 row reproduces the literature's own operating point so critics are arguing with their own papers.
- **Baselines not run "properly".** Run each exactly as published, document every deviation, and invite corrections publicly after release (no organising required).
- **A hypothesis fails.** Report it. A method that survives the confounders is news too.
- **Someone publishes first.** Two independent stress tests corroborate each other; ship the Tier 0–2 result even if Tier 3 slips.

---

## 9. What paper 2 becomes

The confounder-FPR table will name the largest unaddressed failure. Paper 2 is the method that fixes it — multipole-aware training, LOS-aware SBI, calibrated free-form maps, or concentration-realistic detectors — now with proof it matters. If none is compelling, paper 2 is project 01.
