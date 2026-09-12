# Assumptions under stress — a common-suite test of published dark-matter substructure detectors for strong lensing

Code, configurations, seeds, trained weights and evaluation scripts for the paper
*Assumptions under stress: a common-suite test of published dark-matter substructure detectors for strong lensing*
(R. Ramteke, submitted to A&A).

Contents to be released with the accepted version:

- `src/lensing/` — the `lenstronomy`-based simulator (Tier 0 Sérsic sources, Tier 1 COSMOS sources, multipole and two-component lens-light confounders); every population is regenerable from its `manifest.json` and seed.
- `src/baseline_a/` — the parametric perturber scan (frozen and joint-refit variants, blind initialisation, lens-light fitting).
- `scripts/run_family_b.py` — the potential-correction driver built on `PyAutoLens`'s `potential_correction` module.
- `src/detector/` — the U-Net detector and its training/evaluation scripts; released checkpoints.
- `scripts/evaluate_*.py`, `scripts/make_paper_figures.py` — every number and figure in the paper regenerates from these.
- `metrics_definitions.md` — the exact operational definition of every metric.

Until then this repository is a placeholder for the URL cited in the paper.
