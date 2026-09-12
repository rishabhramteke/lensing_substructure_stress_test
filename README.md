# Assumptions under stress — code and data release

Companion repository for *Assumptions under stress: a common-suite test of published dark-matter
substructure detectors for strong lensing* (R. Ramteke, submitted to A&A). Everything needed to
regenerate every population, every fit and every number in the paper.

## Layout

| path | what |
|---|---|
| `src/lensing/` | `lenstronomy`-based simulator: configs (`config.py`), truth sampler and renderer (`simulate.py`), COSMOS source stamps (`cosmos_source.py`), concentration–mass relation, two-component lens light |
| `src/baseline_a/fit.py` | Family A: smooth EPL+shear+Sérsic fit (near-truth or blind initialisation, optional single/double-Sérsic lens light) and the 72-hypothesis subhalo scan |
| `src/detector/` | Family C: U-Net, dataset, training |
| `src/baseline_d/` | the per-lens NPE stand-in (Appendix A, negative result) |
| `scripts/generate_tier0.py` | writes any population from a named config and a seed (`--config`, `--n`, `--seed`, `--out`) |
| `scripts/run_baseline_a.py`, `run_baseline_a_joint.py` | Family A scans: frozen macro-model (`--macro-multipole --macro-multipole-orders 3,4` adds free multipole terms) and joint re-fit at every grid cell with the polish and macro-basin controls (every cell's solution stored) |
| `scripts/run_family_b.py`, `launch_family_b.sh` | Family B: `PyAutoLens` `potential_correction` driver |
| `scripts/train_detector.py`, `evaluate_detector.py` | Family C training and scoring |
| `scripts/evaluate_*.py`, `aggregate_*.py`, `conservative_completeness.py`, `reweight_by_mass_function.py`, `oracle_position_mass_test.py`, `joint_refit_mass_test.py`, `evaluate_lens_light.py`, `noise_decoy_control.py`, `make_decoy_populations.py` | every evaluation in the paper, `completeness_vs_signal.py` (completeness vs perturbation S/N and projected mass), `evaluate_multipole_orders.py` (Table on multipole freedom), `evaluate_joint_vs_frozen.py` (joint vs frozen scan, basin control) |
| `scripts/make_paper_figures.py` | every figure in the paper (`python scripts/make_paper_figures.py fig5` rebuilds one) |
| `data/<population>/manifest.json`, `truth.jsonl` | the hidden truth and regeneration recipe for every population used (image arrays are not stored; regenerate with `generate_tier0.py` from the manifest's config and seed) |
| `checkpoints/unet_*` | trained U-Net weights (Tier-0 seeds, 30k scale-up, Tier-1, c–M training) |
| `results/**/*.json`, `results/**/NOTES.md` | every summary number quoted in the paper |
| `metrics_definitions.md` | the operational definition of every metric |
| `problem_statement.md`, `methods_survey.md`, `first_results.md` | the research questions, the literature survey, and the running lab notebook |

## Environment

Python 3.14, `pip install -r requirements.txt` (pinned versions used for the paper; `lenstronomy`, `galsim` with the COSMOS 23.5 training sample, `PyAutoLens` 2026.9.8.1, `torch` with MPS or CPU). Everything ran on a single laptop CPU/MPS.

## Regenerating a result

```bash
python scripts/generate_tier0.py --config tsang_fixed60 --n 1000 --seed 101 --out data/test_fixed60
python scripts/run_baseline_a.py --population test_fixed60 --n 300 --seed 0 --out results/baseline_a/test_fixed60
python scripts/evaluate_baseline_a.py --root results/baseline_a --out results/baseline_a/summary
python scripts/make_paper_figures.py fig2
```

Seeds and subsample rules are recorded in each `manifest.json`; `metrics_definitions.md` §7 gives the aggregation convention.

## Licence

Code: MIT. Population manifests and results: CC-BY 4.0. The COSMOS stamps and the B1938+666 data belong to their respective releases (`galsim` COSMOS 23.5 sample; Şengül et al. 2022).
