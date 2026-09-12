"""Family D baseline: amortized simulation-based inference (NPE), reimplementing
the *algorithm* of Wagner-Carena+2023/Brehmer+2019 (train a neural density
estimator on (image, subhalo-parameter) pairs, get a calibrated posterior per
lens) directly on `sbi` + a small CNN embedding net, rather than depending on
`paltas` (which does not install on Python 3.14 -- same issue as project 05's
main simulator; see plan.md's 2026-09-10 decision).

Two stated simplifications relative to the literature's own pipelines:

1. **Single scalar target, not a hierarchical population fit.** We infer
   log10(subhalo mass) *per lens*, with a floor value (6.0, below the 8-11
   simulation prior) standing in for "no detectable subhalo" -- not
   Wagner-Carena's population-level SHMF normalization inferred jointly
   across many lenses. This keeps the comparison to Family A/C, which are
   also per-lens detectors, apples-to-apples; it is not a like-for-like
   reproduction of the field's actual population-inference deliverable.
2. **No line-of-sight halos, no mass-concentration marginalization** -- same
   Tier-0 scope as the rest of this project's baselines.

What is NOT simplified: the actual inference algorithm. Training uses `sbi`'s
NPE (`sbi.inference.NPE`, a normalizing-flow conditional density estimator),
the same class of method (amortized neural posterior estimation) the
literature uses, not a re-badged classifier.
"""
