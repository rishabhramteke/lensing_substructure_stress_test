# Tier-1 (COSMOS source) multi-seed aggregate (n=2)

AUC across seeds: 0.484 ± 0.009  (values: [np.float64(0.491), np.float64(0.478)])

## RQ2 — confounder FPR, mean ± std across seeds

| condition | mean FPR | std |
|---|---|---|
| no_subhalo (calibration set) | 10.0% | 0.0% |
| multipole m=4, a=0.01*thetaE | 9.7% | 0.6% |
| multipole m=4, a=0.03*thetaE | 8.8% | 0.5% |

## RQ4 — completeness by mass bin, mean ± std across seeds

| mass bin | c=60 mean±std | c=15 mean±std |
|---|---|---|
| 8.0-8.5 | 11.7% ± 4.6% | 12.0% ± 0.5% |
| 8.5-9.0 | 10.9% ± 2.1% | 10.9% ± 3.1% |
| 9.0-9.5 | 7.6% ± 1.7% | 7.2% ± 1.1% |
| 9.5-10.0 | 6.8% ± 2.9% | 6.5% ± 1.4% |
| 10.0-10.5 | 9.7% ± 1.7% | 10.0% ± 0.4% |
| 10.5-11.0 | 7.6% ± 0.8% | 9.4% ± 0.8% |

**Paired flip, averaged over seeds:** c60-not-c15 = 12.5 ± 3.5, c15-not-c60 = 15.5 ± 4.9  (ratio of means: 0.8:1)