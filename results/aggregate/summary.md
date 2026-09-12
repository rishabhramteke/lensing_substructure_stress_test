# Multi-seed aggregate (n=4 independent training runs)

AUC across seeds: 0.619 ± 0.018  (values: [np.float64(0.623), np.float64(0.593), np.float64(0.627), np.float64(0.634)])

## RQ2 — confounder FPR, mean ± std across seeds

| condition | mean FPR | std | resolved at 2σ from baseline? |
|---|---|---|---|
| no_subhalo (calibration set) | 10.0% | 0.0% | — |
| multipole m=4, a=0.01*thetaE | 11.3% | 2.1% | no |
| multipole m=4, a=0.03*thetaE | 10.5% | 1.9% | no |

## RQ4 — completeness by mass bin, mean ± std across seeds

| mass bin | c=60 mean±std | c=15 mean±std |
|---|---|---|
| 8.0-8.5 | 11.7% ± 5.0% | 9.3% ± 2.7% |
| 8.5-9.0 | 8.0% ± 1.1% | 8.7% ± 0.8% |
| 9.0-9.5 | 14.7% ± 3.8% | 12.7% ± 4.0% |
| 9.5-10.0 | 28.6% ± 7.8% | 11.0% ± 2.8% |
| 10.0-10.5 | 51.0% ± 10.1% | 15.4% ± 4.4% |
| 10.5-11.0 | 63.4% ± 4.2% | 13.2% ± 3.1% |

**Paired flip, averaged over seeds:** c60-not-c15 = 198.8 ± 41.2, c15-not-c60 = 30.2 ± 16.8  (ratio of means: 6.6:1)