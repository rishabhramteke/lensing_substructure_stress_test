# Detector v0 — evaluation results

Threshold calibrated for 10% FPR on `no_subhalo` (n=1000): **0.8450**  ·  ROC AUC (in-distribution, all masses) = **0.484**


## RQ4 — completeness vs. concentration (paired lenses, same mass & position)

| mass bin (log10 Msun) | n | completeness c=60 | completeness c=15 |
|---|---|---|---|
| 8.0-8.5 | 150 | 10.0% | 9.3% |
| 8.5-9.0 | 156 | 10.3% | 10.9% |
| 9.0-9.5 | 136 | 9.6% | 9.6% |
| 9.5-10.0 | 132 | 18.9% | 18.9% |
| 10.0-10.5 | 178 | 13.5% | 13.5% |
| 10.5-11.0 | 153 | 6.5% | 6.5% |

**Paired flip test** (905 identical lenses, only concentration changed): detected at c=60: 103, at c=15: 103. 3 lenses flipped from detected->missed when concentration dropped; 3 flipped the other way.

## RQ2 — confounder-induced false-positive rate (zero subhalos present)

| condition | FPR |
|---|---|
| no_subhalo (calibration set) | 10.0% |
| multipole m=4, a=0.01*thetaE | 14.5% |
| multipole m=4, a=0.03*thetaE | 12.0% |