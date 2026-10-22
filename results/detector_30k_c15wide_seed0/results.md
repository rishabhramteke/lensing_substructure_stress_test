# Detector v0 — evaluation results

Threshold calibrated for 10% FPR on `no_subhalo` (n=1000): **0.8173**  ·  ROC AUC (in-distribution, all masses) = **0.480**


## RQ4 — completeness vs. concentration (paired lenses, same mass & position)

| mass bin (log10 Msun) | n | completeness c=60 | completeness c=15 |
|---|---|---|---|
| 8.0-8.5 | 150 | 6.0% | 5.3% |
| 8.5-9.0 | 156 | 9.6% | 9.6% |
| 9.0-9.5 | 136 | 3.7% | 4.4% |
| 9.5-10.0 | 132 | 11.4% | 11.4% |
| 10.0-10.5 | 178 | 7.9% | 7.9% |
| 10.5-11.0 | 153 | 5.2% | 5.9% |

**Paired flip test** (905 identical lenses, only concentration changed): detected at c=60: 66, at c=15: 67. 2 lenses flipped from detected->missed when concentration dropped; 3 flipped the other way.

## RQ2 — confounder-induced false-positive rate (zero subhalos present)

| condition | FPR |
|---|---|
| no_subhalo (calibration set) | 10.0% |
| multipole m=4, a=0.01*thetaE | 10.9% |
| multipole m=4, a=0.03*thetaE | 10.3% |