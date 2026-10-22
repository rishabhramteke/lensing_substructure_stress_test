# Detector v0 — evaluation results

Threshold calibrated for 10% FPR on `no_subhalo` (n=1000): **0.8131**  ·  ROC AUC (in-distribution, all masses) = **0.480**


## RQ4 — completeness vs. concentration (paired lenses, same mass & position)

| mass bin (log10 Msun) | n | completeness c=60 | completeness c=15 |
|---|---|---|---|
| 8.0-8.5 | 150 | 12.7% | 13.3% |
| 8.5-9.0 | 156 | 8.3% | 9.0% |
| 9.0-9.5 | 136 | 9.6% | 9.6% |
| 9.5-10.0 | 132 | 12.9% | 13.6% |
| 10.0-10.5 | 178 | 9.6% | 9.6% |
| 10.5-11.0 | 153 | 4.6% | 5.9% |

**Paired flip test** (905 identical lenses, only concentration changed): detected at c=60: 86, at c=15: 91. 2 lenses flipped from detected->missed when concentration dropped; 7 flipped the other way.

## RQ2 — confounder-induced false-positive rate (zero subhalos present)

| condition | FPR |
|---|---|
| no_subhalo (calibration set) | 10.0% |
| multipole m=4, a=0.01*thetaE | 11.4% |
| multipole m=4, a=0.03*thetaE | 10.9% |