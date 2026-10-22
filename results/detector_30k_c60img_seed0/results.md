# Detector v0 — evaluation results

Threshold calibrated for 10% FPR on `no_subhalo` (n=1000): **0.5019**  ·  ROC AUC (in-distribution, all masses) = **0.505**


## RQ4 — completeness vs. concentration (paired lenses, same mass & position)

| mass bin (log10 Msun) | n | completeness c=60 | completeness c=15 |
|---|---|---|---|
| 8.0-8.5 | 150 | 9.3% | 7.3% |
| 8.5-9.0 | 156 | 10.3% | 10.9% |
| 9.0-9.5 | 136 | 14.7% | 14.0% |
| 9.5-10.0 | 132 | 15.2% | 14.4% |
| 10.0-10.5 | 178 | 11.8% | 11.2% |
| 10.5-11.0 | 153 | 10.5% | 7.8% |

**Paired flip test** (905 identical lenses, only concentration changed): detected at c=60: 107, at c=15: 98. 22 lenses flipped from detected->missed when concentration dropped; 13 flipped the other way.

## RQ2 — confounder-induced false-positive rate (zero subhalos present)

| condition | FPR |
|---|---|
| no_subhalo (calibration set) | 10.0% |
| multipole m=4, a=0.01*thetaE | 10.2% |
| multipole m=4, a=0.03*thetaE | 9.0% |