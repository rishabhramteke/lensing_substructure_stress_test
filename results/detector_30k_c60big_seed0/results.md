# Detector v0 — evaluation results

Threshold calibrated for 10% FPR on `no_subhalo` (n=1000): **0.9901**  ·  ROC AUC (in-distribution, all masses) = **0.711**


## RQ4 — completeness vs. concentration (paired lenses, same mass & position)

| mass bin (log10 Msun) | n | completeness c=60 | completeness c=15 |
|---|---|---|---|
| 8.0-8.5 | 150 | 9.3% | 8.0% |
| 8.5-9.0 | 156 | 14.1% | 14.7% |
| 9.0-9.5 | 136 | 32.4% | 4.4% |
| 9.5-10.0 | 132 | 63.6% | 14.4% |
| 10.0-10.5 | 178 | 78.1% | 26.4% |
| 10.5-11.0 | 153 | 86.3% | 44.4% |

**Paired flip test** (905 identical lenses, only concentration changed): detected at c=60: 435, at c=15: 175. 294 lenses flipped from detected->missed when concentration dropped; 34 flipped the other way.

## RQ2 — confounder-induced false-positive rate (zero subhalos present)

| condition | FPR |
|---|---|
| no_subhalo (calibration set) | 10.0% |
| multipole m=4, a=0.01*thetaE | 10.3% |
| multipole m=4, a=0.03*thetaE | 12.7% |