# Detector v0 — evaluation results

Threshold calibrated for 10% FPR on `no_subhalo` (n=1000): **0.9858**  ·  ROC AUC (in-distribution, all masses) = **0.657**


## RQ4 — completeness vs. concentration (paired lenses, same mass & position)

| mass bin (log10 Msun) | n | completeness c=60 | completeness c=15 |
|---|---|---|---|
| 8.0-8.5 | 150 | 10.0% | 8.0% |
| 8.5-9.0 | 156 | 9.0% | 6.4% |
| 9.0-9.5 | 136 | 9.6% | 6.6% |
| 9.5-10.0 | 132 | 39.4% | 9.8% |
| 10.0-10.5 | 178 | 57.9% | 12.4% |
| 10.5-11.0 | 153 | 75.2% | 21.6% |

**Paired flip test** (905 identical lenses, only concentration changed): detected at c=60: 312, at c=15: 99. 234 lenses flipped from detected->missed when concentration dropped; 21 flipped the other way.

## RQ2 — confounder-induced false-positive rate (zero subhalos present)

| condition | FPR |
|---|---|
| no_subhalo (calibration set) | 10.0% |
| multipole m=4, a=0.03*thetaE | 10.6% |