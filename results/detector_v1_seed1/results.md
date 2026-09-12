# Detector v0 — evaluation results

Threshold calibrated for 10% FPR on `no_subhalo` (n=1000): **0.9675**  ·  ROC AUC (in-distribution, all masses) = **0.593**


## RQ4 — completeness vs. concentration (paired lenses, same mass & position)

| mass bin (log10 Msun) | n | completeness c=60 | completeness c=15 |
|---|---|---|---|
| 8.0-8.5 | 150 | 8.0% | 8.0% |
| 8.5-9.0 | 156 | 9.6% | 9.6% |
| 9.0-9.5 | 136 | 11.8% | 12.5% |
| 9.5-10.0 | 132 | 18.9% | 14.4% |
| 10.0-10.5 | 178 | 36.5% | 9.0% |
| 10.5-11.0 | 153 | 57.5% | 9.8% |

**Paired flip test** (905 identical lenses, only concentration changed): detected at c=60: 221, at c=15: 94. 145 lenses flipped from detected->missed when concentration dropped; 18 flipped the other way.

## RQ2 — confounder-induced false-positive rate (zero subhalos present)

| condition | FPR |
|---|---|
| no_subhalo (calibration set) | 10.0% |
| multipole m=4, a=0.01*thetaE | 9.5% |
| multipole m=4, a=0.03*thetaE | 8.9% |