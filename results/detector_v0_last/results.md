# Detector v0 — evaluation results

Threshold calibrated for 10% FPR on `no_subhalo` (n=1000): **0.9739**  ·  ROC AUC (in-distribution, all masses) = **0.649**


## RQ4 — completeness vs. concentration (paired lenses, same mass & position)

| mass bin (log10 Msun) | n | completeness c=60 | completeness c=15 |
|---|---|---|---|
| 8.0-8.5 | 150 | 8.0% | 8.7% |
| 8.5-9.0 | 156 | 9.0% | 7.7% |
| 9.0-9.5 | 136 | 17.6% | 9.6% |
| 9.5-10.0 | 132 | 37.1% | 15.2% |
| 10.0-10.5 | 178 | 50.0% | 11.8% |
| 10.5-11.0 | 153 | 66.7% | 11.8% |

**Paired flip test** (905 identical lenses, only concentration changed): detected at c=60: 290, at c=15: 97. 232 lenses flipped from detected->missed when concentration dropped; 39 flipped the other way.

## RQ2 — confounder-induced false-positive rate (zero subhalos present)

| condition | FPR |
|---|---|
| no_subhalo (calibration set) | 10.0% |
| multipole m=4, a=0.01*thetaE | 10.9% |
| multipole m=4, a=0.03*thetaE | 12.5% |