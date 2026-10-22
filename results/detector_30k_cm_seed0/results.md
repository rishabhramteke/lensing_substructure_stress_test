# Detector v0 — evaluation results

Threshold calibrated for 10% FPR on `no_subhalo` (n=1000): **0.8112**  ·  ROC AUC (in-distribution, all masses) = **0.474**


## RQ4 — completeness vs. concentration (paired lenses, same mass & position)

| mass bin (log10 Msun) | n | completeness c=60 | completeness c=15 |
|---|---|---|---|
| 8.0-8.5 | 150 | 11.3% | 11.3% |
| 8.5-9.0 | 156 | 5.1% | 5.8% |
| 9.0-9.5 | 136 | 10.3% | 8.8% |
| 9.5-10.0 | 132 | 10.6% | 10.6% |
| 10.0-10.5 | 178 | 9.0% | 9.0% |
| 10.5-11.0 | 153 | 3.3% | 6.5% |

**Paired flip test** (905 identical lenses, only concentration changed): detected at c=60: 74, at c=15: 78. 2 lenses flipped from detected->missed when concentration dropped; 6 flipped the other way.

## RQ2 — confounder-induced false-positive rate (zero subhalos present)

| condition | FPR |
|---|---|
| no_subhalo (calibration set) | 10.0% |
| multipole m=4, a=0.01*thetaE | 10.3% |
| multipole m=4, a=0.03*thetaE | 10.1% |