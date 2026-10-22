# Detector v0 — evaluation results

Threshold calibrated for 10% FPR on `no_subhalo` (n=1000): **0.9136**  ·  ROC AUC (in-distribution, all masses) = **0.503**


## RQ4 — completeness vs. concentration (paired lenses, same mass & position)

| mass bin (log10 Msun) | n | completeness c=60 | completeness c=15 |
|---|---|---|---|
| 8.0-8.5 | 150 | 10.7% | 10.7% |
| 8.5-9.0 | 156 | 7.7% | 7.7% |
| 9.0-9.5 | 136 | 7.4% | 8.1% |
| 9.5-10.0 | 132 | 12.9% | 12.9% |
| 10.0-10.5 | 178 | 12.9% | 11.2% |
| 10.5-11.0 | 153 | 11.8% | 5.9% |

**Paired flip test** (905 identical lenses, only concentration changed): detected at c=60: 96, at c=15: 85. 14 lenses flipped from detected->missed when concentration dropped; 3 flipped the other way.

## RQ2 — confounder-induced false-positive rate (zero subhalos present)

| condition | FPR |
|---|---|
| no_subhalo (calibration set) | 10.0% |
| multipole m=4, a=0.03*thetaE | 10.9% |