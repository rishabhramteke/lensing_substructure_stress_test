# Detector v0 — evaluation results

Threshold calibrated for 10% FPR on `no_subhalo` (n=1000): **0.8317**  ·  ROC AUC (in-distribution, all masses) = **0.488**


## RQ4 — completeness vs. concentration (paired lenses, same mass & position)

| mass bin (log10 Msun) | n | completeness c=60 | completeness c=15 |
|---|---|---|---|
| 8.0-8.5 | 150 | 14.7% | 14.7% |
| 8.5-9.0 | 156 | 10.9% | 10.3% |
| 9.0-9.5 | 136 | 6.6% | 6.6% |
| 9.5-10.0 | 132 | 9.8% | 9.8% |
| 10.0-10.5 | 178 | 6.7% | 7.3% |
| 10.5-11.0 | 153 | 7.8% | 5.2% |

**Paired flip test** (905 identical lenses, only concentration changed): detected at c=60: 85, at c=15: 81. 7 lenses flipped from detected->missed when concentration dropped; 3 flipped the other way.

## RQ2 — confounder-induced false-positive rate (zero subhalos present)

| condition | FPR |
|---|---|
| no_subhalo (calibration set) | 10.0% |
| multipole m=4, a=0.01*thetaE | 13.6% |
| multipole m=4, a=0.03*thetaE | 11.9% |