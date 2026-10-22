# Detector v0 — evaluation results

Threshold calibrated for 10% FPR on `no_subhalo` (n=1000): **0.8611**  ·  ROC AUC (in-distribution, all masses) = **0.498**


## RQ4 — completeness vs. concentration (paired lenses, same mass & position)

| mass bin (log10 Msun) | n | completeness c=60 | completeness c=15 |
|---|---|---|---|
| 8.0-8.5 | 150 | 10.7% | 10.7% |
| 8.5-9.0 | 156 | 8.3% | 7.7% |
| 9.0-9.5 | 136 | 10.3% | 10.3% |
| 9.5-10.0 | 132 | 17.4% | 14.4% |
| 10.0-10.5 | 178 | 11.8% | 9.6% |
| 10.5-11.0 | 153 | 13.7% | 7.8% |

**Paired flip test** (905 identical lenses, only concentration changed): detected at c=60: 108, at c=15: 90. 21 lenses flipped from detected->missed when concentration dropped; 3 flipped the other way.

## RQ2 — confounder-induced false-positive rate (zero subhalos present)

| condition | FPR |
|---|---|
| no_subhalo (calibration set) | 10.0% |
| multipole m=4, a=0.01*thetaE | 11.4% |
| multipole m=4, a=0.03*thetaE | 13.3% |