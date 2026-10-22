# Detector v0 — evaluation results

Threshold calibrated for 10% FPR on `no_subhalo` (n=1000): **0.9708**  ·  ROC AUC (in-distribution, all masses) = **0.675**


## RQ4 — completeness vs. concentration (paired lenses, same mass & position)

| mass bin (log10 Msun) | n | completeness c=60 | completeness c=15 |
|---|---|---|---|
| 8.0-8.5 | 150 | 19.3% | 13.3% |
| 8.5-9.0 | 156 | 14.7% | 15.4% |
| 9.0-9.5 | 136 | 15.4% | 8.1% |
| 9.5-10.0 | 132 | 37.9% | 9.8% |
| 10.0-10.5 | 178 | 62.9% | 21.9% |
| 10.5-11.0 | 153 | 81.0% | 29.4% |

**Paired flip test** (905 identical lenses, only concentration changed): detected at c=60: 359, at c=15: 152. 239 lenses flipped from detected->missed when concentration dropped; 32 flipped the other way.

## RQ2 — confounder-induced false-positive rate (zero subhalos present)

| condition | FPR |
|---|---|
| no_subhalo (calibration set) | 10.0% |
| multipole m=4, a=0.03*thetaE | 15.3% |