# Detector v0 — evaluation results

Threshold calibrated for 10% FPR on `no_subhalo` (n=1000): **0.9809**  ·  ROC AUC (in-distribution, all masses) = **0.708**


## RQ4 — completeness vs. concentration (paired lenses, same mass & position)

| mass bin (log10 Msun) | n | completeness c=60 | completeness c=15 |
|---|---|---|---|
| 8.0-8.5 | 150 | 8.7% | 9.3% |
| 8.5-9.0 | 156 | 10.9% | 7.7% |
| 9.0-9.5 | 136 | 27.9% | 11.8% |
| 9.5-10.0 | 132 | 64.4% | 15.2% |
| 10.0-10.5 | 178 | 78.7% | 23.0% |
| 10.5-11.0 | 153 | 85.0% | 39.2% |

**Paired flip test** (905 identical lenses, only concentration changed): detected at c=60: 423, at c=15: 163. 296 lenses flipped from detected->missed when concentration dropped; 36 flipped the other way.

## RQ2 — confounder-induced false-positive rate (zero subhalos present)

| condition | FPR |
|---|---|
| no_subhalo (calibration set) | 10.0% |
| multipole m=4, a=0.01*thetaE | 9.7% |
| multipole m=4, a=0.03*thetaE | 9.7% |