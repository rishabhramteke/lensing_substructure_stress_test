# Detector v0 — evaluation results

Threshold calibrated for 10% FPR on `no_subhalo` (n=1000): **0.9605**  ·  ROC AUC (in-distribution, all masses) = **0.654**


## RQ4 — completeness vs. concentration (paired lenses, same mass & position)

| mass bin (log10 Msun) | n | completeness c=60 | completeness c=15 |
|---|---|---|---|
| 8.0-8.5 | 150 | 9.3% | 8.7% |
| 8.5-9.0 | 156 | 10.9% | 8.3% |
| 9.0-9.5 | 136 | 17.6% | 7.4% |
| 9.5-10.0 | 132 | 49.2% | 12.9% |
| 10.0-10.5 | 178 | 68.5% | 11.8% |
| 10.5-11.0 | 153 | 77.8% | 34.6% |

**Paired flip test** (905 identical lenses, only concentration changed): detected at c=60: 361, at c=15: 127. 269 lenses flipped from detected->missed when concentration dropped; 35 flipped the other way.

## RQ2 — confounder-induced false-positive rate (zero subhalos present)

| condition | FPR |
|---|---|
| no_subhalo (calibration set) | 10.0% |
| multipole m=4, a=0.01*thetaE | 9.4% |
| multipole m=4, a=0.03*thetaE | 11.5% |