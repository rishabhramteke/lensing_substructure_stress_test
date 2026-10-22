# Detector v0 — evaluation results

Threshold calibrated for 10% FPR on `no_subhalo` (n=1000): **0.9774**  ·  ROC AUC (in-distribution, all masses) = **0.717**


## RQ4 — completeness vs. concentration (paired lenses, same mass & position)

| mass bin (log10 Msun) | n | completeness c=60 | completeness c=15 |
|---|---|---|---|
| 8.0-8.5 | 150 | 12.0% | 17.3% |
| 8.5-9.0 | 156 | 14.7% | 13.5% |
| 9.0-9.5 | 136 | 26.5% | 6.6% |
| 9.5-10.0 | 132 | 60.6% | 17.4% |
| 10.0-10.5 | 178 | 78.1% | 25.8% |
| 10.5-11.0 | 153 | 86.3% | 47.7% |

**Paired flip test** (905 identical lenses, only concentration changed): detected at c=60: 428, at c=15: 198. 277 lenses flipped from detected->missed when concentration dropped; 47 flipped the other way.

## RQ2 — confounder-induced false-positive rate (zero subhalos present)

| condition | FPR |
|---|---|
| no_subhalo (calibration set) | 10.0% |
| multipole m=4, a=0.01*thetaE | 7.8% |
| multipole m=4, a=0.03*thetaE | 11.2% |