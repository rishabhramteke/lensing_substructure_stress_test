# Detector v0 — evaluation results

Threshold calibrated for 10% FPR on `no_subhalo` (n=1000): **0.8387**  ·  ROC AUC (in-distribution, all masses) = **0.482**


## RQ4 — completeness vs. concentration (paired lenses, same mass & position)

| mass bin (log10 Msun) | n | completeness c=60 | completeness c=15 |
|---|---|---|---|
| 8.0-8.5 | 150 | 10.0% | 10.7% |
| 8.5-9.0 | 156 | 12.2% | 12.2% |
| 9.0-9.5 | 136 | 7.4% | 8.1% |
| 9.5-10.0 | 132 | 12.1% | 12.1% |
| 10.0-10.5 | 178 | 5.1% | 6.2% |
| 10.5-11.0 | 153 | 5.9% | 7.2% |

**Paired flip test** (905 identical lenses, only concentration changed): detected at c=60: 78, at c=15: 84. 0 lenses flipped from detected->missed when concentration dropped; 6 flipped the other way.

## RQ2 — confounder-induced false-positive rate (zero subhalos present)

| condition | FPR |
|---|---|
| no_subhalo (calibration set) | 10.0% |
| multipole m=4, a=0.01*thetaE | 13.0% |
| multipole m=4, a=0.03*thetaE | 11.3% |