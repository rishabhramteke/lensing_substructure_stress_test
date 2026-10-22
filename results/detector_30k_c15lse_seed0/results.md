# Detector v0 — evaluation results

Threshold calibrated for 10% FPR on `no_subhalo` (n=1000): **0.6599**  ·  ROC AUC (in-distribution, all masses) = **0.502**


## RQ4 — completeness vs. concentration (paired lenses, same mass & position)

| mass bin (log10 Msun) | n | completeness c=60 | completeness c=15 |
|---|---|---|---|
| 8.0-8.5 | 150 | 10.7% | 10.7% |
| 8.5-9.0 | 156 | 7.7% | 7.7% |
| 9.0-9.5 | 136 | 8.8% | 8.8% |
| 9.5-10.0 | 132 | 9.1% | 8.3% |
| 10.0-10.5 | 178 | 6.2% | 6.2% |
| 10.5-11.0 | 153 | 11.8% | 8.5% |

**Paired flip test** (905 identical lenses, only concentration changed): detected at c=60: 81, at c=15: 75. 8 lenses flipped from detected->missed when concentration dropped; 2 flipped the other way.

## RQ2 — confounder-induced false-positive rate (zero subhalos present)

| condition | FPR |
|---|---|
| no_subhalo (calibration set) | 10.0% |
| multipole m=4, a=0.01*thetaE | 8.9% |
| multipole m=4, a=0.03*thetaE | 9.0% |