# Detector v0 — evaluation results

Threshold calibrated for 10% FPR on `no_subhalo` (n=1000): **0.9879**  ·  ROC AUC (in-distribution, all masses) = **0.725**


## RQ4 — completeness vs. concentration (paired lenses, same mass & position)

| mass bin (log10 Msun) | n | completeness c=60 | completeness c=15 |
|---|---|---|---|
| 8.0-8.5 | 150 | 8.0% | 8.0% |
| 8.5-9.0 | 156 | 16.7% | 13.5% |
| 9.0-9.5 | 136 | 38.2% | 10.3% |
| 9.5-10.0 | 132 | 71.2% | 20.5% |
| 10.0-10.5 | 178 | 79.8% | 28.7% |
| 10.5-11.0 | 153 | 89.5% | 52.9% |

**Paired flip test** (905 identical lenses, only concentration changed): detected at c=60: 463, at c=15: 206. 294 lenses flipped from detected->missed when concentration dropped; 37 flipped the other way.

## RQ2 — confounder-induced false-positive rate (zero subhalos present)

| condition | FPR |
|---|---|
| no_subhalo (calibration set) | 10.0% |
| multipole m=4, a=0.01*thetaE | 9.8% |
| multipole m=4, a=0.03*thetaE | 11.1% |