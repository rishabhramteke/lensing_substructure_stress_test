# Detector v0 — evaluation results

Threshold calibrated for 10% FPR on `no_subhalo` (n=1000): **0.9990**  ·  ROC AUC (in-distribution, all masses) = **0.620**


## RQ4 — completeness vs. concentration (paired lenses, same mass & position)

| mass bin (log10 Msun) | n | completeness c=60 | completeness c=15 |
|---|---|---|---|
| 8.0-8.5 | 150 | 7.3% | 9.3% |
| 8.5-9.0 | 156 | 7.7% | 7.7% |
| 9.0-9.5 | 136 | 8.1% | 8.8% |
| 9.5-10.0 | 132 | 22.0% | 9.8% |
| 10.0-10.5 | 178 | 42.7% | 6.7% |
| 10.5-11.0 | 153 | 66.0% | 15.0% |

**Paired flip test** (905 identical lenses, only concentration changed): detected at c=60: 240, at c=15: 86. 170 lenses flipped from detected->missed when concentration dropped; 16 flipped the other way.

## RQ2 — confounder-induced false-positive rate (zero subhalos present)

| condition | FPR |
|---|---|
| no_subhalo (calibration set) | 10.0% |
| multipole m=4, a=0.03*thetaE | 11.4% |