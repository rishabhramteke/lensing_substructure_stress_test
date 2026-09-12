# Detector v0 — evaluation results

Threshold calibrated for 10% FPR on `no_subhalo` (n=1000): **0.9712**  ·  ROC AUC (in-distribution, all masses) = **0.627**


## RQ4 — completeness vs. concentration (paired lenses, same mass & position)

| mass bin (log10 Msun) | n | completeness c=60 | completeness c=15 |
|---|---|---|---|
| 8.0-8.5 | 150 | 18.0% | 13.3% |
| 8.5-9.0 | 156 | 7.7% | 9.0% |
| 9.0-9.5 | 136 | 19.9% | 18.4% |
| 9.5-10.0 | 132 | 37.1% | 11.4% |
| 10.0-10.5 | 178 | 56.7% | 18.5% |
| 10.5-11.0 | 153 | 64.7% | 17.0% |

**Paired flip test** (905 identical lenses, only concentration changed): detected at c=60: 315, at c=15: 133. 237 lenses flipped from detected->missed when concentration dropped; 55 flipped the other way.

## RQ2 — confounder-induced false-positive rate (zero subhalos present)

| condition | FPR |
|---|---|
| no_subhalo (calibration set) | 10.0% |
| multipole m=4, a=0.01*thetaE | 14.3% |
| multipole m=4, a=0.03*thetaE | 13.1% |