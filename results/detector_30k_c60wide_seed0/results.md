# Detector v0 — evaluation results

Threshold calibrated for 10% FPR on `no_subhalo` (n=1000): **0.9377**  ·  ROC AUC (in-distribution, all masses) = **0.678**


## RQ4 — completeness vs. concentration (paired lenses, same mass & position)

| mass bin (log10 Msun) | n | completeness c=60 | completeness c=15 |
|---|---|---|---|
| 8.0-8.5 | 150 | 12.7% | 11.3% |
| 8.5-9.0 | 156 | 11.5% | 10.9% |
| 9.0-9.5 | 136 | 14.0% | 7.4% |
| 9.5-10.0 | 132 | 43.9% | 7.6% |
| 10.0-10.5 | 178 | 68.5% | 13.5% |
| 10.5-11.0 | 153 | 79.1% | 24.8% |

**Paired flip test** (905 identical lenses, only concentration changed): detected at c=60: 357, at c=15: 116. 277 lenses flipped from detected->missed when concentration dropped; 36 flipped the other way.

## RQ2 — confounder-induced false-positive rate (zero subhalos present)

| condition | FPR |
|---|---|
| no_subhalo (calibration set) | 10.0% |
| multipole m=4, a=0.01*thetaE | 9.9% |
| multipole m=4, a=0.03*thetaE | 9.8% |