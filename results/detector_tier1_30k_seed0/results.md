# Detector v0 — evaluation results

Threshold calibrated for 10% FPR on `no_subhalo` (n=1000): **0.9806**  ·  ROC AUC (in-distribution, all masses) = **0.559**


## RQ4 — completeness vs. concentration (paired lenses, same mass & position)

| mass bin (log10 Msun) | n | completeness c=60 | completeness c=15 |
|---|---|---|---|
| 8.0-8.5 | 154 | 9.7% | 11.0% |
| 8.5-9.0 | 137 | 10.2% | 10.9% |
| 9.0-9.5 | 125 | 10.4% | 11.2% |
| 9.5-10.0 | 147 | 10.9% | 8.8% |
| 10.0-10.5 | 165 | 12.7% | 9.7% |
| 10.5-11.0 | 170 | 33.5% | 9.4% |

**Paired flip test** (898 identical lenses, only concentration changed): detected at c=60: 136, at c=15: 91. 84 lenses flipped from detected->missed when concentration dropped; 39 flipped the other way.

## RQ2 — confounder-induced false-positive rate (zero subhalos present)

| condition | FPR |
|---|---|
| no_subhalo (calibration set) | 10.0% |
| multipole m=4, a=0.01*thetaE | 8.5% |
| multipole m=4, a=0.03*thetaE | 10.7% |