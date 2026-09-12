# Detector v0 — evaluation results

Threshold calibrated for 10% FPR on `no_subhalo` (n=1000): **0.8350**  ·  ROC AUC (in-distribution, all masses) = **0.487**


## RQ4 — completeness vs. concentration (paired lenses, same mass & position)

| mass bin (log10 Msun) | n | completeness c=60 | completeness c=15 |
|---|---|---|---|
| 8.0-8.5 | 150 | 9.3% | 10.0% |
| 8.5-9.0 | 156 | 5.8% | 6.4% |
| 9.0-9.5 | 136 | 11.0% | 11.0% |
| 9.5-10.0 | 132 | 9.8% | 10.6% |
| 10.0-10.5 | 178 | 10.7% | 10.7% |
| 10.5-11.0 | 153 | 6.5% | 7.2% |

**Paired flip test** (905 identical lenses, only concentration changed): detected at c=60: 80, at c=15: 84. 2 lenses flipped from detected->missed when concentration dropped; 6 flipped the other way.

## RQ2 — confounder-induced false-positive rate (zero subhalos present)

| condition | FPR |
|---|---|
| no_subhalo (calibration set) | 10.0% |
| multipole m=4, a=0.01*thetaE | 9.7% |
| multipole m=4, a=0.03*thetaE | 11.6% |