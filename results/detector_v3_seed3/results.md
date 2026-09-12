# Detector v0 — evaluation results

Threshold calibrated for 10% FPR on `no_subhalo` (n=1000): **0.9773**  ·  ROC AUC (in-distribution, all masses) = **0.634**


## RQ4 — completeness vs. concentration (paired lenses, same mass & position)

| mass bin (log10 Msun) | n | completeness c=60 | completeness c=15 |
|---|---|---|---|
| 8.0-8.5 | 150 | 13.3% | 7.3% |
| 8.5-9.0 | 156 | 7.7% | 7.7% |
| 9.0-9.5 | 136 | 15.4% | 10.3% |
| 9.5-10.0 | 132 | 31.8% | 10.6% |
| 10.0-10.5 | 178 | 59.0% | 16.3% |
| 10.5-11.0 | 153 | 67.3% | 14.4% |

**Paired flip test** (905 identical lenses, only concentration changed): detected at c=60: 303, at c=15: 102. 224 lenses flipped from detected->missed when concentration dropped; 23 flipped the other way.

## RQ2 — confounder-induced false-positive rate (zero subhalos present)

| condition | FPR |
|---|---|
| no_subhalo (calibration set) | 10.0% |
| multipole m=4, a=0.01*thetaE | 10.7% |
| multipole m=4, a=0.03*thetaE | 10.7% |