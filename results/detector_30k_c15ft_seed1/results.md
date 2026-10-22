# Detector v0 — evaluation results

Threshold calibrated for 10% FPR on `no_subhalo` (n=1000): **0.9625**  ·  ROC AUC (in-distribution, all masses) = **0.649**


## RQ4 — completeness vs. concentration (paired lenses, same mass & position)

| mass bin (log10 Msun) | n | completeness c=60 | completeness c=15 |
|---|---|---|---|
| 8.0-8.5 | 150 | 10.7% | 8.7% |
| 8.5-9.0 | 156 | 9.0% | 6.4% |
| 9.0-9.5 | 136 | 19.9% | 7.4% |
| 9.5-10.0 | 132 | 47.0% | 16.7% |
| 10.0-10.5 | 178 | 69.1% | 12.4% |
| 10.5-11.0 | 153 | 77.8% | 28.8% |

**Paired flip test** (905 identical lenses, only concentration changed): detected at c=60: 361, at c=15: 121. 268 lenses flipped from detected->missed when concentration dropped; 28 flipped the other way.

## RQ2 — confounder-induced false-positive rate (zero subhalos present)

| condition | FPR |
|---|---|
| no_subhalo (calibration set) | 10.0% |
| multipole m=4, a=0.01*thetaE | 9.3% |
| multipole m=4, a=0.03*thetaE | 10.6% |