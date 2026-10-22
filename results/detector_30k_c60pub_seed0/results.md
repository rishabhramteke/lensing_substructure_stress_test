# Detector v0 — evaluation results

Threshold calibrated for 10% FPR on `no_subhalo` (n=1000): **0.9904**  ·  ROC AUC (in-distribution, all masses) = **0.757**


## RQ4 — completeness vs. concentration (paired lenses, same mass & position)

| mass bin (log10 Msun) | n | completeness c=60 | completeness c=15 |
|---|---|---|---|
| 8.0-8.5 | 150 | 9.3% | 10.0% |
| 8.5-9.0 | 156 | 17.3% | 6.4% |
| 9.0-9.5 | 136 | 38.2% | 10.3% |
| 9.5-10.0 | 132 | 74.2% | 23.5% |
| 10.0-10.5 | 178 | 84.3% | 41.0% |
| 10.5-11.0 | 153 | 91.5% | 58.8% |

**Paired flip test** (905 identical lenses, only concentration changed): detected at c=60: 481, at c=15: 233. 280 lenses flipped from detected->missed when concentration dropped; 32 flipped the other way.

## RQ2 — confounder-induced false-positive rate (zero subhalos present)

| condition | FPR |
|---|---|
| no_subhalo (calibration set) | 10.0% |
| multipole m=4, a=0.01*thetaE | 10.1% |
| multipole m=4, a=0.03*thetaE | 15.4% |