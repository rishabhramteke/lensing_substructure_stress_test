# Detector v0 — evaluation results

Threshold calibrated for 10% FPR on `no_subhalo` (n=1000): **0.9878**  ·  ROC AUC (in-distribution, all masses) = **0.663**


## RQ4 — completeness vs. concentration (paired lenses, same mass & position)

| mass bin (log10 Msun) | n | completeness c=60 | completeness c=15 |
|---|---|---|---|
| 8.0-8.5 | 150 | 11.3% | 5.3% |
| 8.5-9.0 | 156 | 6.4% | 9.6% |
| 9.0-9.5 | 136 | 18.4% | 7.4% |
| 9.5-10.0 | 132 | 45.5% | 17.4% |
| 10.0-10.5 | 178 | 69.1% | 12.9% |
| 10.5-11.0 | 153 | 77.8% | 24.2% |

**Paired flip test** (905 identical lenses, only concentration changed): detected at c=60: 354, at c=15: 116. 274 lenses flipped from detected->missed when concentration dropped; 36 flipped the other way.

## RQ2 — confounder-induced false-positive rate (zero subhalos present)

| condition | FPR |
|---|---|
| no_subhalo (calibration set) | 10.0% |
| multipole m=4, a=0.01*thetaE | 9.5% |
| multipole m=4, a=0.03*thetaE | 11.0% |