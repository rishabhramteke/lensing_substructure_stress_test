# Detector v0 — evaluation results

Threshold calibrated for 10% FPR on `no_subhalo` (n=1000): **0.9894**  ·  ROC AUC (in-distribution, all masses) = **0.675**


## RQ4 — completeness vs. concentration (paired lenses, same mass & position)

| mass bin (log10 Msun) | n | completeness c=60 | completeness c=15 |
|---|---|---|---|
| 8.0-8.5 | 150 | 10.7% | 8.7% |
| 8.5-9.0 | 156 | 7.1% | 10.9% |
| 9.0-9.5 | 136 | 17.6% | 7.4% |
| 9.5-10.0 | 132 | 56.1% | 9.1% |
| 10.0-10.5 | 178 | 70.8% | 18.5% |
| 10.5-11.0 | 153 | 80.4% | 34.0% |

**Paired flip test** (905 identical lenses, only concentration changed): detected at c=60: 374, at c=15: 137. 270 lenses flipped from detected->missed when concentration dropped; 33 flipped the other way.

## RQ2 — confounder-induced false-positive rate (zero subhalos present)

| condition | FPR |
|---|---|
| no_subhalo (calibration set) | 10.0% |
| multipole m=4, a=0.01*thetaE | 7.4% |
| multipole m=4, a=0.03*thetaE | 10.5% |