# Detector v0 — evaluation results

Threshold calibrated for 10% FPR on `no_subhalo` (n=1000): **0.8686**  ·  ROC AUC (in-distribution, all masses) = **0.478**


## RQ4 — completeness vs. concentration (paired lenses, same mass & position)

| mass bin (log10 Msun) | n | completeness c=60 | completeness c=15 |
|---|---|---|---|
| 8.0-8.5 | 154 | 8.4% | 11.7% |
| 8.5-9.0 | 137 | 9.5% | 8.8% |
| 9.0-9.5 | 125 | 8.8% | 6.4% |
| 9.5-10.0 | 147 | 4.8% | 5.4% |
| 10.0-10.5 | 165 | 10.9% | 10.3% |
| 10.5-11.0 | 170 | 7.1% | 8.8% |

**Paired flip test** (898 identical lenses, only concentration changed): detected at c=60: 74, at c=15: 78. 15 lenses flipped from detected->missed when concentration dropped; 19 flipped the other way.

## RQ2 — confounder-induced false-positive rate (zero subhalos present)

| condition | FPR |
|---|---|
| no_subhalo (calibration set) | 10.0% |
| multipole m=4, a=0.01*thetaE | 9.3% |
| multipole m=4, a=0.03*thetaE | 8.4% |