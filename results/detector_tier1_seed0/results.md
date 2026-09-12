# Detector v0 — evaluation results

Threshold calibrated for 10% FPR on `no_subhalo` (n=1000): **0.8346**  ·  ROC AUC (in-distribution, all masses) = **0.491**


## RQ4 — completeness vs. concentration (paired lenses, same mass & position)

| mass bin (log10 Msun) | n | completeness c=60 | completeness c=15 |
|---|---|---|---|
| 8.0-8.5 | 154 | 14.9% | 12.3% |
| 8.5-9.0 | 137 | 12.4% | 13.1% |
| 9.0-9.5 | 125 | 6.4% | 8.0% |
| 9.5-10.0 | 147 | 8.8% | 7.5% |
| 10.0-10.5 | 165 | 8.5% | 9.7% |
| 10.5-11.0 | 170 | 8.2% | 10.0% |

**Paired flip test** (898 identical lenses, only concentration changed): detected at c=60: 89, at c=15: 91. 10 lenses flipped from detected->missed when concentration dropped; 12 flipped the other way.

## RQ2 — confounder-induced false-positive rate (zero subhalos present)

| condition | FPR |
|---|---|
| no_subhalo (calibration set) | 10.0% |
| multipole m=4, a=0.01*thetaE | 10.1% |
| multipole m=4, a=0.03*thetaE | 9.1% |