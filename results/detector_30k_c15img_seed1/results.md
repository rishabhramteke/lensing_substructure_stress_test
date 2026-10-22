# Detector v0 — evaluation results

Threshold calibrated for 10% FPR on `no_subhalo` (n=1000): **0.5010**  ·  ROC AUC (in-distribution, all masses) = **0.519**


## RQ4 — completeness vs. concentration (paired lenses, same mass & position)

| mass bin (log10 Msun) | n | completeness c=60 | completeness c=15 |
|---|---|---|---|
| 8.0-8.5 | 150 | 12.0% | 10.0% |
| 8.5-9.0 | 156 | 16.0% | 15.4% |
| 9.0-9.5 | 136 | 14.0% | 15.4% |
| 9.5-10.0 | 132 | 10.6% | 9.8% |
| 10.0-10.5 | 178 | 12.9% | 11.2% |
| 10.5-11.0 | 153 | 11.8% | 11.8% |

**Paired flip test** (905 identical lenses, only concentration changed): detected at c=60: 117, at c=15: 111. 35 lenses flipped from detected->missed when concentration dropped; 29 flipped the other way.

## RQ2 — confounder-induced false-positive rate (zero subhalos present)

| condition | FPR |
|---|---|
| no_subhalo (calibration set) | 10.0% |
| multipole m=4, a=0.01*thetaE | 10.5% |
| multipole m=4, a=0.03*thetaE | 12.2% |