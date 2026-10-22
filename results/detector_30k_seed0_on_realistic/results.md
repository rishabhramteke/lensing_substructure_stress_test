# Detector v0 — evaluation results

Threshold calibrated for 10% FPR on `no_subhalo` (n=1000): **0.9990**  ·  ROC AUC (in-distribution, all masses) = **0.495**


## RQ4 — completeness vs. concentration (paired lenses, same mass & position)

| mass bin (log10 Msun) | n | completeness c=60 | completeness c=15 |
|---|---|---|---|
| 8.0-8.5 | 144 | 11.8% | 11.1% |
| 8.5-9.0 | 128 | 9.4% | 8.6% |
| 9.0-9.5 | 162 | 11.1% | 11.1% |
| 9.5-10.0 | 149 | 9.4% | 9.4% |
| 10.0-10.5 | 152 | 7.9% | 6.6% |
| 10.5-11.0 | 144 | 9.0% | 6.2% |

**Paired flip test** (879 identical lenses, only concentration changed): detected at c=60: 86, at c=15: 78. 55 lenses flipped from detected->missed when concentration dropped; 47 flipped the other way.

## RQ2 — confounder-induced false-positive rate (zero subhalos present)

| condition | FPR |
|---|---|
| no_subhalo (calibration set) | 10.0% |
| multipole m=4, a=0.01*thetaE | 9.9% |
| multipole m=4, a=0.03*thetaE | 9.9% |