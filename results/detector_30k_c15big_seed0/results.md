# Detector v0 — evaluation results

Threshold calibrated for 10% FPR on `no_subhalo` (n=1000): **0.9052**  ·  ROC AUC (in-distribution, all masses) = **0.559**


## RQ4 — completeness vs. concentration (paired lenses, same mass & position)

| mass bin (log10 Msun) | n | completeness c=60 | completeness c=15 |
|---|---|---|---|
| 8.0-8.5 | 150 | 12.7% | 10.7% |
| 8.5-9.0 | 156 | 6.4% | 7.7% |
| 9.0-9.5 | 136 | 8.1% | 8.1% |
| 9.5-10.0 | 132 | 21.2% | 17.4% |
| 10.0-10.5 | 178 | 21.9% | 12.9% |
| 10.5-11.0 | 153 | 41.2% | 11.1% |

**Paired flip test** (905 identical lenses, only concentration changed): detected at c=60: 170, at c=15: 102. 78 lenses flipped from detected->missed when concentration dropped; 10 flipped the other way.

## RQ2 — confounder-induced false-positive rate (zero subhalos present)

| condition | FPR |
|---|---|
| no_subhalo (calibration set) | 10.0% |
| multipole m=4, a=0.01*thetaE | 10.8% |
| multipole m=4, a=0.03*thetaE | 9.9% |