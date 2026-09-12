# Detector v0 — evaluation results

Threshold calibrated for 10% FPR on `no_subhalo` (n=1000): **0.9864**  ·  ROC AUC (in-distribution, all masses) = **0.662**


## RQ4 — completeness vs. concentration (paired lenses, same mass & position)

| mass bin (log10 Msun) | n | completeness c=60 | completeness c=15 |
|---|---|---|---|
| 8.0-8.5 | 150 | 12.7% | 11.3% |
| 8.5-9.0 | 156 | 9.0% | 8.3% |
| 9.0-9.5 | 136 | 20.6% | 11.8% |
| 9.5-10.0 | 132 | 52.3% | 15.2% |
| 10.0-10.5 | 178 | 70.2% | 12.4% |
| 10.5-11.0 | 153 | 79.7% | 29.4% |

**Paired flip test** (905 identical lenses, only concentration changed): detected at c=60: 377, at c=15: 133. 290 lenses flipped from detected->missed when concentration dropped; 46 flipped the other way.

## RQ2 — confounder-induced false-positive rate (zero subhalos present)

| condition | FPR |
|---|---|
| no_subhalo (calibration set) | 10.0% |
| multipole m=4, a=0.01*thetaE | 10.1% |
| multipole m=4, a=0.03*thetaE | 10.4% |