# Detector v0 — evaluation results

Threshold calibrated for 10% FPR on `no_subhalo` (n=1000): **0.9537**  ·  ROC AUC (in-distribution, all masses) = **0.556**


## RQ4 — completeness vs. concentration (paired lenses, same mass & position)

| mass bin (log10 Msun) | n | completeness c=60 | completeness c=15 |
|---|---|---|---|
| 8.0-8.5 | 150 | 8.7% | 11.3% |
| 8.5-9.0 | 156 | 7.1% | 6.4% |
| 9.0-9.5 | 136 | 10.3% | 9.6% |
| 9.5-10.0 | 132 | 18.2% | 9.1% |
| 10.0-10.5 | 178 | 29.8% | 11.8% |
| 10.5-11.0 | 153 | 53.6% | 17.0% |

**Paired flip test** (905 identical lenses, only concentration changed): detected at c=60: 197, at c=15: 99. 147 lenses flipped from detected->missed when concentration dropped; 49 flipped the other way.

## RQ2 — confounder-induced false-positive rate (zero subhalos present)

| condition | FPR |
|---|---|
| no_subhalo (calibration set) | 10.0% |
| multipole m=4, a=0.03*thetaE | 10.9% |