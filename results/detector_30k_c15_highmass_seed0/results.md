# Detector v0 — evaluation results

Threshold calibrated for 10% FPR on `no_subhalo` (n=1000): **0.9914**  ·  ROC AUC (in-distribution, all masses) = **0.669**


## RQ4 — completeness vs. concentration (paired lenses, same mass & position)

| mass bin (log10 Msun) | n | completeness c=60 | completeness c=15 |
|---|---|---|---|
| 8.0-8.5 | 150 | 9.3% | 8.0% |
| 8.5-9.0 | 156 | 12.2% | 8.3% |
| 9.0-9.5 | 136 | 16.9% | 7.4% |
| 9.5-10.0 | 132 | 50.8% | 13.6% |
| 10.0-10.5 | 178 | 68.0% | 16.9% |
| 10.5-11.0 | 153 | 77.1% | 44.4% |

**Paired flip test** (905 identical lenses, only concentration changed): detected at c=60: 362, at c=15: 151. 227 lenses flipped from detected->missed when concentration dropped; 16 flipped the other way.

## RQ2 — confounder-induced false-positive rate (zero subhalos present)

| condition | FPR |
|---|---|
| no_subhalo (calibration set) | 10.0% |
| multipole m=4, a=0.01*thetaE | 11.6% |
| multipole m=4, a=0.03*thetaE | 9.0% |