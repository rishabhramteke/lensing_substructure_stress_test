# Detector v0 — evaluation results

Threshold calibrated for 10% FPR on `no_subhalo` (n=1000): **0.9827**  ·  ROC AUC (in-distribution, all masses) = **0.670**


## RQ4 — completeness vs. concentration (paired lenses, same mass & position)

| mass bin (log10 Msun) | n | completeness c=60 | completeness c=15 |
|---|---|---|---|
| 8.0-8.5 | 150 | 10.7% | 10.7% |
| 8.5-9.0 | 156 | 10.9% | 7.7% |
| 9.0-9.5 | 136 | 20.6% | 8.1% |
| 9.5-10.0 | 132 | 49.2% | 18.2% |
| 10.0-10.5 | 178 | 69.1% | 14.0% |
| 10.5-11.0 | 153 | 78.4% | 26.1% |

**Paired flip test** (905 identical lenses, only concentration changed): detected at c=60: 369, at c=15: 128. 279 lenses flipped from detected->missed when concentration dropped; 38 flipped the other way.

## RQ2 — confounder-induced false-positive rate (zero subhalos present)

| condition | FPR |
|---|---|
| no_subhalo (calibration set) | 10.0% |
| multipole m=4, a=0.01*thetaE | 9.7% |
| multipole m=4, a=0.03*thetaE | 9.7% |