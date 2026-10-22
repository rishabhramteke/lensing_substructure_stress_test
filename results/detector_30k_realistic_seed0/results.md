# Detector v0 — evaluation results

Threshold calibrated for 10% FPR on `no_subhalo` (n=1000): **0.9331**  ·  ROC AUC (in-distribution, all masses) = **0.515**


## RQ4 — completeness vs. concentration (paired lenses, same mass & position)

| mass bin (log10 Msun) | n | completeness c=60 | completeness c=15 |
|---|---|---|---|
| 8.0-8.5 | 144 | 7.6% | 7.6% |
| 8.5-9.0 | 128 | 10.2% | 9.4% |
| 9.0-9.5 | 162 | 9.9% | 9.9% |
| 9.5-10.0 | 149 | 11.4% | 12.1% |
| 10.0-10.5 | 152 | 12.5% | 12.5% |
| 10.5-11.0 | 144 | 16.7% | 9.7% |

**Paired flip test** (879 identical lenses, only concentration changed): detected at c=60: 100, at c=15: 90. 26 lenses flipped from detected->missed when concentration dropped; 16 flipped the other way.

## RQ2 — confounder-induced false-positive rate (zero subhalos present)

| condition | FPR |
|---|---|
| no_subhalo (calibration set) | 10.0% |
| multipole m=4, a=0.01*thetaE | 11.4% |
| multipole m=4, a=0.03*thetaE | 11.4% |