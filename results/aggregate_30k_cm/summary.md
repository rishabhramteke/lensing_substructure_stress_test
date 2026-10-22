# U-Net aggregate over 2 runs: detector_30k_cm_seed0, detector_30k_cm_seed1

AUC: 0.477 ± 0.004

| mass bin | c=60 | c=15 |
|---|---|---|
| 8.0-8.5 | 12.0% ± 0.9% | 12.3% ± 1.4% |
| 8.5-9.0 | 6.7% ± 2.3% | 7.4% ± 2.3% |
| 9.0-9.5 | 9.9% ± 0.5% | 9.2% ± 0.5% |
| 9.5-10.0 | 11.7% ± 1.6% | 12.1% ± 2.1% |
| 10.0-10.5 | 9.3% ± 0.4% | 9.3% ± 0.4% |
| 10.5-11.0 | 3.9% ± 0.9% | 6.2% ± 0.5% |

| confounder | FPR |
|---|---|
| no_subhalo (calibration set) | 10.0% ± 0.0% |
| multipole m=4, a=0.01*thetaE | 10.8% ± 0.8% |
| multipole m=4, a=0.03*thetaE | 10.5% ± 0.6% |

paired flip c60→missed at c15: 2.0 ± 0.0 vs reverse 6.5 ± 0.7 (ratio 0.3:1)