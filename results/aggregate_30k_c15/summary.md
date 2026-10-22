# U-Net aggregate over 2 runs: detector_30k_c15_seed0, detector_30k_c15_seed1

AUC: 0.485 ± 0.001

| mass bin | c=60 | c=15 |
|---|---|---|
| 8.0-8.5 | 9.3% ± 0.9% | 9.0% ± 0.5% |
| 8.5-9.0 | 9.9% ± 0.5% | 10.6% ± 0.5% |
| 9.0-9.5 | 9.2% ± 0.5% | 9.2% ± 0.5% |
| 9.5-10.0 | 15.9% ± 4.3% | 16.3% ± 3.7% |
| 10.0-10.5 | 12.1% ± 2.0% | 12.4% ± 1.6% |
| 10.5-11.0 | 6.2% ± 0.5% | 6.5% ± 0.0% |

| confounder | FPR |
|---|---|
| no_subhalo (calibration set) | 10.0% ± 0.0% |
| multipole m=4, a=0.01*thetaE | 13.3% ± 1.7% |
| multipole m=4, a=0.03*thetaE | 12.5% ± 0.7% |

paired flip c60→missed at c15: 2.5 ± 0.7 vs reverse 4.5 ± 2.1 (ratio 0.6:1)