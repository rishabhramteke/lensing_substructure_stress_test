# U-Net aggregate over 2 runs: detector_30k_seed0, detector_30k_seed1

AUC: 0.666 ± 0.006

| mass bin | c=60 | c=15 |
|---|---|---|
| 8.0-8.5 | 11.7% ± 1.4% | 11.0% ± 0.5% |
| 8.5-9.0 | 9.9% ± 1.4% | 8.0% ± 0.5% |
| 9.0-9.5 | 20.6% ± 0.0% | 9.9% ± 2.6% |
| 9.5-10.0 | 50.8% ± 2.1% | 16.7% ± 2.1% |
| 10.0-10.5 | 69.7% ± 0.8% | 13.2% ± 1.2% |
| 10.5-11.0 | 79.1% ± 0.9% | 27.8% ± 2.3% |

| confounder | FPR |
|---|---|
| no_subhalo (calibration set) | 10.0% ± 0.0% |
| multipole m=4, a=0.01*thetaE | 9.9% ± 0.3% |
| multipole m=4, a=0.03*thetaE | 10.1% ± 0.5% |

paired flip c60→missed at c15: 284.5 ± 7.8 vs reverse 42.0 ± 5.7 (ratio 6.8:1)