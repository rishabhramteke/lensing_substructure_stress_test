# U-Net aggregate over 4 runs: detector_30k_seed0, detector_30k_seed1, detector_30k_seed2, detector_30k_seed3

AUC: 0.668 ± 0.006

| mass bin | c=60 | c=15 |
|---|---|---|
| 8.0-8.5 | 11.3% ± 0.9% | 9.0% ± 2.7% |
| 8.5-9.0 | 8.3% ± 2.0% | 9.1% ± 1.4% |
| 9.0-9.5 | 19.3% ± 1.5% | 8.6% ± 2.1% |
| 9.5-10.0 | 50.8% ± 4.5% | 15.0% ± 4.1% |
| 10.0-10.5 | 69.8% ± 0.8% | 14.5% ± 2.8% |
| 10.5-11.0 | 79.1% ± 1.2% | 28.4% ± 4.3% |

| confounder | FPR |
|---|---|
| no_subhalo (calibration set) | 10.0% ± 0.0% |
| multipole m=4, a=0.01*thetaE | 9.2% ± 1.2% |
| multipole m=4, a=0.03*thetaE | 10.4% ± 0.5% |

paired flip c60→missed at c15: 278.2 ± 8.7 vs reverse 38.2 ± 5.6 (ratio 7.3:1)