# U-Net aggregate over 2 runs: detector_30k_c15wide_seed0, detector_30k_c15wide_seed1

AUC: 0.480 ± 0.000

| mass bin | c=60 | c=15 |
|---|---|---|
| 8.0-8.5 | 8.3% ± 3.3% | 8.3% ± 4.2% |
| 8.5-9.0 | 9.9% ± 0.5% | 10.3% ± 0.9% |
| 9.0-9.5 | 6.2% ± 3.6% | 6.6% ± 3.1% |
| 9.5-10.0 | 11.7% ± 0.5% | 11.7% ± 0.5% |
| 10.0-10.5 | 9.3% ± 2.0% | 9.8% ± 2.8% |
| 10.5-11.0 | 4.6% ± 0.9% | 5.2% ± 0.9% |

| confounder | FPR |
|---|---|
| no_subhalo (calibration set) | 10.0% ± 0.0% |
| multipole m=4, a=0.01*thetaE | 12.1% ± 1.7% |
| multipole m=4, a=0.03*thetaE | 11.5% ± 1.8% |

paired flip c60→missed at c15: 1.5 ± 0.7 vs reverse 4.5 ± 2.1 (ratio 0.3:1)