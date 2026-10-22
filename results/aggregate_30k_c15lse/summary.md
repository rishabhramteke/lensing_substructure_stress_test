# U-Net aggregate over 2 runs: detector_30k_c15lse_seed0, detector_30k_c15lse_seed1

AUC: 0.504 ± 0.003

| mass bin | c=60 | c=15 |
|---|---|---|
| 8.0-8.5 | 11.0% ± 0.5% | 11.0% ± 0.5% |
| 8.5-9.0 | 9.3% ± 2.3% | 9.6% ± 2.7% |
| 9.0-9.5 | 7.0% ± 2.6% | 6.6% ± 3.1% |
| 9.5-10.0 | 8.7% ± 0.5% | 8.3% ± 0.0% |
| 10.0-10.5 | 7.6% ± 2.0% | 7.6% ± 2.0% |
| 10.5-11.0 | 11.8% ± 0.0% | 10.1% ± 2.3% |

| confounder | FPR |
|---|---|
| no_subhalo (calibration set) | 10.0% ± 0.0% |
| multipole m=4, a=0.01*thetaE | 10.1% ± 1.7% |
| multipole m=4, a=0.03*thetaE | 9.8% ± 1.2% |

paired flip c60→missed at c15: 5.0 ± 4.2 vs reverse 2.0 ± 0.0 (ratio 2.5:1)