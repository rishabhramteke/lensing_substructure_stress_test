# U-Net aggregate over 2 runs: detector_30k_realistic_seed0, detector_30k_realistic_seed1

AUC: 0.535 ± 0.029

| mass bin | c=60 | c=15 |
|---|---|---|
| 8.0-8.5 | 9.7% ± 2.9% | 8.3% ± 1.0% |
| 8.5-9.0 | 12.5% ± 3.3% | 9.8% ± 0.6% |
| 9.0-9.5 | 9.6% ± 0.4% | 9.3% ± 0.9% |
| 9.5-10.0 | 11.7% ± 0.5% | 11.1% ± 1.4% |
| 10.0-10.5 | 18.1% ± 7.9% | 11.2% ± 1.9% |
| 10.5-11.0 | 28.1% ± 16.2% | 11.1% ± 2.0% |

| confounder | FPR |
|---|---|
| no_subhalo (calibration set) | 10.0% ± 0.0% |
| multipole m=4, a=0.01*thetaE | 10.5% ± 1.3% |
| multipole m=4, a=0.03*thetaE | 10.5% ± 1.3% |

paired flip c60→missed at c15: 61.0 ± 49.5 vs reverse 19.0 ± 4.2 (ratio 3.2:1)