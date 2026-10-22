# U-Net aggregate over 2 runs: detector_30k_c15img_seed0, detector_30k_c15img_seed1

AUC: 0.519 ± 0.000

| mass bin | c=60 | c=15 |
|---|---|---|
| 8.0-8.5 | 11.3% ± 0.9% | 11.0% ± 1.4% |
| 8.5-9.0 | 12.2% ± 5.4% | 11.5% ± 5.4% |
| 9.0-9.5 | 11.8% ± 3.1% | 12.5% ± 4.2% |
| 9.5-10.0 | 11.7% ± 1.6% | 9.5% ± 0.5% |
| 10.0-10.5 | 10.1% ± 4.0% | 8.4% ± 4.0% |
| 10.5-11.0 | 10.8% ± 1.4% | 11.1% ± 0.9% |

| confounder | FPR |
|---|---|
| no_subhalo (calibration set) | 10.0% ± 0.0% |
| multipole m=4, a=0.01*thetaE | 10.3% ± 0.2% |
| multipole m=4, a=0.03*thetaE | 10.8% ± 2.1% |

paired flip c60→missed at c15: 27.0 ± 11.3 vs reverse 21.0 ± 11.3 (ratio 1.3:1)