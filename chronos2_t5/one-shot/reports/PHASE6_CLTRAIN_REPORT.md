# Phase 6: train-time cross-learning Chronos-2 (C2-only, not head-to-head)

Group attention active during TRAINING (best config lr0.001/r16/ctx512, 1000 steps, CL val), CL eval, over 25 datasets.

## Aggregate relative score (gmean of model / Seasonal-Naive, lower = better)

| line | MASE | WQL |
| --- | --- | --- |
| C2-CLtrain (Phase 6) | 0.802 | 0.609 |
| C2-uni (Phase 4) | 0.808 | 0.628 |
| C2-CL eval-only (Phase 4b) | 0.809 | 0.614 |
| T5-uni (Phase 4) | 0.757 | 0.596 |

## Per-dataset (C2-CLtrain)

| dataset | MASE | WQL | n_series |
| --- | --- | --- | --- |
| monash_australian_electricity | 0.5871 | 0.0273 | 5 |
| monash_cif_2016 | 0.8305 | 0.0080 | 72 |
| monash_car_parts | 0.8476 | 0.9818 | 1000 |
| monash_covid_deaths | 28.5008 | 0.0286 | 266 |
| dominick | 0.9473 | 0.3379 | 1000 |
| ercot | 1.4023 | 0.0460 | 8 |
| exchange_rate | 1.7385 | 0.0121 | 8 |
| monash_fred_md | 0.8046 | 0.0412 | 107 |
| monash_hospital | 0.7738 | 0.0524 | 767 |
| monash_m1_monthly | 1.0098 | 0.1654 | 617 |
| monash_m1_quarterly | 1.8949 | 0.0895 | 203 |
| monash_m1_yearly | 3.5248 | 0.1048 | 181 |
| monash_m3_monthly | 0.8424 | 0.0884 | 1000 |
| monash_m3_quarterly | 1.1365 | 0.0685 | 756 |
| monash_m3_yearly | 3.4267 | 0.1587 | 645 |
| m4_quarterly | 1.2811 | 0.0764 | 1000 |
| m4_yearly | 3.1695 | 0.1150 | 1000 |
| m5 | 0.9324 | 0.5462 | 1000 |
| nn5 | 0.5600 | 0.1462 | 111 |
| monash_nn5_weekly | 0.8569 | 0.0798 | 111 |
| monash_tourism_monthly | 1.5257 | 0.0987 | 366 |
| monash_tourism_quarterly | 1.5513 | 0.0727 | 427 |
| monash_tourism_yearly | 3.6960 | 0.1411 | 518 |
| monash_traffic | 0.8307 | 0.2440 | 862 |
| monash_weather | 0.7766 | 0.1302 | 1000 |