# One-shot (fine-tuned) Zero-shot-benchmark Reproduction

chronos-t5-small fine-tuned per dataset (lr 1e-3 -> 0 over 1000 steps, explicit PyTorch loop), evaluated via the official gluonts pipeline, cap=1000, on 25 Benchmark II datasets.

## Aggregated relative score (gmean of model / Seasonal-Naive)

| scenario | WQL | MASE |
| --- | --- | --- |
| one-shot (ours) | 0.615 | 0.766 |
| zero-shot (ours) | 0.687 | 0.852 |
| one-shot (paper, Fig. 6) | 0.597 | 0.760 |

## Per-dataset (one-shot, ours)

| dataset | MASE | WQL | n_series |
| --- | --- | --- | --- |
| monash_australian_electricity | 0.8713 | 0.0398 | 5 |
| monash_cif_2016 | 0.9850 | 0.0109 | 72 |
| monash_car_parts | 0.8068 | 0.9255 | 1000 |
| monash_covid_deaths | 33.6336 | 0.0303 | 266 |
| dominick | 0.9560 | 0.3586 | 1000 |
| ercot | 0.5972 | 0.0193 | 8 |
| exchange_rate | 1.5661 | 0.0110 | 8 |
| monash_fred_md | 0.5336 | 0.0283 | 107 |
| monash_hospital | 0.7950 | 0.0595 | 767 |
| monash_m1_monthly | 1.0458 | 0.1675 | 617 |
| monash_m1_quarterly | 1.7376 | 0.0922 | 203 |
| monash_m1_yearly | 3.4777 | 0.1233 | 181 |
| monash_m3_monthly | 0.8423 | 0.0958 | 1000 |
| monash_m3_quarterly | 1.2007 | 0.0763 | 756 |
| monash_m3_yearly | 2.8146 | 0.1467 | 645 |
| m4_quarterly | 1.1103 | 0.0765 | 1000 |
| m4_yearly | 2.9696 | 0.1142 | 1000 |
| m5 | 0.9622 | 0.6061 | 1000 |
| nn5 | 0.5675 | 0.1601 | 111 |
| monash_nn5_weekly | 0.9462 | 0.0951 | 111 |
| monash_tourism_monthly | 1.4346 | 0.0821 | 366 |
| monash_tourism_quarterly | 1.5632 | 0.0758 | 427 |
| monash_tourism_yearly | 3.1575 | 0.1397 | 518 |
| monash_traffic | 0.7976 | 0.2560 | 862 |
| monash_weather | 0.8103 | 0.1464 | 1000 |

## Inference efficiency (ours, fine-tuned chronos-t5-small, GPU (bf16))

Total forecast wall-time 248.1s over 25 datasets; peak GPU memory 5581 MB; mean 51.8 ms/series.

| dataset | n_series | latency_s | ms/series | peak_mem_MB |
| --- | --- | --- | --- | --- |
| monash_australian_electricity | 5 | 2.83 | 565.7 | 892 |
| monash_cif_2016 | 72 | 1.51 | 21.0 | 1235 |
| monash_car_parts | 1000 | 15.56 | 15.6 | 702 |
| monash_covid_deaths | 266 | 11.53 | 43.4 | 2091 |
| dominick | 1000 | 13.11 | 13.1 | 4154 |
| ercot | 8 | 1.11 | 138.8 | 1577 |
| exchange_rate | 8 | 1.28 | 160.3 | 1619 |
| monash_fred_md | 107 | 2.04 | 19.1 | 5456 |
| monash_hospital | 767 | 11.88 | 15.5 | 1227 |
| monash_m1_monthly | 617 | 14.62 | 23.7 | 1834 |
| monash_m1_quarterly | 203 | 2.38 | 11.7 | 1585 |
| monash_m1_yearly | 181 | 1.47 | 8.1 | 1142 |
| monash_m3_monthly | 1000 | 23.09 | 23.1 | 1912 |
| monash_m3_quarterly | 756 | 8.61 | 11.4 | 787 |
| monash_m3_yearly | 645 | 6.14 | 9.5 | 616 |
| m4_quarterly | 1000 | 7.22 | 7.2 | 5246 |
| m4_yearly | 1000 | 5.71 | 5.7 | 3420 |
| m5 | 1000 | 30.99 | 31.0 | 5330 |
| nn5 | 111 | 5.79 | 52.1 | 5372 |
| monash_nn5_weekly | 111 | 0.66 | 5.9 | 1408 |
| monash_tourism_monthly | 366 | 10.94 | 29.9 | 3461 |
| monash_tourism_quarterly | 427 | 4.80 | 11.2 | 1659 |
| monash_tourism_yearly | 518 | 3.09 | 6.0 | 963 |
| monash_traffic | 862 | 26.26 | 30.5 | 5581 |
| monash_weather | 1000 | 35.54 | 35.5 | 5161 |