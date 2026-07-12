# Strict (official-method) Zero-shot Reproduction

chronos-t5-small via the official gluonts metric pipeline (MASE + MeanWeightedSumQuantileLoss, gluonts split), cap=1000/dataset, on 25 Benchmark II datasets.

## Aggregated relative score (gmean of model / Seasonal-Naive, paper's method)

| metric | ours | paper (Chronos-T5 Small) |
| --- | --- | --- |
| WQL  | 0.687 | 0.675 |
| MASE | 0.852 | 0.839 |

## Per-dataset: ours vs paper (official Chronos-T5 Small)

| dataset | WQL_ours | WQL_paper | MASE_ours | MASE_paper |
| --- | --- | --- | --- | --- |
| monash_australian_electricity | 0.0724 | 0.0695 | 1.1849 | 1.2241 |
| monash_cif_2016 | 0.0115 | 0.0146 | 1.0017 | 1.0188 |
| monash_car_parts | 1.0505 | 1.0315 | 0.8768 | 0.8918 |
| monash_covid_deaths | 0.0673 | 0.0634 | 42.7023 | 42.2990 |
| dominick | 0.3686 | 0.3368 | 0.9398 | 0.8109 |
| ercot | 0.0170 | 0.0155 | 0.5922 | 0.5649 |
| exchange_rate | 0.0138 | 0.0145 | 1.9954 | 1.8143 |
| monash_fred_md | 0.0149 | 0.0149 | 0.4533 | 0.4742 |
| monash_hospital | 0.0583 | 0.0570 | 0.8068 | 0.7098 |
| monash_m1_monthly | 0.1454 | 0.1380 | 1.1712 | 1.1723 |
| monash_m1_quarterly | 0.1138 | 0.1132 | 1.8246 | 1.8078 |
| monash_m1_yearly | 0.1806 | 0.1731 | 4.8971 | 4.7400 |
| monash_m3_monthly | 0.1006 | 0.0999 | 0.8772 | 0.8857 |
| monash_m3_quarterly | 0.0822 | 0.0809 | 1.2915 | 1.2789 |
| monash_m3_yearly | 0.1630 | 0.1574 | 3.4174 | 3.3825 |
| m4_quarterly | 0.0848 | 0.0838 | 1.2334 | 1.2415 |
| m4_yearly | 0.1404 | 0.1385 | 3.6549 | 3.7387 |
| m5 | 0.6108 | 0.5896 | 0.9617 | 0.9369 |
| nn5 | 0.1731 | 0.1677 | 0.6147 | 0.6131 |
| monash_nn5_weekly | 0.0947 | 0.0896 | 0.9625 | 0.9277 |
| monash_tourism_monthly | 0.1115 | 0.1094 | 1.9379 | 1.9251 |
| monash_tourism_quarterly | 0.0662 | 0.0686 | 1.7683 | 1.7623 |
| monash_tourism_yearly | 0.2114 | 0.1996 | 3.9751 | 3.9877 |
| monash_traffic | 0.2645 | 0.2571 | 0.8268 | 0.8204 |
| monash_weather | 0.1536 | 0.1480 | 0.8336 | 0.8551 |

## Inference efficiency (ours, chronos-t5-small, GPU (bf16))

Total forecast wall-time 202.4s over 25 datasets; peak GPU memory 5581 MB; mean 29.2 ms/series.

| dataset | n_series | latency_s | ms/series | peak_mem_MB |
| --- | --- | --- | --- | --- |
| monash_australian_electricity | 5 | 1.14 | 227.8 | 892 |
| monash_cif_2016 | 72 | 0.91 | 12.6 | 1235 |
| monash_car_parts | 1000 | 7.72 | 7.7 | 702 |
| monash_covid_deaths | 266 | 4.38 | 16.4 | 2091 |
| dominick | 1000 | 10.74 | 10.7 | 4154 |
| ercot | 8 | 0.74 | 92.9 | 1577 |
| exchange_rate | 8 | 0.54 | 67.2 | 1619 |
| monash_fred_md | 107 | 1.50 | 14.0 | 5456 |
| monash_hospital | 767 | 6.72 | 8.8 | 1227 |
| monash_m1_monthly | 617 | 13.70 | 22.2 | 1834 |
| monash_m1_quarterly | 203 | 2.10 | 10.4 | 1585 |
| monash_m1_yearly | 181 | 1.37 | 7.6 | 1142 |
| monash_m3_monthly | 1000 | 15.80 | 15.8 | 1912 |
| monash_m3_quarterly | 756 | 8.17 | 10.8 | 787 |
| monash_m3_yearly | 645 | 5.97 | 9.3 | 616 |
| m4_quarterly | 1000 | 8.69 | 8.7 | 5246 |
| m4_yearly | 1000 | 5.78 | 5.8 | 3420 |
| m5 | 1000 | 30.82 | 30.8 | 5330 |
| nn5 | 111 | 5.54 | 49.9 | 5372 |
| monash_nn5_weekly | 111 | 1.02 | 9.2 | 1408 |
| monash_tourism_monthly | 366 | 6.76 | 18.5 | 3461 |
| monash_tourism_quarterly | 427 | 3.12 | 7.3 | 1659 |
| monash_tourism_yearly | 518 | 2.79 | 5.4 | 963 |
| monash_traffic | 862 | 23.10 | 26.8 | 5581 |
| monash_weather | 1000 | 33.27 | 33.3 | 5161 |