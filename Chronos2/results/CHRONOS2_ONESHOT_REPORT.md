# Chronos-2 one-shot (LoRA fine-tuned) on Chronos Benchmark II

`amazon/chronos-2` LoRA-fine-tuned per dataset (explicit PyTorch loop, lr 1e-3 -> 0 over 1000 steps, LoRA r=8/alpha=16 on q/k/v/o + output layer), evaluated in univariate mode via the official gluonts pipeline (MASE + MeanWeightedSumQuantileLoss, gluonts split), cap=1000/dataset, quantile grid [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], on 25 Benchmark II datasets.

## Aggregated relative score (gmean of model / Seasonal-Naive)

Lower is better. The one-shot reference is the Chronos-**T5** paper's one-shot aggregate (Fig. 6); the C2-vs-T5 one-shot head-to-head lives in `../../chronos2_t5/one-shot/`.

| scenario | WQL | MASE |
| --- | --- | --- |
| one-shot LoRA (ours) | 0.592 | 0.768 |
| zero-shot univariate (ours) | 0.588 | 0.773 |
| zero-shot cross_learning (ours) | 0.563 | 0.751 |
| one-shot (Chronos-T5 paper, Fig. 6) | 0.597 | 0.760 |

## Per-dataset (one-shot LoRA, ours)

| dataset | MASE | WQL | n_series |
| --- | --- | --- | --- |
| monash_australian_electricity | 0.5104 | 0.0225 | 5 |
| monash_cif_2016 | 0.9286 | 0.0112 | 72 |
| monash_car_parts | 0.8194 | 0.9253 | 1000 |
| monash_covid_deaths | 32.8032 | 0.0333 | 266 |
| dominick | 0.9329 | 0.3356 | 1000 |
| ercot | 0.9513 | 0.0293 | 8 |
| exchange_rate | 1.4115 | 0.0098 | 8 |
| monash_fred_md | 0.6583 | 0.0549 | 107 |
| monash_hospital | 0.7498 | 0.0493 | 767 |
| monash_m1_monthly | 1.0288 | 0.1705 | 617 |
| monash_m1_quarterly | 1.6493 | 0.0806 | 203 |
| monash_m1_yearly | 3.3127 | 0.1052 | 181 |
| monash_m3_monthly | 0.8518 | 0.0881 | 1000 |
| monash_m3_quarterly | 1.1642 | 0.0706 | 756 |
| monash_m3_yearly | 3.2807 | 0.1485 | 645 |
| m4_quarterly | 1.1001 | 0.0708 | 1000 |
| m4_yearly | 2.9273 | 0.1066 | 1000 |
| m5 | 0.9287 | 0.5444 | 1000 |
| nn5 | 0.5515 | 0.1432 | 111 |
| monash_nn5_weekly | 0.8805 | 0.0815 | 111 |
| monash_tourism_monthly | 1.4368 | 0.0751 | 366 |
| monash_tourism_quarterly | 1.8798 | 0.0753 | 427 |
| monash_tourism_yearly | 3.6142 | 0.1454 | 518 |
| monash_traffic | 0.8076 | 0.2405 | 862 |
| monash_weather | 0.7845 | 0.1305 | 1000 |

## Inference efficiency (GPU (bfloat16))

Total forecast wall-time 34.1s over 25 datasets; peak GPU memory 7028 MB; mean 11.7 ms/series.

| dataset | n_series | latency_s | ms/series | peak_mem_MB |
| --- | --- | --- | --- | --- |
| monash_australian_electricity | 5 | 0.82 | 163.4 | 335 |
| monash_cif_2016 | 72 | 0.27 | 3.7 | 273 |
| monash_car_parts | 1000 | 0.59 | 0.6 | 319 |
| monash_covid_deaths | 266 | 0.51 | 1.9 | 455 |
| dominick | 1000 | 1.16 | 1.2 | 608 |
| ercot | 8 | 0.26 | 32.6 | 385 |
| exchange_rate | 8 | 0.34 | 42.4 | 379 |
| monash_fred_md | 107 | 0.38 | 3.5 | 440 |
| monash_hospital | 767 | 0.42 | 0.6 | 344 |
| monash_m1_monthly | 617 | 0.47 | 0.8 | 409 |
| monash_m1_quarterly | 203 | 0.41 | 2.0 | 333 |
| monash_m1_yearly | 181 | 0.23 | 1.3 | 297 |
| monash_m3_monthly | 1000 | 0.67 | 0.7 | 397 |
| monash_m3_quarterly | 756 | 0.50 | 0.7 | 330 |
| monash_m3_yearly | 645 | 0.42 | 0.7 | 319 |
| m4_quarterly | 1000 | 1.01 | 1.0 | 870 |
| m4_yearly | 1000 | 0.61 | 0.6 | 557 |
| m5 | 1000 | 2.32 | 2.3 | 1899 |
| nn5 | 111 | 0.45 | 4.1 | 466 |
| monash_nn5_weekly | 111 | 0.25 | 2.2 | 289 |
| monash_tourism_monthly | 366 | 0.56 | 1.5 | 559 |
| monash_tourism_quarterly | 427 | 0.38 | 0.9 | 383 |
| monash_tourism_yearly | 518 | 0.45 | 0.9 | 319 |
| monash_traffic | 862 | 9.15 | 10.6 | 7028 |
| monash_weather | 1000 | 11.50 | 11.5 | 7028 |