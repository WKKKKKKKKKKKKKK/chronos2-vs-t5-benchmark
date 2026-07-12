# Chronos-2 zero-shot on Chronos Benchmark II — univariate vs cross-learning

`amazon/chronos-2` via the official gluonts metric pipeline (MASE + MeanWeightedSumQuantileLoss, gluonts split), cap=1000/dataset, quantile grid [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], on 25 Benchmark II datasets. Cross-learning is Chronos-2's full cross-learning mode (technical report §5.1): 1-d inputs, every item in a batch shares one group id, group/batch size 100.

## Aggregated relative score (gmean of model / Seasonal-Naive) — vs the Chronos-2 paper

Lower is better. `Chronos-2 (paper)` = the report's Benchmark II skill scores (arXiv:2510.15821 Table 5) converted via G = 1 - skill/100; it aggregates all 27 datasets + full data (ours: 25 + cap=1000 + bf16). The Chronos-T5 head-to-head is in the sibling `chronos2_t5/zero-shot/` project.

| metric | C2 univariate | C2 cross-learning | Chronos-2 (paper) |
| --- | --- | --- | --- |
| WQL  | 0.588 | 0.563 | 0.534 |
| MASE | 0.773 | 0.751 | 0.735 |

## Per-dataset MASE / WQL (univariate vs cross-learning)

| dataset | MASE uni | MASE xl | WQL uni | WQL xl |
| --- | --- | --- | --- | --- |
| monash_australian_electricity | 0.6272 | 0.6174 | 0.0307 | 0.0289 |
| monash_cif_2016 | 0.9369 | 0.8374 | 0.0113 | 0.0081 |
| monash_car_parts | 0.8324 | 0.8340 | 0.9787 | 0.9848 |
| monash_covid_deaths | 35.4297 | 32.5474 | 0.0396 | 0.0350 |
| dominick | 0.9394 | 0.9409 | 0.3456 | 0.3401 |
| ercot | 0.8418 | 0.7846 | 0.0260 | 0.0249 |
| exchange_rate | 1.8164 | 1.8491 | 0.0118 | 0.0118 |
| monash_fred_md | 0.4848 | 0.4258 | 0.0200 | 0.0216 |
| monash_hospital | 0.7723 | 0.7401 | 0.0539 | 0.0510 |
| monash_m1_monthly | 1.0271 | 0.9849 | 0.1466 | 0.1184 |
| monash_m1_quarterly | 1.6627 | 1.6443 | 0.0833 | 0.0862 |
| monash_m1_yearly | 3.8298 | 3.5519 | 0.1378 | 0.1287 |
| monash_m3_monthly | 0.8017 | 0.8139 | 0.0857 | 0.0851 |
| monash_m3_quarterly | 1.1635 | 1.1169 | 0.0705 | 0.0673 |
| monash_m3_yearly | 2.9967 | 2.9382 | 0.1469 | 0.1430 |
| m4_quarterly | 1.1671 | 1.1292 | 0.0735 | 0.0706 |
| m4_yearly | 3.2243 | 3.1416 | 0.1135 | 0.1093 |
| m5 | 0.9284 | 0.9336 | 0.5460 | 0.5498 |
| nn5 | 0.5768 | 0.5559 | 0.1488 | 0.1449 |
| monash_nn5_weekly | 0.8829 | 0.8757 | 0.0816 | 0.0801 |
| monash_tourism_monthly | 1.4230 | 1.3775 | 0.0726 | 0.0719 |
| monash_tourism_quarterly | 1.5662 | 1.5515 | 0.0648 | 0.0609 |
| monash_tourism_yearly | 3.6676 | 3.6533 | 0.1513 | 0.1497 |
| monash_traffic | 0.8189 | 0.8316 | 0.2417 | 0.2428 |
| monash_weather | 0.7535 | 0.7559 | 0.1252 | 0.1259 |

## Cross-learning grouping (each batch of ~100 series = one group)

All series participate; `n_groups` = number of cross-learning groups (= batches).

| dataset | n_series | n_groups |
| --- | --- | --- |
| monash_australian_electricity | 5 | 1 |
| monash_cif_2016 | 72 | 1 |
| monash_car_parts | 1000 | 10 |
| monash_covid_deaths | 266 | 3 |
| dominick | 1000 | 10 |
| ercot | 8 | 1 |
| exchange_rate | 8 | 1 |
| monash_fred_md | 107 | 2 |
| monash_hospital | 767 | 8 |
| monash_m1_monthly | 617 | 7 |
| monash_m1_quarterly | 203 | 3 |
| monash_m1_yearly | 181 | 2 |
| monash_m3_monthly | 1000 | 10 |
| monash_m3_quarterly | 756 | 8 |
| monash_m3_yearly | 645 | 7 |
| m4_quarterly | 1000 | 10 |
| m4_yearly | 1000 | 10 |
| m5 | 1000 | 10 |
| nn5 | 111 | 2 |
| monash_nn5_weekly | 111 | 2 |
| monash_tourism_monthly | 366 | 4 |
| monash_tourism_quarterly | 427 | 5 |
| monash_tourism_yearly | 518 | 6 |
| monash_traffic | 862 | 9 |
| monash_weather | 1000 | 10 |

## Inference efficiency (GPU (bfloat16))

| mode | total latency_s | mean ms/series | peak_mem_MB |
| --- | --- | --- | --- |
| univariate | 32.6 | 6.9 | 7026 |
| cross_learning | 28.6 | 2.4 | 2176 |