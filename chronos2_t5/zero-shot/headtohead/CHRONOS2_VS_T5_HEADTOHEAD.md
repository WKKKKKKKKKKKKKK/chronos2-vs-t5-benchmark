# Chronos-2 vs Chronos-T5 — zero-shot head-to-head (Benchmark II)

Same 25 datasets, same cap=1000, same gluonts pipeline (MASE + WQL), same machine, both bf16. Chronos-T5 is univariate-only; C2 has univariate & full cross-learning.

## Aggregated relative score (gmean of model / Seasonal-Naive; lower is better)

| metric | C2 uni | C2 cross-learning | C2 paper¹ | T5 measured | T5 paper |
| --- | --- | --- | --- | --- | --- |
| WQL  | 0.588 | **0.563** | 0.534 | 0.687 | 0.675 |
| MASE | 0.773 | **0.751** | 0.735 | 0.852 | 0.839 |

¹ `C2 paper` = Chronos-2 paper (arXiv:2510.15821, Table 5) skill scores via G=1-skill/100; aggregate over all 27 datasets/full data (ours: 25 + cap=1000).

**Win rate vs T5 measured (per-dataset):** C2 cross-learning MASE 22/25, WQL 23/25.

## Inference efficiency (same machine, both bf16; lower is better)

| run | total forecast time (s) | mean ms/series | peak GPU mem (MB) |
| --- | --- | --- | --- |
| C2 univariate | 32.6 | 6.9 | 7026 |
| C2 cross-learning | 28.6 | 2.4 | 2176 |
| T5 measured | 202.4 | 29.2 | 5581 |

C2 cross-learning vs T5: **7.1x** less total time, **2.6x** less peak memory.

## Per-dataset MASE

| dataset | C2 uni | C2 xl | T5 measured |
| --- | --- | --- | --- |
| dominick | 0.9394 | 0.9409 | 0.9398 |
| ercot | 0.8418 | 0.7846 | 0.5922 |
| exchange_rate | 1.8164 | 1.8491 | 1.9954 |
| m4_quarterly | 1.1671 | 1.1292 | 1.2334 |
| m4_yearly | 3.2243 | 3.1416 | 3.6549 |
| m5 | 0.9284 | 0.9336 | 0.9617 |
| monash_australian_electricity | 0.6272 | 0.6174 | 1.1849 |
| monash_car_parts | 0.8324 | 0.8340 | 0.8768 |
| monash_cif_2016 | 0.9369 | 0.8374 | 1.0017 |
| monash_covid_deaths | 35.4297 | 32.5474 | 42.7023 |
| monash_fred_md | 0.4848 | 0.4258 | 0.4533 |
| monash_hospital | 0.7723 | 0.7401 | 0.8068 |
| monash_m1_monthly | 1.0271 | 0.9849 | 1.1712 |
| monash_m1_quarterly | 1.6627 | 1.6443 | 1.8246 |
| monash_m1_yearly | 3.8298 | 3.5519 | 4.8971 |
| monash_m3_monthly | 0.8017 | 0.8139 | 0.8772 |
| monash_m3_quarterly | 1.1635 | 1.1169 | 1.2915 |
| monash_m3_yearly | 2.9967 | 2.9382 | 3.4174 |
| monash_nn5_weekly | 0.8829 | 0.8757 | 0.9625 |
| monash_tourism_monthly | 1.4230 | 1.3775 | 1.9379 |
| monash_tourism_quarterly | 1.5662 | 1.5515 | 1.7683 |
| monash_tourism_yearly | 3.6676 | 3.6533 | 3.9751 |
| monash_traffic | 0.8189 | 0.8316 | 0.8268 |
| monash_weather | 0.7535 | 0.7559 | 0.8336 |
| nn5 | 0.5768 | 0.5559 | 0.6147 |

## Per-dataset WQL

| dataset | C2 uni | C2 xl | T5 measured |
| --- | --- | --- | --- |
| dominick | 0.3456 | 0.3401 | 0.3686 |
| ercot | 0.0260 | 0.0249 | 0.0170 |
| exchange_rate | 0.0118 | 0.0118 | 0.0138 |
| m4_quarterly | 0.0735 | 0.0706 | 0.0848 |
| m4_yearly | 0.1135 | 0.1093 | 0.1404 |
| m5 | 0.5460 | 0.5498 | 0.6108 |
| monash_australian_electricity | 0.0307 | 0.0289 | 0.0724 |
| monash_car_parts | 0.9787 | 0.9848 | 1.0505 |
| monash_cif_2016 | 0.0113 | 0.0081 | 0.0115 |
| monash_covid_deaths | 0.0396 | 0.0350 | 0.0673 |
| monash_fred_md | 0.0200 | 0.0216 | 0.0149 |
| monash_hospital | 0.0539 | 0.0510 | 0.0583 |
| monash_m1_monthly | 0.1466 | 0.1184 | 0.1454 |
| monash_m1_quarterly | 0.0833 | 0.0862 | 0.1138 |
| monash_m1_yearly | 0.1378 | 0.1287 | 0.1806 |
| monash_m3_monthly | 0.0857 | 0.0851 | 0.1006 |
| monash_m3_quarterly | 0.0705 | 0.0673 | 0.0822 |
| monash_m3_yearly | 0.1469 | 0.1430 | 0.1630 |
| monash_nn5_weekly | 0.0816 | 0.0801 | 0.0947 |
| monash_tourism_monthly | 0.0726 | 0.0719 | 0.1115 |
| monash_tourism_quarterly | 0.0648 | 0.0609 | 0.0662 |
| monash_tourism_yearly | 0.1513 | 0.1497 | 0.2114 |
| monash_traffic | 0.2417 | 0.2428 | 0.2645 |
| monash_weather | 0.1252 | 0.1259 | 0.1536 |
| nn5 | 0.1488 | 0.1449 | 0.1731 |