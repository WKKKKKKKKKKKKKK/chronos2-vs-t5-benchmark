# E1 -- Clamping analysis (spikes_intensity, MASE)

Chronos-T5 = `amazon/chronos-t5-small`; value grid `[-15, +15]` over 4093 bins, 
clamp tokens 3 / 4095. 250 (dataset, severity) cells over 
25 datasets. Degradation is the sweep's own 
`MASE_degr` (per-dataset ratio to that model's clean score).

## Q1 - aggregate: does clamping switch on where the curve recovers?

|   severity |   degr_gmean |   clamped_frac |   series_any_clamped |   nominal_p99 |   realized_p99 |   scale_mean |
|-----------:|-------------:|---------------:|---------------------:|--------------:|---------------:|-------------:|
|          1 |       1.086  |         0.0008 |               0.0348 |        2.7591 |         2.6569 |  1.60514e+06 |
|          2 |       1.0995 |         0.0007 |               0.0316 |        2.7517 |         2.6795 |  1.57859e+06 |
|          4 |       1.1115 |         0.0006 |               0.0252 |        3.0077 |         2.9566 |  1.59801e+06 |
|          6 |       1.2275 |         0.0005 |               0.0212 |        3.4545 |         3.4105 |  1.66009e+06 |
|          8 |       1.3089 |         0.0006 |               0.0244 |        3.9711 |         3.9213 |  1.69256e+06 |
|         12 |       1.3414 |         0.0008 |               0.0328 |        4.8464 |         4.794  |  1.79555e+06 |
|         16 |       1.3143 |         0.0013 |               0.0412 |        5.6249 |         5.5488 |  1.82859e+06 |
|         20 |       1.1788 |         0.0019 |               0.0564 |        6.3196 |         6.2205 |  1.91091e+06 |
|         30 |       1.0408 |         0.0032 |               0.08   |        7.6852 |         7.5401 |  2.0672e+06  |
|         40 |       1.0679 |         0.0045 |               0.1124 |        8.7343 |         8.5115 |  2.2392e+06  |

## Q2 - per-dataset: do the datasets that clamp recover?

Spearman(max clamped fraction, recovery) **rho = -0.279**, 
p = 0.1772, n = 25 datasets. 
`recovery` = peak degradation / degradation at the largest severity; 
a value above 1 means the curve came back down after peaking.

| dataset                       |   n_series |   peak_degr |   peak_severity |   end_degr |   end_severity |   recovery |   max_clamped_frac |   max_any_clamped |   mad_over_meanabs |
|:------------------------------|-----------:|------------:|----------------:|-----------:|---------------:|-----------:|-------------------:|------------------:|-------------------:|
| monash_car_parts              |        100 |      1.0343 |               6 |     0.9805 |             40 |     1.0549 |             0.0354 |              0.71 |             0.3944 |
| m5                            |        100 |      1.0188 |               1 |     0.9879 |             40 |     1.0313 |             0.0303 |              0.78 |             0.4023 |
| dominick                      |        100 |      1.1935 |              12 |     1.0235 |             40 |     1.1661 |             0.0166 |              0.44 |             0.2898 |
| monash_covid_deaths           |        100 |      1.2041 |              12 |     1.0398 |             40 |     1.1581 |             0.0152 |              0.39 |             0.344  |
| monash_weather                |        100 |      1.057  |              12 |     0.9317 |             40 |     1.1344 |             0.0095 |              0.25 |             0.2815 |
| monash_traffic                |        100 |      1.0324 |              20 |     1.0097 |             40 |     1.0225 |             0.0029 |              0.14 |             0.3089 |
| m4_yearly                     |        100 |      1.9378 |              16 |     1.6366 |             40 |     1.184  |             0.0014 |              0.04 |             0.1894 |
| monash_fred_md                |        100 |      7.6358 |               6 |     1.4531 |             40 |     5.2548 |             0.0005 |              0.04 |             0.2381 |
| monash_tourism_yearly         |        100 |      2.7724 |              12 |     1.1271 |             40 |     2.4597 |             0.0004 |              0.01 |             0.1938 |
| monash_tourism_quarterly      |        100 |      1.3636 |               6 |     1.0042 |             40 |     1.3579 |             0.0001 |              0.01 |             0.2689 |
| monash_cif_2016               |         72 |      1.4779 |               6 |     1.0811 |             40 |     1.367  |             0      |              0    |             0.158  |
| exchange_rate                 |          8 |      6.3899 |              16 |     0.9058 |             40 |     7.0543 |             0      |              0    |             0.1476 |
| m4_quarterly                  |        100 |      1.2099 |               4 |     1.0431 |             40 |     1.1599 |             0      |              0    |             0.1939 |
| ercot                         |          8 |      4.7038 |              20 |     1.0577 |             40 |     4.4473 |             0      |              0    |             0.1475 |
| monash_australian_electricity |          5 |      4.0535 |               8 |     0.9693 |             40 |     4.1819 |             0      |              0    |             0.1851 |
| monash_hospital               |        100 |      1.0349 |               4 |     0.9743 |             40 |     1.0623 |             0      |              0    |             0.1663 |
| monash_m1_monthly             |        100 |      1.1998 |              40 |     1.1998 |             40 |     1      |             0      |              0    |             0.1573 |
| monash_m3_quarterly           |        100 |      1.5875 |               8 |     1.0151 |             40 |     1.5639 |             0      |              0    |             0.1377 |
| monash_m3_monthly             |        100 |      1.7293 |              16 |     1.0946 |             40 |     1.5799 |             0      |              0    |             0.1662 |
| monash_m1_yearly              |        100 |      1.7961 |              20 |     1.0859 |             40 |     1.6541 |             0      |              0    |             0.1383 |
| monash_m1_quarterly           |        100 |      1.5705 |               8 |     1.3207 |             40 |     1.1892 |             0      |              0    |             0.1678 |
| monash_tourism_monthly        |        100 |      1.163  |               4 |     1.0015 |             40 |     1.1613 |             0      |              0    |             0.2767 |
| monash_nn5_weekly             |        100 |      1.0077 |               4 |     1.0045 |             40 |     1.0032 |             0      |              0    |             0.1273 |
| monash_m3_yearly              |        100 |      1.4145 |              16 |     1.005  |             40 |     1.4074 |             0      |              0    |             0.1472 |
| nn5                           |        100 |      1.0537 |               4 |     1.0045 |             40 |     1.049  |             0      |              0    |             0.2309 |

## Q3 - same cells, two x-axes

- vs **nominal** severity: Spearman rho = -0.101 (p = 0.11)
- vs **realised** excursion: Spearman rho = -0.320 (p = 2.37e-07)

![mechanism](fig_clamping_mechanism_spikes_intensity_MASE.png)
