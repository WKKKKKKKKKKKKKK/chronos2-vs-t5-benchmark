# Are the four small datasets the most internally homogeneous?

E9b attributes the cross-learning gain to within-collection homogeneity rather than to group size. This measures the homogeneity instead of asserting it.

Mean pairwise Pearson correlation of **first differences** over the last 512 aligned context steps. First differences rather than levels: two series that merely share a trend correlate near 1 in levels whether or not they move together, which would score every trending collection as homogeneous.

| dataset                       |   n_series |   mean_pairwise_r |   frac_pairs_r_gt_0.5 |   rank | sub100   |
|:------------------------------|-----------:|------------------:|----------------------:|-------:|:---------|
| ercot                         |          8 |            0.9262 |                1      |      1 | True     |
| monash_australian_electricity |          5 |            0.6025 |                0.8    |      2 | True     |
| nn5                           |        100 |            0.5812 |                0.7822 |      3 | False    |
| monash_nn5_weekly             |        100 |            0.3849 |                0.2079 |      4 | False    |
| exchange_rate                 |          8 |            0.3747 |                0.2143 |      5 | True     |
| monash_traffic                |        100 |            0.2876 |                0.1055 |      6 | False    |
| monash_tourism_quarterly      |        100 |            0.1186 |                0.1586 |      7 | False    |
| monash_tourism_yearly         |         78 |            0.0738 |                0.0619 |      8 | False    |
| monash_hospital               |        100 |            0.0731 |                0.004  |      9 | False    |
| monash_covid_deaths           |        100 |            0.0596 |                0.0417 |     10 | False    |
| monash_fred_md                |        100 |            0.0578 |                0.0481 |     11 | False    |
| m4_yearly                     |         85 |            0.0575 |                0.0507 |     12 | False    |
| monash_tourism_monthly        |        100 |            0.0484 |                0.0436 |     13 | False    |
| monash_m1_yearly              |         47 |            0.0377 |                0.0352 |     14 | False    |
| monash_m1_quarterly           |         86 |            0.031  |                0.0465 |     15 | False    |
| monash_m3_quarterly           |         90 |            0.0196 |                0.0362 |     16 | False    |
| dominick                      |        100 |            0.0155 |                0.0012 |     17 | False    |
| monash_weather                |        100 |            0.0148 |                0.0073 |     18 | False    |
| monash_m3_monthly             |        100 |            0.0144 |                0.0036 |     19 | False    |
| m5                            |        100 |            0.01   |                0      |     20 | False    |
| monash_m3_yearly              |         66 |            0.0095 |                0.0354 |     21 | False    |
| m4_quarterly                  |        100 |            0.0047 |                0.0079 |     22 | False    |
| monash_cif_2016               |         71 |            0.0044 |                0.0137 |     23 | True     |
| monash_m1_monthly             |        100 |            0.0034 |                0.0103 |     24 | False    |
| monash_car_parts              |         95 |           -0.0018 |                0.013  |     25 | False    |

**the four sub-100 datasets rank [1, 2, 5, 23] of 25 by mean pairwise correlation.**
