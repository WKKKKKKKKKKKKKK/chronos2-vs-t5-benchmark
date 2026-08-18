# E10 -- persistent trend change (high severity grid, WQL)

Source: `edge_case_regime.csv`, severities [np.float64(0.5), np.float64(1.0), np.float64(2.0), np.float64(4.0), np.float64(8.0), np.float64(12.0), np.float64(16.0), np.float64(20.0)].

`regime_trend` changes the trend in the last 25% of the context and keeps it changed through the forecast origin. Unlike the other six families it does not say "past observations are wrong" but "the process changed and is still in the new mode", which the model must judge rather than filter.

Analysis unit is the dataset (n = 25); seeds averaged within dataset before testing; CIs bootstrap the datasets (2000 draws).

## Q1 -- does the C1 split reappear?

| model      |   n |   mean_rho |   ci_lo |   ci_hi | datasets_positive   |
|:-----------|----:|-----------:|--------:|--------:|:--------------------|
| chronos-2  |  25 |     0.9988 |  0.9967 |  1      | 25/25               |
| chronos-t5 |  25 |     0.9976 |  0.9943 |  0.9998 | 25/25               |

Paired Wilcoxon (Chronos-2 rho > Chronos-T5 rho), n = 25: p = 0.231

**C1 IS BOUNDED: Chronos-T5 does track severity on this family**

## Q2 -- matched against `drift` at equal origin displacement

Both families displace the final context point by the same amount at a given severity; they differ only in whether a breakpoint is present. Seed 0 only, because the seed-0 sweep is the only run containing `drift`.

| model      |   n |   drift_mean |   regime_mean |    diff |   ci_lo |   ci_hi |      p | severities         |
|:-----------|----:|-------------:|--------------:|--------:|--------:|--------:|-------:|:-------------------|
| chronos-2  |  25 |      37.2367 |       38.7688 |  1.5321 |  0.2197 |  3.3539 | 0.0667 | 0.5,1,2,4,12,16,20 |
| chronos-t5 |  25 |      31.9961 |       31.7446 | -0.2515 | -1.4637 |  0.6285 | 0.173  | 0.5,1,2,4,12,16,20 |

## Q3 -- absolute degradation by severity

|   severity |   chronos-2 |   chronos-t5 |
|-----------:|------------:|-------------:|
|        0.5 |      2.6993 |       2.3463 |
|        1   |      4.8005 |       4.112  |
|        2   |      9.3366 |       7.966  |
|        4   |     18.8516 |      15.9411 |
|        8   |     38.8185 |      32.5676 |
|       12   |     58.7186 |      48.4675 |
|       16   |     79.0735 |      64.9694 |
|       20   |     99.0866 |      81.0107 |

![regime](fig_regime_high_WQL.png)
