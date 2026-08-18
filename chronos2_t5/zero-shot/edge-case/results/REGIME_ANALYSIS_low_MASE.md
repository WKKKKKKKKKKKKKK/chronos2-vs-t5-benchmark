# E10 -- persistent trend change (low severity grid, MASE)

Source: `edge_case_regime_low.csv`, severities [np.float64(0.01), np.float64(0.02), np.float64(0.04), np.float64(0.06), np.float64(0.09), np.float64(0.13), np.float64(0.18), np.float64(0.25)].

`regime_trend` changes the trend in the last 25% of the context and keeps it changed through the forecast origin. Unlike the other six families it does not say "past observations are wrong" but "the process changed and is still in the new mode", which the model must judge rather than filter.

Analysis unit is the dataset (n = 25); seeds averaged within dataset before testing; CIs bootstrap the datasets (2000 draws).

## Q1 -- does the C1 split reappear?

| model      |   n |   mean_rho |   ci_lo |   ci_hi | datasets_positive   |
|:-----------|----:|-----------:|--------:|--------:|:--------------------|
| chronos-2  |  25 |     0.8695 |  0.7976 |  0.9281 | 25/25               |
| chronos-t5 |  25 |     0.7005 |  0.5519 |  0.8195 | 24/25               |

Paired Wilcoxon (Chronos-2 rho > Chronos-T5 rho), n = 25: p = 0.00141

**C1 IS BOUNDED: Chronos-T5 does track severity on this family**

## Q3 -- absolute degradation by severity

|   severity |   chronos-2 |   chronos-t5 |
|-----------:|------------:|-------------:|
|       0.01 |      1.004  |       1.007  |
|       0.02 |      1.013  |       1.0197 |
|       0.04 |      1.0294 |       1.0366 |
|       0.06 |      1.073  |       1.0708 |
|       0.09 |      1.1368 |       1.1348 |
|       0.13 |      1.2278 |       1.2176 |
|       0.18 |      1.366  |       1.3433 |
|       0.25 |      1.5507 |       1.5157 |

## Q4 -- effect-size-matched control

The first regime sweep started at 2.3x degradation while spike magnitude tops out at 1.5x, so a difference in slope was confounded with a difference in how damaging the corruption is. Here the regime sweep is restricted to severities [0.04, 0.06, 0.09, 0.13, 0.18, 0.25], whose mean degradation falls inside the spike band, and both slopes are computed by the same code path on seeds 1-3.

| family                        | model      |   n |   mean_rho |   ci_lo |   ci_hi | datasets_positive   |
|:------------------------------|:-----------|----:|-----------:|--------:|--------:|:--------------------|
| regime_trend (effect-matched) | chronos-2  |  25 |     0.8895 |  0.8209 |  0.9451 | 25/25               |
| regime_trend (effect-matched) | chronos-t5 |  25 |     0.7059 |  0.5657 |  0.8324 | 24/25               |
| spikes_intensity              | chronos-2  |  25 |     0.8259 |  0.7765 |  0.872  | 25/25               |
| spikes_intensity              | chronos-t5 |  25 |    -0.108  | -0.2703 |  0.0525 | 9/25                |

![regime](fig_regime_low_MASE.png)
