# Statistical appendix

**Analysis unit.** Every test below uses the *dataset* as the unit of analysis (n = 25). The multi-seed run provides four independent corruption draws per dataset; those are averaged within a dataset before testing, because seeds of the same dataset are not independent observations. An earlier pass treated the 100 (seed, dataset) curves as independent, which inflates significance.

**Pairing.** Both models receive byte-identical corrupted contexts on the same datasets, so model comparisons use a paired Wilcoxon signed-rank test rather than an unpaired test.

**Multiplicity.** Holm-Bonferroni and Benjamini-Hochberg adjusted p-values are reported alongside raw p, corrected within each family of tests.

**Metrics.** WQL (9-quantile grid) is the primary metric and carries the claim; MASE is reported as a robustness check. They are two correlated measurements of one hypothesis rather than two hypotheses, so each is corrected within its own family and no correction is applied across them. Both are shown for every C1 test so the reader can see that the conclusion does not depend on the choice.

**Intervals.** Bootstrap 95% CIs resample the 25 datasets with replacement (2000 draws).

| family                     | test                                                                 |   n | effect                                  |        p |   p_holm |     p_bh |
|:---------------------------|:---------------------------------------------------------------------|----:|:----------------------------------------|---------:|---------:|---------:|
| C1/spikes_intensity [WQL]  | Chronos-2 rho > Chronos-T5 rho (paired)                              |  25 | mean rho +0.796 vs -0.081               | 2.98e-08 | 8.94e-08 | 8.94e-08 |
| C1/spikes_intensity [WQL]  | chronos-2 rho > 0                                                    |  25 | mean rho +0.796 [95% CI +0.709, +0.867] | 6.13e-06 | 1.23e-05 | 9.2e-06  |
| C1/spikes_intensity [WQL]  | chronos-t5 rho > 0                                                   |  25 | mean rho -0.081 [95% CI -0.234, +0.068] | 0.837    | 0.837    | 0.837    |
| C1/spikes_density [WQL]    | Chronos-2 rho > Chronos-T5 rho (paired)                              |  25 | mean rho +0.952 vs +0.674               | 9.1e-06  | 1.8e-05  | 9.1e-06  |
| C1/spikes_density [WQL]    | chronos-2 rho > 0                                                    |  25 | mean rho +0.952 [95% CI +0.924, +0.974] | 6.01e-06 | 1.8e-05  | 9.1e-06  |
| C1/spikes_density [WQL]    | chronos-t5 rho > 0                                                   |  25 | mean rho +0.674 [95% CI +0.573, +0.770] | 6.94e-06 | 1.8e-05  | 9.1e-06  |
| C1/held-out [WQL]          | chronos-t5: degradation at sev 12 > at sev 40 (seeds 1-3)            |  25 | median 1.051 -> 1.026                   | 0.0377   | 0.0755   | 0.0755   |
| C1/held-out [WQL]          | chronos-2: degradation at sev 12 > at sev 40 (seeds 1-3)             |  25 | median 1.166 -> 1.711                   | 1        | 1        | 1        |
| C1/spikes_intensity [MASE] | Chronos-2 rho > Chronos-T5 rho (paired)                              |  25 | mean rho +0.828 vs -0.123               | 2.98e-08 | 8.94e-08 | 8.94e-08 |
| C1/spikes_intensity [MASE] | chronos-2 rho > 0                                                    |  25 | mean rho +0.828 [95% CI +0.783, +0.869] | 6.14e-06 | 1.23e-05 | 9.21e-06 |
| C1/spikes_intensity [MASE] | chronos-t5 rho > 0                                                   |  25 | mean rho -0.123 [95% CI -0.277, +0.025] | 0.948    | 0.948    | 0.948    |
| C1/spikes_density [MASE]   | Chronos-2 rho > Chronos-T5 rho (paired)                              |  25 | mean rho +0.955 vs +0.617               | 9.11e-06 | 1.77e-05 | 9.11e-06 |
| C1/spikes_density [MASE]   | chronos-2 rho > 0                                                    |  25 | mean rho +0.955 [95% CI +0.936, +0.971] | 5.91e-06 | 1.77e-05 | 9.11e-06 |
| C1/spikes_density [MASE]   | chronos-t5 rho > 0                                                   |  25 | mean rho +0.617 [95% CI +0.493, +0.728] | 8.86e-06 | 1.77e-05 | 9.11e-06 |
| C1/held-out [MASE]         | chronos-t5: degradation at sev 12 > at sev 40 (seeds 1-3)            |  25 | median 1.099 -> 1.038                   | 0.00623  | 0.0125   | 0.0125   |
| C1/held-out [MASE]         | chronos-2: degradation at sev 12 > at sev 40 (seeds 1-3)             |  25 | median 1.170 -> 1.464                   | 1        | 1        | 1        |
| C2/refutation              | MASE: Spearman(recovery, max clamped fraction (clamping hypothesis)) |  25 | rho = -0.279                            | 0.177    | 0.316    | 0.177    |
| C2/refutation              | MASE: Spearman(recovery, mean-scale inflation (scaling hypothesis))  |  25 | rho = -0.335                            | 0.102    | 0.316    | 0.141    |
| C2/refutation              | MASE: Spearman(recovery, growth of excursion reaching the model)     |  25 | rho = +0.532                            | 0.00625  | 0.0312   | 0.0187   |
| C2/refutation              | WQL: Spearman(recovery, max clamped fraction (clamping hypothesis))  |  25 | rho = -0.321                            | 0.118    | 0.316    | 0.141    |
| C2/refutation              | WQL: Spearman(recovery, mean-scale inflation (scaling hypothesis))   |  25 | rho = -0.358                            | 0.0791   | 0.316    | 0.141    |
| C2/refutation              | WQL: Spearman(recovery, growth of excursion reaching the model)      |  25 | rho = +0.545                            | 0.0048   | 0.0288   | 0.0187   |
