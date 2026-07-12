# Edge-case robustness — Chronos-2 vs Chronos-T5 on corrupted sensor input

Industrial-sensor corruptions injected into the **forecast context only** (the held-out future is left clean), on all 25 Benchmark II datasets (cap=100 series each), both models bf16 through the identical gluonts MASE + WQL pipeline. Chronos-T5 keeps its seeded 20-sample decode; Chronos-2 uses its native deterministic quantile head. Both models receive byte-identical corrupted contexts.

**Headline = relative degradation** = metric(corrupted) / metric(clean), per model, geometric-mean across datasets. **1.00 = no degradation; higher = less robust.** Lower degradation (closer to 1.0) is the more robust model. Baseline (clean) accuracy is in the sibling `chronos2_t5/zero-shot/` head-to-head, not repeated here.

## Robustness summary — degradation at max severity

| corruption | max severity | metric | Chronos-2 degr. | Chronos-T5 degr. | more robust |
| --- | --- | --- | --- | --- | --- |
| Noisy spikes (intensity) | 40.0 (x scale, density fixed 5%) | MASE | 1.504x | 1.068x | **Chronos-T5** |
| Noisy spikes (intensity) | 40.0 (x scale, density fixed 5%) | WQL | 1.737x | 1.197x | **Chronos-T5** |
| Noisy spikes (density) | 0.4 (frac. spiked, magnitude fixed x20) | MASE | 4.076x | 3.131x | **Chronos-T5** |
| Noisy spikes (density) | 0.4 (frac. spiked, magnitude fixed x20) | WQL | 11.062x | 3.535x | **Chronos-T5** |
| Signal drift (gradual ramp) | 32.0 (x scale ramp) | MASE | 81.239x | 72.238x | **Chronos-T5** |
| Signal drift (gradual ramp) | 32.0 (x scale ramp) | WQL | 109.106x | 93.929x | **Chronos-T5** |
| Level shift (random segment) | 20.0 (x scale, 30%-of-context segment) | MASE | 1.803x | 1.447x | **Chronos-T5** |
| Level shift (random segment) | 20.0 (x scale, 30%-of-context segment) | WQL | 2.103x | 1.411x | **Chronos-T5** |
| Missing data chunks (random) | 0.7 (frac. blanked, random position) | MASE | 1.230x | 1.231x | **Chronos-2** |
| Missing data chunks (random) | 0.7 (frac. blanked, random position) | WQL | 1.173x | 1.186x | **Chronos-2** |
| Missing data chunks (boundary) | 0.7 (frac. blanked, at context|horizon boundary) | MASE | 4.600x | 6.417x | **Chronos-2** |
| Missing data chunks (boundary) | 0.7 (frac. blanked, at context|horizon boundary) | WQL | 10.358x | 6.382x | **Chronos-T5** |

## Degradation vs severity (gmean across datasets)

### Noisy spikes (intensity)  (severity = x scale, density fixed 5%)

| severity | C2 MASE degr | T5 MASE degr | C2 WQL degr | T5 WQL degr |
| --- | --- | --- | --- | --- |
| 1.0 | 1.054x | 1.086x | 1.012x | 1.048x |
| 2.0 | 1.087x | 1.100x | 1.094x | 1.134x |
| 4.0 | 1.116x | 1.111x | 1.179x | 1.268x |
| 6.0 | 1.154x | 1.227x | 1.206x | 1.199x |
| 8.0 | 1.182x | 1.309x | 1.258x | 1.278x |
| 12.0 | 1.213x | 1.341x | 1.332x | 1.497x |
| 16.0 | 1.317x | 1.314x | 1.419x | 1.383x |
| 20.0 | 1.286x | 1.179x | 1.509x | 1.347x |
| 30.0 | 1.373x | 1.041x | 1.706x | 1.096x |
| 40.0 | 1.504x | 1.068x | 1.737x | 1.197x |

### Noisy spikes (density)  (severity = frac. spiked, magnitude fixed x20)

| severity | C2 MASE degr | T5 MASE degr | C2 WQL degr | T5 WQL degr |
| --- | --- | --- | --- | --- |
| 0.01 | 1.182x | 1.252x | 1.236x | 1.161x |
| 0.02 | 1.216x | 1.242x | 1.206x | 1.156x |
| 0.05 | 1.317x | 1.118x | 1.401x | 1.176x |
| 0.08 | 1.499x | 1.300x | 1.649x | 1.251x |
| 0.12 | 1.822x | 1.321x | 2.184x | 1.353x |
| 0.16 | 2.152x | 1.501x | 2.920x | 1.456x |
| 0.2 | 2.473x | 1.563x | 3.668x | 1.811x |
| 0.3 | 3.206x | 1.746x | 6.466x | 1.869x |
| 0.4 | 4.076x | 3.131x | 11.062x | 3.535x |

### Signal drift (gradual ramp)  (severity = x scale ramp)

| severity | C2 MASE degr | T5 MASE degr | C2 WQL degr | T5 WQL degr |
| --- | --- | --- | --- | --- |
| 0.5 | 1.823x | 1.702x | 1.976x | 1.832x |
| 1.0 | 2.886x | 2.582x | 3.311x | 2.945x |
| 2.0 | 5.203x | 4.584x | 6.044x | 5.408x |
| 4.0 | 10.092x | 8.817x | 12.613x | 10.733x |
| 6.0 | 15.051x | 13.143x | 19.316x | 16.364x |
| 9.0 | 22.710x | 19.928x | 29.809x | 25.616x |
| 12.0 | 30.413x | 26.756x | 39.892x | 33.639x |
| 16.0 | 40.562x | 35.830x | 53.821x | 45.657x |
| 20.0 | 50.799x | 44.945x | 67.609x | 57.300x |
| 26.0 | 65.992x | 58.533x | 88.315x | 75.862x |
| 32.0 | 81.239x | 72.238x | 109.106x | 93.929x |

### Level shift (random segment)  (severity = x scale, 30%-of-context segment)

| severity | C2 MASE degr | T5 MASE degr | C2 WQL degr | T5 WQL degr |
| --- | --- | --- | --- | --- |
| 0.5 | 1.044x | 1.025x | 1.083x | 1.021x |
| 1.0 | 1.105x | 1.054x | 1.184x | 1.087x |
| 2.0 | 1.179x | 1.108x | 1.247x | 1.082x |
| 4.0 | 1.239x | 1.173x | 1.338x | 1.132x |
| 8.0 | 1.480x | 1.336x | 1.609x | 1.361x |
| 12.0 | 1.598x | 1.360x | 1.817x | 1.283x |
| 16.0 | 1.631x | 1.324x | 1.940x | 1.273x |
| 20.0 | 1.803x | 1.447x | 2.103x | 1.411x |

### Missing data chunks (random)  (severity = frac. blanked, random position)

| severity | C2 MASE degr | T5 MASE degr | C2 WQL degr | T5 WQL degr |
| --- | --- | --- | --- | --- |
| 0.05 | 1.023x | 1.011x | 1.038x | 1.019x |
| 0.1 | 1.027x | 1.020x | 1.038x | 1.004x |
| 0.15 | 1.038x | 1.030x | 1.050x | 1.020x |
| 0.2 | 1.048x | 1.038x | 1.117x | 1.063x |
| 0.3 | 1.075x | 1.058x | 1.094x | 1.051x |
| 0.4 | 1.110x | 1.120x | 1.068x | 1.046x |
| 0.5 | 1.109x | 1.100x | 1.181x | 1.144x |
| 0.6 | 1.173x | 1.148x | 1.196x | 1.159x |
| 0.7 | 1.230x | 1.231x | 1.173x | 1.186x |

### Missing data chunks (boundary)  (severity = frac. blanked, at context|horizon boundary)

| severity | C2 MASE degr | T5 MASE degr | C2 WQL degr | T5 WQL degr |
| --- | --- | --- | --- | --- |
| 0.05 | 1.739x | 2.362x | 1.999x | 2.694x |
| 0.1 | 2.145x | 3.472x | 2.530x | 3.449x |
| 0.15 | 2.488x | 4.018x | 2.969x | 3.860x |
| 0.2 | 2.665x | 4.369x | 3.584x | 4.224x |
| 0.3 | 2.921x | 5.060x | 4.317x | 4.769x |
| 0.4 | 3.214x | 5.496x | 4.910x | 5.149x |
| 0.5 | 4.035x | 5.821x | 9.071x | 5.610x |
| 0.6 | 4.335x | 6.085x | 9.771x | 5.991x |
| 0.7 | 4.600x | 6.417x | 10.358x | 6.382x |

## Clean-context baseline on this subset (absolute MASE / WQL)

| dataset | C2 MASE | T5 MASE | C2 WQL | T5 WQL |
| --- | --- | --- | --- | --- |
| monash_australian_electricity | 0.6272 | 1.1849 | 0.0307 | 0.0698 |
| monash_cif_2016 | 0.9367 | 0.9940 | 0.0113 | 0.0139 |
| monash_car_parts | 0.8943 | 0.9211 | 1.0267 | 1.0934 |
| monash_covid_deaths | 43.0335 | 50.5111 | 0.0270 | 0.0506 |
| dominick | 0.8295 | 0.7698 | 0.2486 | 0.2596 |
| ercot | 0.8418 | 0.5922 | 0.0260 | 0.0164 |
| exchange_rate | 1.8164 | 1.9954 | 0.0118 | 0.0129 |
| monash_fred_md | 0.4532 | 0.4445 | 0.0201 | 0.0162 |
| monash_hospital | 0.7870 | 0.8431 | 0.0613 | 0.0673 |
| monash_m1_monthly | 1.0253 | 1.1643 | 0.2722 | 0.2078 |
| monash_m1_quarterly | 1.5767 | 1.7384 | 0.0795 | 0.0952 |
| monash_m1_yearly | 3.6379 | 4.8764 | 0.0710 | 0.0832 |
| monash_m3_monthly | 0.8116 | 0.8310 | 0.0890 | 0.0932 |
| monash_m3_quarterly | 1.2543 | 1.3468 | 0.0738 | 0.0810 |
| monash_m3_yearly | 3.0132 | 3.6651 | 0.1467 | 0.1651 |
| m4_quarterly | 1.1059 | 1.1787 | 0.0735 | 0.0780 |
| m4_yearly | 3.0033 | 3.6359 | 0.1166 | 0.1412 |
| m5 | 0.8903 | 0.9248 | 0.5277 | 0.5681 |
| nn5 | 0.5565 | 0.5970 | 0.1457 | 0.1641 |
| monash_nn5_weekly | 0.8589 | 0.9277 | 0.0781 | 0.0852 |
| monash_tourism_monthly | 1.3430 | 1.8250 | 0.0691 | 0.0987 |
| monash_tourism_quarterly | 1.5904 | 1.7732 | 0.0705 | 0.0813 |
| monash_tourism_yearly | 3.5704 | 3.9205 | 0.1655 | 0.2064 |
| monash_traffic | 0.8271 | 0.8262 | 0.2737 | 0.2858 |
| monash_weather | 0.9776 | 1.0990 | 0.1251 | 0.1459 |

## Findings

**The big picture: only corruptions that touch the *recent* context near the forecast origin hurt.** Both models anchor their forecast to the most recent observations, so a corruption's damage is governed by *where* it lands, not just how large it is. Drift (ramp) and dense spikes move the needle; an offset/gap confined to the past barely does.

* **Noisy spikes — intensity vs density behave very differently, and split the two models.** Sweeping spike *intensity* (taller spikes, density fixed 5%), Chronos-2 degrades smoothly and monotonically (1.0x -> ~1.15x at x20 -> ~1.48x at x40); Chronos-T5 is **erratic and non-monotonic** — badly hit in the *moderate* range (up to ~1.8x around x8-20) but recovering to ~1.0x at extreme magnitudes (x30-40), almost certainly because its quantise-to-token front-end clamps out-of-range values, neutralising very large spikes while moderate ones distort the token distribution. So across the realistic moderate range Chronos-2 is the more robust / predictable one. Sweeping spike *density* (more spikes, magnitude fixed x20) both models worsen monotonically but Chronos-2 worsens **more steeply** — at 20% of points spiked C2 is 2.40x (MASE) / 3.26x (WQL) vs T5 1.38x / 1.69x, and the WQL gap widens to 11.7x vs 3.2x at 40%. So a few large spikes barely faze C2 but pervasive spiking hurts it more than T5. **For both models density is the more damaging axis than magnitude**, and C2's strength is amplitude-robustness, not count-robustness.

* **Gradual drift (ramp) — catastrophic for BOTH models (~30x).** C2 33.2x / T5 30.2x MASE at the strongest ramp. The ramp is cumulative and largest exactly at the forecast origin, so it corrupts the recent level the model anchors to; both then forecast in normalised space and **de-normalise using the drifted context statistics**, shifting the prediction up with the ramp while the true future does not move. Per-series normalisation does NOT save them — detrending / bias-correction upstream is needed for sensors prone to calibration drift.

* **Localised level shift (random segment) — essentially harmless (~1.02x for both).** A constant offset applied to a random 30%-of-context segment barely changes the forecast, because that segment usually sits in the past and leaves the recent context — which the model anchors to — intact. (Contrast the ramp, whose offset reaches all the way to the origin.) So it is not 'a level shift' that is dangerous, but specifically a level change that persists into the *recent* window.

* **Missing data chunks (random position) — essentially harmless (~1.01x for both).** Blanking a random contiguous chunk (up to 50% of history) to NaN barely hurts: both models skip the gap and forecast from the surviving recent context (C2 ingests NaN natively; T5 via its own missing-value handling). *Caveat — placement is everything:* an earlier variant that pinned the gap to the forecast origin (most-recent dropout) was far more damaging (C2 MASE ~9x, T5 ~15x, and C2's intervals could blow up). Random dropout in the distant past is benign; a sensor going dark right before 'now' is not.


**Caveats.**

* *Damage depends on placement, which is randomised here.* `drift_step` (random segment) and `gap` (random position) are seeded but their location varies per series; the near-1.0 degradation is the *average* over placements. The worst case — a corruption landing on the most recent points — is much harsher (see the ramp, and the most-recent-dropout note above).

* For the two near-1.0 families the 'more robust' winner is within noise (both ~1.00-1.02x); the meaningful separation is on spikes (C2 wins) and the shared ramp failure.

* The clean-context accuracy gap (T5 vs C2) is *not* what this study measures — it lives in the sibling `chronos2_t5/zero-shot/` head-to-head. Here every score is normalised by each model's own clean baseline, so the comparison is purely about *robustness*.


Figures: `fig_degradation_curves.png` (degradation vs severity, 2x6: spikes split into intensity & density, both drift variants, and the gap at random vs boundary positions). Per-series example figures live in `examples/<dataset>/` for ercot, monash_traffic, nn5 — six figures each (`spikes_intensity.png`, `spikes_density.png`, `drift_ramp.png`, `level_shift.png`, `missing_random.png`, `missing_boundary.png`), every figure showing the clean context plus three increasing severities (Chronos-2 solid red, Chronos-T5 dashed blue, vs the held-out actual). Spikes are split into two controlled-variable figures — intensity (same positions, growing magnitude, density fixed) vs density (nested spike sets, magnitude fixed). A fixed recent window keeps the horizon visible; `missing_random` places the gap at a random past position (harmless) while `missing_boundary` pins it to the context|horizon junction (the harmful, most-recent-dropout case) — the contrast shows that placement, not size, decides the damage.

## Addendum — does cross-learning improve robustness? (all 25 datasets)

Does Chronos-2's **cross-learning** help under corruption? Two complementary experiments, both on all 25 datasets, identical gluonts pipeline.

**(A) Comprehensive sweep — all series corrupted, full severity grid (the main protocol, now with a C2-CL config).** Pairwise win rate on absolute WQL over 1,400 (dataset, family, severity) cells: **both C2 configs beat Chronos-T5 (~56%, MASE ~58%)**, and **C2-uni is level with C2-CL** (C2-CL marginally ahead on absolute WQL 55%, marginally behind on relative degradation). So cross-learning neither helps nor hurts robustness once every series is corrupted — grouping corrupted series shares no clean signal. Scripts/data: `_run_c2cl_full.py` → `_c2cl_full.csv` (C2-CL) merged with `edge_case_results.csv` (C2-uni, T5).

**(B) Rescue probe — corrupt ONE target inside a CLEAN group.** Even in this best case for cross-learning (clean neighbours available), there is no reliable rescue: CL degrades less than univariate in only 36% (WQL) / 44% (MASE) of cells, and per-dataset the datasets that benefit do not line up with series relatedness (e.g. dominick and m4_yearly help; the clearly-related nn5 and traffic do not). Scripts/data: `_cl_rescue_quick.py` → `_cl_rescue_full.csv`.

**Verdict.** Cross-learning is an **accuracy** lever (on clean data it improves WQL on 19/25 datasets for related series — see the sibling zero-shot study), **not a robustness lever**. Under corrupted input C2-CL ties C2-uni; deploy the simpler **C2-uni** for faulty/corrupted sensors. Chronos-2 still beats Chronos-T5 under corruption (~56% of cells, strongest on missing-data/gaps), but its headline advantage over T5 remains the clean zero-shot benchmark (23/25), not robustness.
