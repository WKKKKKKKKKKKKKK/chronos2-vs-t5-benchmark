# Examples Analysis — Chronos-2 vs Chronos-T5 under corrupted input

> Based on `results/examples/<dataset>/` (25 datasets x 6 corruption categories; each figure =
> clean + 3 increasing severities), read figure-by-figure, cross-checked against
> `results/fig_degradation_curves.png` (relative degradation aggregated over all 25 datasets).
> Red = Chronos-2 p50, blue dashed = Chronos-T5 p50, green = held-out actual, grey = corrupted
> context, shaded = 10-90% interval.

---

## 0. Two lenses that must not be conflated (or the conclusions look contradictory)

1. **Clean accuracy / calibration (no corruption)** — Chronos-2 is clearly better.
   Zero-shot head-to-head: WQL 0.563 vs 0.687, MASE 0.751 vs 0.852, wins 22-23/25 datasets.
   In the figures: C2's interval covers the truth more often and its median is smoother.
2. **Robustness = relative degradation** (metric_corrupted / metric_clean, each model vs its OWN
   clean baseline) — more subtle.
   **Key trap:** relative degradation *penalises the model with the better clean baseline* (C2
   starts lower, so it has more room to fall). So "T5 degrades less than C2" does **NOT** mean
   "T5 has lower error after corruption" — C2 may still be more accurate post-corruption.
   Absolute accuracy = lens 1; this figure set measures *sensitivity to corruption*.

---

## 1. Per-corruption: typical exemplars + verdict (improves / degrades / par)

### A. Noisy spikes — INTENSITY (taller spikes, density fixed 5%)
- **Quant (25 ds):** C2 rises monotonically -> 1.5x (MASE) / 1.73x (WQL) at x40; **T5 is
  non-monotonic** — a bump to ~1.34x around x8-12, then **drops back to ~1.05x** at x30-40.
- **Mechanism:** T5's quantise-to-token front-end **clamps out-of-range extreme spikes** (so it
  neutralises huge spikes); moderate spikes distort the token histogram (T5's worst zone is the
  *middle*). C2's continuous patching has no clamp -> smooth degradation.
- **Typical exemplars:**
  - `exchange_rate/spikes_intensity` ★ — under spikes T5's band makes wild **downward excursions**
    (collapses to 0.3-0.5), while C2 stays tight at ~0.75.
  - `ercot/spikes_intensity` — T5's p50 **oscillates wildly** (down to -25000); C2 stays on target.
- **Verdict:** in the realistic moderate range **C2 is steadier / more predictable (better)**;
  only at extreme magnitudes does T5's clamping accidentally make it look more robust.

### B. Noisy spikes — DENSITY (more spikes, magnitude fixed x20)
- **Quant:** both worsen monotonically, but **C2 much more steeply**, especially WQL
  (C2 -> ~11x at 40% vs T5 -> ~3.5x).
- **Mechanism:** with many spikes **C2 inflates its predictive band massively** (honest but
  over-uncertain -> WQL penalty); T5 keeps a narrow band.
- **Typical exemplar:** `m5/spikes_density` ★ — from 2%->10%, **C2's band balloons from +-2 to
  +-7**, while T5's band barely changes.
- **Verdict:** C2's strength is amplitude-robustness, **not count-robustness**; under pervasive
  spiking **C2 is the more sensitive one (worse)**.

### C. Signal drift — gradual RAMP
- **Quant:** **catastrophic for both**, ~linear up to ~70-80x (MASE) / ~95-110x (WQL) at slope 32;
  C2 slightly worse than T5 throughout.
- **Mechanism:** the ramp reaches the origin, corrupting the recent level both models anchor to;
  forecasting in normalised space then de-normalising with the drifted stats drags the forecast
  along with the ramp. Per-series normalisation does not save it.
- **Typical exemplars:** `nn5/drift_ramp` ★, `ercot/drift_ramp` — forecast dragged far from the
  flat true future.
- **Verdict:** **the single most dangerous corruption; neither model is robust.** Detrend /
  bias-correct upstream.

### D. Level shift — random segment
- **Quant (25 ds):** moderate, rises to ~1.8x (C2 MASE) / ~1.45x (T5) at offset x20, **C2 worse**.
  NOTE: on the 5 sensor-like datasets it looked ~harmless; the all-25 aggregate is higher because
  on **short series** (yearly/quarterly) a "30%-of-context" segment more often overlaps the recent
  context.
- **Typical exemplar:** `monash_m1_monthly/level_shift` ★ — the shifted segment sits in the past,
  recent tail intact -> **both models almost unaffected** (the benign case).
- **Verdict:** **depends on whether the segment reaches recent context** — benign on long series,
  bites on short ones (C2 slightly more sensitive).

### E. Missing chunk — random position
- **Quant:** the **mildest** corruption — only ~1.2x (MASE) / ~1.18x (WQL) even at 70% blanked;
  both similar (C2 marginally worse at high fractions).
- **Typical exemplar:** `nn5/missing_random` ★ — gap in the past, recent context intact ->
  forecast still reproduces the weekly cycle.
- **Verdict:** both **robustly skip a randomly-placed gap** and forecast from the surviving recent
  context (C2 ingests NaN natively, T5 via its own missing handling). **Roughly par.**

### F. Missing chunk — boundary (most-recent dropout)
- The damaging placement: the gap is pinned to the context|horizon junction, removing the recent
  anchor.
- **Typical exemplar:** `monash_hospital/missing_boundary` ★ — as the recent block grows, both
  flatten; **C2 stays better-centred (near the series mean), T5 biases low.**
- **Verdict:** **placement >> size**; a sensor going dark right before "now" is the real risk.

---

## 2. Cross-cutting / broader findings

- **Unifying principle:** only corruptions that reach the **recent context** (ramp, boundary gap)
  hurt; corruptions in the **past** (random gap, past segment shift) are mild — because both
  models anchor their forecast to the most recent observations.
- **C2's "personality":** continuous, smooth, **honestly-wide uncertainty**; trend-aware
  (sometimes overshoots, e.g. `covid_deaths` keeps climbing while the truth has flattened).
- **T5's "personality":** discrete tokens -> **clamps extremes** (helps against huge spikes);
  narrower (sometimes over-confident) bands that go **erratic/multimodal** under moderate noise;
  tends to **mean-revert** on trends.
- **Intermittent / count data** (`car_parts`, `m5`): both give median ~0 (correct); T5 occasionally
  injects spurious probability spikes, C2's band is cleaner but over-inflates under dense spiking.

---

## 3. Effect of corruption SEVERITY

- **Monotone & smooth:** spikes density (both; WQL accelerates), drift (both; ~linear), level
  shift, missing (mild). More severe -> more degradation.
- **Non-monotone (important):** **T5 under spike intensity** — worst at *moderate* magnitude,
  recovers at *extreme* magnitude (clamping). For spike magnitude, "more severe" is NOT "more
  harmful" for T5.
- **Accelerating:** C2's WQL under density is the steepest curve in the whole panel (band
  inflation).

---

## 4. One-paragraph takeaway

- **No corruption:** Chronos-2 is uniformly more accurate with better intervals (a genuine
  capability *improvement*).
- **Under corruption (relative sensitivity):** it splits by case — **better at spike amplitude**
  (C2 steadier), **worse at spike density** (C2's band inflates), **both fail on drift**, **par on
  random gaps / past segment shifts**, **both hurt by most-recent dropout but C2 stays better
  centred**.
- **Always remember:** these curves are *relative degradation*, which is unfavourable to C2's
  better clean baseline; to ask "who is more accurate AFTER corruption (absolute)", pair this with
  the zero-shot head-to-head (where C2 wins).

> Six most representative figures (one per category, good for a report):
> `exchange_rate/spikes_intensity`, `m5/spikes_density`, `nn5/drift_ramp`,
> `monash_m1_monthly/level_shift`, `nn5/missing_random`, `monash_hospital/missing_boundary`.