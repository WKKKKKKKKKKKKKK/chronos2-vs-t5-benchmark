"""Controlled corruptions for the Chronos-2 vs Chronos-T5 edge-case robustness study.

Every perturbation acts on the *forecast context* only (the observed history fed to
the model), never on the held-out future the forecast is scored against. So each
function takes a 1-D float32 context array and returns a perturbed **copy** of the
same length; the clean ground-truth future is left untouched. This isolates one
question: given a corrupted input, how much worse is the forecast of the *true*
future? (We report the degradation relative to each model's own clean-context score,
so the robustness comparison is independent of the models' baseline accuracy gap.)

Three industrial-sensor failure modes (Saudi Aramco context):

  * noisy sensor spikes  -- sparse, large impulse outliers (sensor glitches / bit flips)
  * signal drift         -- a slow additive ramp (calibration drift / baseline wander)
  * missing data chunks  -- a contiguous block set to NaN (sensor dropout / telemetry loss)

All randomness is drawn from a caller-supplied ``numpy.random.Generator`` so a run is
fully reproducible. Magnitudes are expressed in units of each series' own robust scale
(median absolute deviation, fallback std) so a given severity is comparable across
datasets with wildly different units.

NAMING. Four functions produce seven swept families, and the example figures historically
used a SECOND set of names for four of them. Ten names, seven things. Canonical name is
the sweep name (left column); the paper uses the right column.

    sweep / results CSV   example figures      paper
    -------------------   ------------------   ---------------------------
    spikes_intensity      spikes_intensity     spike magnitude
    spikes_density        spikes_density       spike density
    drift                 drift_ramp           global drift
    drift_step            level_shift          transient level shift
    gap                   missing_random       dropout (random position)
    gap_boundary          missing_boundary     dropout (at forecast origin)
    regime_trend          --                   persistent trend change

Do not introduce a fourth naming scheme. If a figure and a table appear to describe
different experiments, this table is why.
"""
from __future__ import annotations

import numpy as np

# Held-constant spike parameters so spikes can be swept as two CONTROLLED variables:
#   spikes_intensity -> vary magnitude at this fixed density;
#   spikes_density   -> vary density (fraction) at this fixed magnitude.
SPIKE_FIX_FRAC = 0.05    # density held constant in the intensity sweep/figure
SPIKE_FIX_MAG = 20.0     # magnitude (x scale) held constant in the density sweep/figure

# Where the regime switch sits, as a fraction of context measured back from the forecast
# origin. Held CONSTANT while severity is swept, so severity drives exactly one quantity.
REGIME_SWITCH_FRAC = 0.25

__all__ = ["robust_scale", "add_spikes", "add_drift", "add_step", "drop_chunk",
           "add_regime_trend", "apply"]


def robust_scale(x: np.ndarray) -> float:
    """Per-series scale used to make severities unit-independent.

    MAD (median absolute deviation) rescaled to be a std-consistent estimator
    (x1.4826); robust to the very outliers/gaps we inject. Falls back to std, then
    to 1.0 for a constant or all-missing series, so the scale is always positive
    and finite.
    """
    f = x[np.isfinite(x)]
    if f.size < 2:
        return 1.0
    mad = np.median(np.abs(f - np.median(f))) * 1.4826
    if mad > 0:
        return float(mad)
    sd = float(np.std(f))
    return sd if sd > 0 else 1.0


def add_spikes(ctx: np.ndarray, rng: np.random.Generator,
               magnitude: float = 10.0, fraction: float = 0.05) -> np.ndarray:
    """Displace a random subset of context points by a large amount (sensor glitches).

    A random ``fraction`` of the *finite* context points are displaced by
    ``+/- magnitude * robust_scale`` (random sign per spike). Only previously
    observed points are spiked (NaNs stay NaN); the displacement is per-point and
    never a sustained shift of a contiguous block (that is `add_step`).

    Note on terminology: at the low end of the density grid (<=0.05) these are
    isolated impulses in the sensor-glitch sense. At the top of the grid (0.3-0.4)
    30-40% of points are displaced, which is heavy impulse *noise*, not a glitch.
    The write-up must say "impulse noise at increasing density" rather than "sparse
    spikes" when describing the density sweep; the high-density end is kept because
    it is where Chronos-T5 demonstrably *does* track severity.

    The sign is random per spike, which does NOT make the corruption invisible to
    Chronos-T5's scaler: that scaler is mean(|x|), so a -20*MAD spike inflates it
    exactly as much as +20*MAD. Measured scale inflation across the intensity sweep
    is ~1.8x. See `measure_clamping.py`.
    """
    x = np.array(ctx, dtype=np.float32, copy=True)
    finite = np.flatnonzero(np.isfinite(x))
    if finite.size < 2 or fraction <= 0 or magnitude == 0:
        return x
    k = max(1, int(round(fraction * finite.size)))
    idx = rng.choice(finite, size=min(k, finite.size), replace=False)
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=idx.size)
    x[idx] = x[idx] + signs * np.float32(magnitude * robust_scale(x))
    return x


def add_drift(ctx: np.ndarray, rng: np.random.Generator,
              slope: float = 5.0) -> np.ndarray:
    """Add a slow linear baseline drift (sensor calibration drift / wander).

    An additive ramp from 0 at the first context step to ``+/- slope * robust_scale``
    at the last step (random sign). Because the true future carries no such drift,
    this probes whether a model naively extrapolates the spurious trend.
    """
    x = np.array(ctx, dtype=np.float32, copy=True)
    n = x.size
    if n < 2 or slope == 0:
        return x
    sign = float(rng.choice([-1.0, 1.0]))
    ramp = np.linspace(0.0, sign * slope * robust_scale(x), n, dtype=np.float32)
    return x + ramp


def add_step(ctx: np.ndarray, rng: np.random.Generator,
             offset: float = 5.0, seg_frac: float = 0.3) -> np.ndarray:
    """Level-shift a *random contiguous segment* of the context (localised bias jump).

    Picks a random run covering ``seg_frac`` of the context and displaces only that block by
    ``+/- offset * robust_scale`` (random sign); the rest of the series is untouched. Models a
    sensor that drifts to a wrong baseline for a stretch of time and then recovers (vs
    `add_drift`'s ramp over the whole context). Whether it hurts the forecast depends on
    whether the shifted segment overlaps the recent context near the forecast origin.
    NaNs inside the segment are left as NaN.
    """
    x = np.array(ctx, dtype=np.float32, copy=True)
    n = x.size
    if n < 2 or offset == 0:
        return x
    seg = min(max(1, int(round(seg_frac * n))), n)
    start = int(rng.integers(0, n - seg + 1))
    sign = float(rng.choice([-1.0, 1.0]))
    shift = np.float32(sign * offset * robust_scale(x))
    block = x[start:start + seg]
    fin = np.isfinite(block)
    block[fin] = block[fin] + shift
    x[start:start + seg] = block
    return x


def drop_chunk(ctx: np.ndarray, rng: np.random.Generator,
               fraction: float = 0.25, position: str = "recent") -> np.ndarray:
    """Blank out a contiguous block as NaN (sensor dropout / telemetry loss).

    ``fraction`` of the context length is set to NaN as one contiguous run.
    ``position`` selects where the gap sits:
      * ``"recent"`` -- the gap ends at the forecast origin (most recent observations
        missing); the hardest, most realistic dropout case.
      * ``"middle"`` -- centred in the context.
      * ``"random"`` -- a random start (seeded).
    At least one point is always retained.
    """
    x = np.array(ctx, dtype=np.float32, copy=True)
    n = x.size
    if n < 2 or fraction <= 0:
        return x
    g = min(max(1, int(round(fraction * n))), n - 1)
    if position == "recent":
        start = n - g
    elif position == "middle":
        start = (n - g) // 2
    elif position == "random":
        start = int(rng.integers(0, n - g + 1))
    else:
        raise ValueError(f"unknown position {position!r}")
    x[start:start + g] = np.nan
    return x


def add_regime_trend(ctx: np.ndarray, rng: np.random.Generator,
                     slope: float = 5.0,
                     switch_frac: float = REGIME_SWITCH_FRAC) -> np.ndarray:
    """Change the *trend* partway through the context and keep it changed (regime shift).

    The first ``1 - switch_frac`` of the context is left untouched. From the switch point
    onward an additive ramp runs from 0 to ``+/- slope * robust_scale``, so the series is
    continuous at the switch (no level jump -- that is `add_step`) but its slope changes,
    and the change is still in force at the forecast origin.

    Why this is not a duplicate of the other families. The other five say "the past
    observations are wrong or missing", and the right response to all of them is to
    discard the damaged part. A regime shift says "the generating process changed and is
    still in the new mode", which is a *judgement*: extrapolate the new trend, or treat it
    as an anomaly? Nothing in the context settles it. It is also the only family in which
    the recent and the distant context disagree with each other.

    Matched to `add_drift` by construction: at the same ``slope`` both displace the final
    observation by exactly ``slope * robust_scale``. They differ only in whether the
    displacement accumulated over the whole history (drift, self-consistent, no
    breakpoint) or only over the last ``switch_frac`` (here, ~4x steeper, breakpoint
    present). The pair isolates the breakpoint while holding origin displacement fixed.

    NaNs are preserved (adding to NaN yields NaN).
    """
    x = np.array(ctx, dtype=np.float32, copy=True)
    n = x.size
    if n < 2 or slope == 0:
        return x
    m = min(max(2, int(round(switch_frac * n))), n)
    sign = float(rng.choice([-1.0, 1.0]))
    ramp = np.linspace(0.0, sign * slope * robust_scale(x), m, dtype=np.float32)
    x[n - m:] = x[n - m:] + ramp
    return x


# Registry of (family, severity-label, severity-value) -> perturbation callable.
# `apply` dispatches on family; severities are swept by the runner. "clean" is the
# identity (the per-model baseline every degradation is measured against).
def apply(family: str, ctx: np.ndarray, rng: np.random.Generator,
          severity: float) -> np.ndarray:
    """Dispatch one perturbation `family` at the given `severity` onto `ctx`."""
    if family == "clean":
        return np.array(ctx, dtype=np.float32, copy=True)
    if family == "spikes_intensity":                       # vary magnitude, density fixed
        return add_spikes(ctx, rng, magnitude=severity, fraction=SPIKE_FIX_FRAC)
    if family == "spikes_density":                         # vary density, magnitude fixed
        return add_spikes(ctx, rng, magnitude=SPIKE_FIX_MAG, fraction=severity)
    if family == "drift":
        return add_drift(ctx, rng, slope=severity)
    if family == "drift_step":
        return add_step(ctx, rng, offset=severity)
    if family == "gap":
        return drop_chunk(ctx, rng, fraction=severity, position="random")
    if family == "gap_boundary":                           # most-recent dropout (pinned to origin)
        return drop_chunk(ctx, rng, fraction=severity, position="recent")
    if family == "regime_trend":                           # persistent trend change near origin
        return add_regime_trend(ctx, rng, slope=severity)
    raise ValueError(f"unknown perturbation family {family!r}")
