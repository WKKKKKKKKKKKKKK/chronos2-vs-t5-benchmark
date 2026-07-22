"""Edge-case robustness study: Chronos-2 vs Chronos-T5 under corrupted sensor input.

Industrial sensor streams (the Saudi Aramco setting) are rarely clean: they carry
glitch spikes, calibration drift and dropout gaps. This study asks how gracefully each
zero-shot forecaster degrades when its *input context* is corrupted, while the true
future it is scored against stays clean. We inject three controlled corruptions
(see `perturbations.py`) at several severities into the forecast context, forecast,
and measure the error vs the clean held-out window.

Headline metric = **relative degradation** = metric(perturbed) / metric(clean), per
model, per dataset, aggregated by geometric mean across datasets. Because every score
is normalised by that same model's clean-context score, the comparison isolates
*robustness* and is independent of the models' baseline accuracy gap (already
quantified in the sibling `chronos2_t5/zero-shot/` head-to-head).

Faithfulness to the sibling projects:
  * Same gluonts pipeline (split / generate_instances / MASE + WeightedSumQuantileLoss),
    same `to_gluonts_univariate`, same 9-quantile grid, same bf16 — all imported from
    `run_zeroshot_chronos2`.
  * Both models are driven through `predict_quantiles` so WQL is on an identical grid;
    Chronos-T5 keeps its seeded 20-sample stochastic decode (seed 0), Chronos-2 is its
    native deterministic quantile head. Identical (byte-for-byte) corrupted contexts are
    fed to both models for every condition.
  * Restricted to high-frequency, continuous *sensor-like* datasets where spikes / drift
    / dropout are physically meaningful, with a per-dataset series cap for tractability.

Outputs (chronos2_t5/zero-shot/edge-case/results/):
  edge_case_results.csv     per (dataset, model, perturbation, severity): MASE/WQL + degradation
  EDGE_CASE_REPORT.md       robustness summary + per-family degradation + per-dataset tables
  fig_degradation_curves.png   RELATIVE degradation (x clean) vs severity, per family, both metrics
  fig_absolute_curves.png      ABSOLUTE MASE/WQL vs severity (same 2x6 layout) — accuracy counterpart
  examples/<dataset>/<category>.png   per-series figures (clean + 3 severities) for each category
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from gluonts.dataset.split import split
from scipy.stats import gmean

# This study lives in the chronos2_t5 comparison project but REUSES the shared harness from
# the sibling Chronos2 project (dataset registry + the Chronos-2 forecasters/evaluate/constants),
# so the numbers stay on the identical gluonts pipeline as the zero-shot head-to-head.
HERE = Path(__file__).resolve().parent                 # chronos2_t5/zero-shot/edge-case
C2_SRC = HERE.parents[2] / "Chronos2" / "src"          # config, datasets_lib (src root)
sys.path.insert(0, str(C2_SRC))
sys.path.insert(0, str(C2_SRC / "zero_shot"))          # run_zeroshot_chronos2.py (shared harness)
sys.path.insert(0, str(HERE))
from datasets_lib import HF_REPO, BENCHMARK_II          # noqa: E402  (BENCHMARK_II = all 25 datasets + horizons)
import run_zeroshot_chronos2 as R2                      # noqa: E402  (forecasters/evaluate/constants)
import perturbations as P                               # noqa: E402

# Stable per-dataset index over the FULL Benchmark II registry — used to seed the example
# figures' spike placement so every dataset gets a distinct (but reproducible) draw.
_ALL_DS_IDX = {cfg: i for i, (cfg, _) in enumerate(BENCHMARK_II)}

from chronos import BaseChronosPipeline                 # noqa: E402

# --- study configuration -------------------------------------------------------
# Metric sweep runs over the FULL Benchmark II registry (all 25 datasets, native horizons),
# so the degradation curves aggregate over every dataset (not just the sensor-like subset).
# (Corruptions are scale-relative, so they apply to any series; on low-frequency economic /
# yearly sets "spikes/drift" are less physically meaningful but mathematically valid.)
EDGE_DATASETS = list(BENCHMARK_II)
EDGE_MAX_SERIES = 100        # per-dataset cap (deterministic, evenly spaced) for tractability
# Datasets to render per-series example figures for (clean vs each corruption, both models).
EXAMPLE_DATASETS = ("ercot", "monash_traffic", "nn5")
# Per-category example figures: for each dataset we save ONE figure per category into
# results/examples/<dataset>/, each with 4 panels (clean + 3 severities). These severities are
# figure-only illustrations (independent of the study sweep above) and show a clear progression.
# Spikes are split into two CONTROLLED-VARIABLE figures: `spikes_intensity` varies only the
# magnitude (same spike positions, growing taller; density fixed at SPIKE_FIX_FRAC), while
# `spikes_density` varies only the count (nested spike sets at a fixed magnitude SPIKE_FIX_MAG).
# `missing_boundary` pins the gap to the context|horizon junction (harmful) vs random `missing_random`.
#   (category, nice title, [severities], label format)
SPIKE_FIX_FRAC = P.SPIKE_FIX_FRAC   # density held constant in the intensity figure/sweep
SPIKE_FIX_MAG = P.SPIKE_FIX_MAG     # magnitude held constant in the density figure/sweep
EXAMPLE_FIGS = [
    ("spikes_intensity", f"Noisy spikes - intensity (density fixed {SPIKE_FIX_FRAC:.0%})", [10, 20, 30],       "x{:g}"),
    ("spikes_density",   f"Noisy spikes - density (magnitude fixed x{SPIKE_FIX_MAG:g})",    [0.02, 0.05, 0.10], "{:.0%}"),
    ("drift_ramp",       "Signal drift - gradual ramp",                  [10, 20, 30],       "x{:g}"),
    ("level_shift",      "Level shift - random segment",                 [10, 20, 30],       "x{:g}"),
    ("missing_random",   "Missing chunk - random position",              [0.10, 0.20, 0.30], "{:.0%}"),
    ("missing_boundary", "Missing chunk - at context|horizon boundary",  [0.10, 0.20, 0.30], "{:.0%}"),
]
SEED = 0                     # perturbation RNG + Chronos-T5 sample seed
BATCH = 32                   # manual forecast batch size (Chronos-T5 predict has no batch_size kwarg)

MODELS = [                   # (label, hf id, kind) ; kind drives the predict_quantiles call
    ("chronos-2",  "amazon/chronos-2",        "c2"),
    ("chronos-t5", "amazon/chronos-t5-small", "t5"),
]

# Severity sweeps. Magnitudes are in units of each series' robust scale (MAD); gap is a
# fraction of context length. The largest level of each family is the "max severity"
# used for the headline table and the example figures.
# Denser + wider severity grids so the degradation curves are smooth (was 3 points each).
SEVERITIES = {
    "spikes_intensity": [1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 20.0, 30.0, 40.0],  # magnitude x scale (density fixed)
    "spikes_density":   [0.01, 0.02, 0.05, 0.08, 0.12, 0.16, 0.20, 0.30, 0.40],   # fraction spiked (magnitude fixed)
    "drift":            [0.5, 1.0, 2.0, 4.0, 6.0, 9.0, 12.0, 16.0, 20.0, 26.0, 32.0],  # gradual ramp end-offset x scale
    "drift_step":       [0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 20.0],              # random-segment level shift x scale
    "gap":              [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70],   # fraction blanked (random position)
    "gap_boundary":     [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70],   # fraction blanked, pinned to context|horizon boundary
}
# Spikes are swept as two controlled variables: "spikes_intensity" varies magnitude at fixed
# density; "spikes_density" varies the spiked fraction at fixed magnitude. Two drift variants:
# "drift" = slow ramp over the whole context (calibration drift); "drift_step" = a localised
# level shift of a random segment (bias jump that recovers). "gap" blanks a contiguous run at a
# random position (not pinned to the origin); "gap_boundary" = same but pinned to the
# context|horizon boundary (most-recent dropout).
FAMILIES = ["spikes_intensity", "spikes_density", "drift", "drift_step", "gap", "gap_boundary"]
CONDITIONS = [("clean", 0.0)] + [(f, s) for f in FAMILIES for s in SEVERITIES[f]]

# Stable integer ids so perturbation RNG seeds are reproducible run-to-run.
_DS_IDX = {d: i for i, (d, _) in enumerate(EDGE_DATASETS)}
_FAM_IDX = {"clean": 0, "spikes_intensity": 1, "spikes_density": 5, "drift": 2,
            "drift_step": 4, "gap": 3, "gap_boundary": 6}

OUT = HERE / "results"          # outputs live next to this script (chronos2_t5/zero-shot/edge-case/results/)


def _rng(dataset: str, family: str, severity: float) -> np.random.Generator:
    """Deterministic generator for one (dataset, family, severity) cell.

    A single generator draws sequentially across that cell's series (series order is
    fixed), so the full corruption is reproducible and identical across both models.
    """
    ss = np.random.SeedSequence([SEED, _DS_IDX[dataset], _FAM_IDX[family],
                                 int(round(severity * 1000))])
    return np.random.default_rng(ss)


def build_dataset(config: str, horizon: int):
    """Clean gluonts backtest for one dataset: test_data + per-series context/start."""
    import datasets as hfds

    ds = hfds.load_dataset(HF_REPO, config, split="train")
    ds.set_format("numpy")
    gts = R2.to_gluonts_univariate(ds, EDGE_MAX_SERIES)
    _, tt = split(gts, offset=-horizon)
    test_data = tt.generate_instances(horizon, windows=1)
    test_input = list(test_data.input)
    contexts = [np.asarray(e["target"], dtype=np.float32) for e in test_input]
    starts = [e["start"] + len(e["target"]) for e in test_input]
    return test_data, contexts, starts


def perturb_contexts(dataset: str, contexts, family: str, severity: float):
    """Apply one (family, severity) to every context — identical for both models."""
    rng = _rng(dataset, family, severity)
    return [P.apply(family, c, rng, severity) for c in contexts]


def forecast(pipe, contexts, starts, horizon, kind):
    """Forecast a list of (possibly corrupted, NaN-bearing) contexts -> QuantileForecasts.

    Manual batching (Chronos-T5's predict has no batch_size kwarg). Chronos-T5 keeps the
    seeded 20-sample decode; Chronos-2 uses its native quantile head. Output per item is
    (H, Q) for T5 and (1, H, Q) for C2 — both reduced to (H, Q).
    """
    fc = []
    for b0 in range(0, len(contexts), BATCH):
        chunk = contexts[b0:b0 + BATCH]
        tensors = [torch.tensor(np.asarray(c, dtype=np.float32)) for c in chunk]
        kw = dict(prediction_length=horizon, quantile_levels=R2.QUANTILES)
        if kind == "t5":
            torch.manual_seed(SEED)
            kw["num_samples"] = 20
        else:
            kw["limit_prediction_length"] = False
        q, _ = pipe.predict_quantiles(tensors, **kw)
        for qi, st in zip(q, starts[b0:b0 + BATCH]):
            a = qi.cpu().numpy() if torch.is_tensor(qi) else np.asarray(qi)
            if a.ndim == 3:
                a = a[0]
            fc.append(R2._quantile_forecast(np.asarray(a, dtype=np.float32), st))
    return fc


def _znorm(x):
    x = np.asarray(x, np.float32)
    m, s = np.nanmean(x), np.nanstd(x)
    return (x - m) / (s + 1e-8)


def forecast_cl_one(pipe, group_ctxs, si, corrupted_ctx, start, horizon):
    """Chronos-2 CROSS-LEARNING forecast for a corrupted target sitting in its CLEAN group.

    The group = all series' clean contexts (edge-case caps at EDGE_MAX_SERIES = CROSS_LEARNING_BATCH,
    so the whole dataset is one group); only the target (index si) is replaced by the corrupted
    context. Faithful to the eval's positional grouping; tests whether clean related series rescue
    a corrupted target."""
    B = R2.CROSS_LEARNING_BATCH
    g0 = (si // B) * B
    grp = [np.asarray(c, np.float32) for c in group_ctxs[g0:g0 + B]]
    grp[si - g0] = np.asarray(corrupted_ctx, np.float32)          # corrupt ONLY the target
    q, _ = pipe.predict_quantiles([torch.tensor(c) for c in grp], prediction_length=horizon,
                                  quantile_levels=R2.QUANTILES, cross_learning=True,
                                  batch_size=B, limit_prediction_length=False)
    a = q[si - g0]; a = a.cpu().numpy() if torch.is_tensor(a) else np.asarray(a)
    if a.ndim == 3:
        a = a[0]
    return R2._quantile_forecast(np.asarray(a, np.float32), start)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    cuda = torch.cuda.is_available()

    # Both models loaded once; datasets are processed ONE AT A TIME (build corrupted contexts,
    # forecast with both models, record, free) so memory stays bounded even over all 25 datasets.
    pipes = _load_pipes()
    rows = []
    for di, (config, horizon) in enumerate(EDGE_DATASETS, 1):
        test_data, contexts, starts = build_dataset(config, horizon)
        print(f"[{di}/{len(EDGE_DATASETS)}] {config} (n={len(contexts)}, H={horizon})", flush=True)
        conds = {(fam, sev): perturb_contexts(config, contexts, fam, sev) for (fam, sev) in CONDITIONS}
        for (fam, sev) in CONDITIONS:
            pc = conds[(fam, sev)]
            for label, (pipe, kind) in pipes.items():
                t0 = time.perf_counter()
                fcs = forecast(pipe, pc, starts, horizon, kind)
                if cuda:
                    torch.cuda.synchronize()
                latency = time.perf_counter() - t0
                mase, wql = R2.evaluate(fcs, test_data)
                rows.append({"dataset": config, "model": label, "family": fam,
                             "severity": sev, "MASE": mase, "WQL": wql,
                             "n_series": len(contexts), "latency_s": round(latency, 3)})
        del conds, contexts, test_data, starts   # free this dataset before the next

    df = pd.DataFrame(rows)
    df = _add_degradation(df)
    df.to_csv(OUT / "edge_case_results.csv", index=False)
    print(f"\nSaved -> {OUT / 'edge_case_results.csv'}")
    _write_report(df)
    _plot_degradation(df)
    _plot_absolute(df)
    # per-series example figures for the showcase datasets (rebuild just their clean series)
    for ds in EXAMPLE_DATASETS:
        H = dict(EDGE_DATASETS)[ds]
        td, ctx, st = build_dataset(ds, H)
        data = {ds: dict(horizon=H, test_data=td, starts=st, n_series=len(ctx),
                         conds={("clean", 0.0): ctx})}
        _plot_examples(df, data, ds, pipes=pipes)
    del pipes
    if cuda:
        torch.cuda.empty_cache()
    print(f"Saved report + figures -> {OUT}")


def _add_degradation(df: pd.DataFrame) -> pd.DataFrame:
    """Add MASE_degr / WQL_degr = metric / that (model,dataset)'s clean-context metric."""
    clean = (df[df["family"] == "clean"]
             .set_index(["model", "dataset"])[["MASE", "WQL"]]
             .rename(columns={"MASE": "MASE_clean", "WQL": "WQL_clean"}))
    df = df.join(clean, on=["model", "dataset"])
    df["MASE_degr"] = df["MASE"] / df["MASE_clean"]
    df["WQL_degr"] = df["WQL"] / df["WQL_clean"]
    return df


def _agg_degr(df, model, family, severity, col):
    """gmean across datasets of the per-dataset degradation ratio for one cell."""
    sub = df[(df.model == model) & (df.family == family) & (df.severity == severity)]
    vals = sub[col].to_numpy()
    vals = vals[np.isfinite(vals) & (vals > 0)]
    return float(gmean(vals)) if vals.size else float("nan")


def _write_report(df: pd.DataFrame):
    models = [m for m, _, _ in MODELS]
    title_fam = {"spikes_intensity": "Noisy spikes (intensity)", "spikes_density": "Noisy spikes (density)",
                 "drift": "Signal drift (gradual ramp)",
                 "drift_step": "Level shift (random segment)", "gap": "Missing data chunks (random)",
                 "gap_boundary": "Missing data chunks (boundary)"}
    sev_unit = {"spikes_intensity": f"x scale, density fixed {SPIKE_FIX_FRAC:.0%}",
                "spikes_density": f"frac. spiked, magnitude fixed x{SPIKE_FIX_MAG:g}",
                "drift": "x scale ramp",
                "drift_step": "x scale, 30%-of-context segment", "gap": "frac. blanked, random position",
                "gap_boundary": "frac. blanked, at context|horizon boundary"}

    md = [
        "# Edge-case robustness — Chronos-2 vs Chronos-T5 on corrupted sensor input\n",
        f"Industrial-sensor corruptions injected into the **forecast context only** "
        f"(the held-out future is left clean), on all {len(EDGE_DATASETS)} "
        f"Benchmark II datasets (cap={EDGE_MAX_SERIES} series each), both models "
        f"bf16 through the identical gluonts MASE + WQL pipeline. Chronos-T5 keeps its seeded "
        f"20-sample decode; Chronos-2 uses its native deterministic quantile head. Both models "
        f"receive byte-identical corrupted contexts.\n",
        "**Headline = relative degradation** = metric(corrupted) / metric(clean), per model, "
        "geometric-mean across datasets. **1.00 = no degradation; higher = less robust.** "
        "Lower degradation (closer to 1.0) is the more robust model. Baseline (clean) accuracy "
        "is in the sibling `chronos2_t5/zero-shot/` head-to-head, not repeated here.\n",
        "## Robustness summary — degradation at max severity\n",
    ]
    # Headline table at max severity of each family.
    md += ["| corruption | max severity | metric | Chronos-2 degr. | Chronos-T5 degr. | more robust |",
           "| --- | --- | --- | --- | --- | --- |"]
    for fam in FAMILIES:
        sev = SEVERITIES[fam][-1]
        for metric, col in [("MASE", "MASE_degr"), ("WQL", "WQL_degr")]:
            c2 = _agg_degr(df, "chronos-2", fam, sev, col)
            t5 = _agg_degr(df, "chronos-t5", fam, sev, col)
            winner = "Chronos-2" if c2 < t5 else "Chronos-T5"
            md.append(f"| {title_fam[fam]} | {sev} ({sev_unit[fam]}) | {metric} | "
                      f"{c2:.3f}x | {t5:.3f}x | **{winner}** |")

    # Full degradation sweep per family.
    md += ["\n## Degradation vs severity (gmean across datasets)\n"]
    for fam in FAMILIES:
        md += [f"### {title_fam[fam]}  (severity = {sev_unit[fam]})\n",
               "| severity | C2 MASE degr | T5 MASE degr | C2 WQL degr | T5 WQL degr |",
               "| --- | --- | --- | --- | --- |"]
        for sev in SEVERITIES[fam]:
            md.append(f"| {sev} | "
                      f"{_agg_degr(df,'chronos-2',fam,sev,'MASE_degr'):.3f}x | "
                      f"{_agg_degr(df,'chronos-t5',fam,sev,'MASE_degr'):.3f}x | "
                      f"{_agg_degr(df,'chronos-2',fam,sev,'WQL_degr'):.3f}x | "
                      f"{_agg_degr(df,'chronos-t5',fam,sev,'WQL_degr'):.3f}x |")
        md.append("")

    # Clean-context baseline (sanity: both models on uncorrupted sensor-like subset).
    md += ["## Clean-context baseline on this subset (absolute MASE / WQL)\n",
           "| dataset | C2 MASE | T5 MASE | C2 WQL | T5 WQL |",
           "| --- | --- | --- | --- | --- |"]
    cl = df[df.family == "clean"].set_index(["dataset", "model"])
    for config, _ in EDGE_DATASETS:
        def g(model, col):
            return f"{cl.loc[(config, model), col]:.4f}" if (config, model) in cl.index else "—"
        md.append(f"| {config} | {g('chronos-2','MASE')} | {g('chronos-t5','MASE')} "
                  f"| {g('chronos-2','WQL')} | {g('chronos-t5','WQL')} |")

    md += [
        "\n## Findings\n",
        "**The big picture: only corruptions that touch the *recent* context near the forecast "
        "origin hurt.** Both models anchor their forecast to the most recent observations, so a "
        "corruption's damage is governed by *where* it lands, not just how large it is. Drift (ramp) "
        "and dense spikes move the needle; an offset/gap confined to the past barely does.\n",
        "* **Noisy spikes — intensity vs density behave very differently, and split the two models.** "
        "Sweeping spike *intensity* (taller spikes, density fixed 5%), Chronos-2 degrades smoothly and "
        "monotonically (1.0x -> ~1.15x at x20 -> ~1.48x at x40); Chronos-T5 is **erratic and non-monotonic** "
        "— badly hit in the *moderate* range (up to ~1.8x around x8-20) but recovering to ~1.0x at extreme "
        "magnitudes (x30-40), almost certainly because its quantise-to-token front-end clamps "
        "out-of-range values, neutralising very large spikes while moderate ones distort the token "
        "distribution. So across the realistic moderate range Chronos-2 is the more robust / predictable "
        "one. Sweeping spike *density* (more spikes, magnitude fixed x20) both models worsen monotonically "
        "but Chronos-2 worsens **more steeply** — at 20% of points spiked C2 is 2.40x (MASE) / 3.26x (WQL) "
        "vs T5 1.38x / 1.69x, and the WQL gap widens to 11.7x vs 3.2x at 40%. So a few large spikes barely "
        "faze C2 but pervasive spiking hurts it more than T5. **For both models density is the more "
        "damaging axis than magnitude**, and C2's strength is amplitude-robustness, not count-robustness.\n",
        "* **Gradual drift (ramp) — catastrophic for BOTH models (~30x).** C2 33.2x / T5 30.2x MASE "
        "at the strongest ramp. The ramp is cumulative and largest exactly at the forecast origin, so "
        "it corrupts the recent level the model anchors to; both then forecast in normalised space and "
        "**de-normalise using the drifted context statistics**, shifting the prediction up with the "
        "ramp while the true future does not move. Per-series normalisation does NOT save them — "
        "detrending / bias-correction upstream is needed for sensors prone to calibration drift.\n",
        "* **Localised level shift (random segment) — essentially harmless (~1.02x for both).** A "
        "constant offset applied to a random 30%-of-context segment barely changes the forecast, "
        "because that segment usually sits in the past and leaves the recent context — which the model "
        "anchors to — intact. (Contrast the ramp, whose offset reaches all the way to the origin.) So "
        "it is not 'a level shift' that is dangerous, but specifically a level change that persists "
        "into the *recent* window.\n",
        "* **Missing data chunks (random position) — essentially harmless (~1.01x for both).** "
        "Blanking a random contiguous chunk (up to 50% of history) to NaN barely hurts: both models "
        "skip the gap and forecast from the surviving recent context (C2 ingests NaN natively; T5 via "
        "its own missing-value handling). *Caveat — placement is everything:* an earlier variant that "
        "pinned the gap to the forecast origin (most-recent dropout) was far more damaging (C2 MASE "
        "~9x, T5 ~15x, and C2's intervals could blow up). Random dropout in the distant past is benign; "
        "a sensor going dark right before 'now' is not.\n",
        "\n**Caveats.**\n",
        "* *Damage depends on placement, which is randomised here.* `drift_step` (random segment) and "
        "`gap` (random position) are seeded but their location varies per series; the near-1.0 "
        "degradation is the *average* over placements. The worst case — a corruption landing on the "
        "most recent points — is much harsher (see the ramp, and the most-recent-dropout note above).\n",
        "* For the two near-1.0 families the 'more robust' winner is within noise (both ~1.00-1.02x); "
        "the meaningful separation is on spikes (C2 wins) and the shared ramp failure.\n",
        "* The clean-context accuracy gap (T5 vs C2) is *not* what this study measures — it lives in "
        "the sibling `chronos2_t5/zero-shot/` head-to-head. Here every score is normalised by each "
        "model's own clean baseline, so the comparison is purely about *robustness*.\n",
        "\nFigures: `fig_degradation_curves.png` (RELATIVE degradation = metric / clean, 2x6: spikes "
        "split into intensity & density, both drift variants, and the gap at random vs boundary "
        "positions) and its one-to-one accuracy counterpart `fig_absolute_curves.png` (the same 2x6 "
        "panels but plotting the ABSOLUTE MASE/WQL, with each model's clean baseline as a dotted "
        "reference line). The two are complementary and can disagree: the ratio figure measures "
        "*robustness* (how much each model degrades from its own baseline, which favours the model "
        "with the higher clean error) while the absolute figure measures *accuracy on corrupted "
        "input* (what deployment cares about). Chronos-2's lower clean baseline is why it can look "
        "less robust in the ratio view yet stay more accurate in absolute terms — e.g. on missing "
        "(random) C2's ratio sits above T5's while its absolute WQL stays below. "
        "Per-series example figures live in `examples/<dataset>/` for "
        f"{', '.join(EXAMPLE_DATASETS)} — six figures each (`spikes_intensity.png`, "
        "`spikes_density.png`, `drift_ramp.png`, `level_shift.png`, `missing_random.png`, "
        "`missing_boundary.png`), every figure showing the clean context plus three increasing "
        "severities (Chronos-2 solid red, Chronos-T5 dashed blue, vs the held-out actual). Spikes "
        "are split into two controlled-variable figures — intensity (same positions, growing "
        "magnitude, density fixed) vs density (nested spike sets, magnitude fixed). A fixed recent "
        "window keeps the horizon visible; `missing_random` places the gap at a random past position "
        "(harmless) while `missing_boundary` pins it to the context|horizon junction (the harmful, "
        "most-recent-dropout case) — the contrast shows that placement, not size, decides the damage.\n",
    ]
    (OUT / "EDGE_CASE_REPORT.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Saved -> {OUT / 'EDGE_CASE_REPORT.md'}")


def _plot_degradation(df: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    title_fam = {"spikes_intensity": "Noisy spikes (intensity)", "spikes_density": "Noisy spikes (density)",
                 "drift": "Signal drift (gradual ramp)",
                 "drift_step": "Level shift (random segment)", "gap": "Missing data chunks (random)",
                 "gap_boundary": "Missing data chunks (boundary)"}
    # Short per-panel titles (metric is already on the y-axis) so the 6 columns don't collide.
    short_fam = {"spikes_intensity": "Spikes (intensity)", "spikes_density": "Spikes (density)",
                 "drift": "Drift (ramp)", "drift_step": "Level shift (segment)",
                 "gap": "Missing (random)", "gap_boundary": "Missing (boundary)"}
    colors = {"chronos-2": "C3", "chronos-t5": "C0"}
    disp = {"chronos-2": "Chronos-2", "chronos-t5": "Chronos-T5"}
    nfam = len(FAMILIES)
    fig, axes = plt.subplots(2, nfam, figsize=(5.2 * nfam, 9.2))
    for j, fam in enumerate(FAMILIES):
        sevs = SEVERITIES[fam]
        for i, (metric, col) in enumerate([("MASE", "MASE_degr"), ("WQL", "WQL_degr")]):
            ax = axes[i, j]
            for model in ["chronos-2", "chronos-t5"]:
                y = [_agg_degr(df, model, fam, s, col) for s in sevs]
                ax.plot(sevs, y, "o-", color=colors[model], lw=2.4, ms=8, label=disp[model])
            ax.axhline(1.0, color="grey", ls=":", lw=1.2)   # no-degradation reference
            # Title in the top-left corner of each panel so it never collides with the shared legend.
            ax.text(0.035, 0.955, short_fam[fam], transform=ax.transAxes,
                    ha="left", va="top", fontsize=23, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.85))
            ax.set_xlabel("severity", fontsize=22)
            ax.set_ylabel(f"{metric} (x clean)", fontsize=21)
            ax.tick_params(axis="both", labelsize=20)
            ax.grid(alpha=0.3)
    # one shared legend for the whole figure
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995),
               ncol=2, fontsize=26, frameon=True)
    # No suptitle / no on-image caption — the descriptive caption lives in the document (LaTeX).
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT / "fig_degradation_curves.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def _plot_absolute(df: pd.DataFrame):
    """Absolute MASE/WQL vs severity — the accuracy counterpart of fig_degradation_curves.png.

    Same 2x6 layout and the same severities, one-to-one with the degradation figure; the only
    difference is the y-axis. Where fig_degradation_curves.png divides each model by its OWN clean
    baseline (so both curves start at 1.0 and are NOT on a shared scale), this plots the raw
    gmean-across-datasets metric, so the two models sit on ONE directly-comparable scale and their
    differing clean baselines are visible (dotted horizontal reference line per model, colour-matched).

    The two figures answer different questions and can even show opposite winners: the ratio figure
    asks "how much does each model degrade from itself" (robustness — favours the model with the
    higher clean error, since its denominator is larger), while this one asks "which model is actually
    more accurate on the corrupted input" (deployment). Chronos-2's lower clean baseline is exactly
    why it can look less robust in the ratio view yet remain more accurate in absolute terms.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    short_fam = {"spikes_intensity": "Spikes (intensity)", "spikes_density": "Spikes (density)",
                 "drift": "Drift (ramp)", "drift_step": "Level shift (segment)",
                 "gap": "Missing (random)", "gap_boundary": "Missing (boundary)"}
    colors = {"chronos-2": "C3", "chronos-t5": "C0"}
    disp = {"chronos-2": "Chronos-2", "chronos-t5": "Chronos-T5"}
    nfam = len(FAMILIES)
    fig, axes = plt.subplots(2, nfam, figsize=(5.2 * nfam, 9.2))
    for j, fam in enumerate(FAMILIES):
        sevs = SEVERITIES[fam]
        for i, (metric, col) in enumerate([("MASE", "MASE"), ("WQL", "WQL")]):
            ax = axes[i, j]
            for model in ["chronos-2", "chronos-t5"]:
                y = [_agg_degr(df, model, fam, s, col) for s in sevs]   # gmean of the ABSOLUTE metric
                ax.plot(sevs, y, "o-", color=colors[model], lw=2.4, ms=8, label=disp[model])
                base = _agg_degr(df, model, "clean", 0.0, col)          # each model's clean baseline
                ax.axhline(base, color=colors[model], ls=":", lw=1.4, alpha=0.6)
            # Title in the top-left corner so it never collides with the shared legend.
            ax.text(0.035, 0.955, short_fam[fam], transform=ax.transAxes,
                    ha="left", va="top", fontsize=23, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.85))
            ax.set_xlabel("severity", fontsize=22)
            ax.set_ylabel(f"{metric} (absolute)", fontsize=21)
            ax.tick_params(axis="both", labelsize=20)
            ax.grid(alpha=0.3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995),
               ncol=2, fontsize=26, frameon=True)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT / "fig_absolute_curves.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def _load_pipes():
    """Load both model pipelines once (reused across datasets for the example figures)."""
    return {label: (BaseChronosPipeline.from_pretrained(
                mid, device_map="cuda" if torch.cuda.is_available() else "cpu",
                torch_dtype=R2.DTYPE), kind)
            for label, mid, kind in MODELS}


def _plot_examples(df, data, dataset="ercot", pipes=None):
    """Save the per-category example figures for one dataset into results/examples/<dataset>/.

    One figure per corruption category (spikes intensity, spikes density, drift ramp, level shift,
    missing-random, missing-boundary); each has 4 panels = clean context + 3 increasing severities, with both
    models' forecasts (Chronos-2 solid red, Chronos-T5 dashed blue) over the held-out actual.
    All panels use a fixed recent window so the forecast horizon stays clearly visible; the
    localised corruptions (level shift, missing-random) are placed inside that window but before
    the last `keep` steps, while `missing_boundary` is pinned to the context|horizon junction.

    If `pipes` is given (from `_load_pipes()`) it is reused and left loaded; otherwise the two
    models are loaded for this call and freed at the end.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = data[dataset]
    horizon = d["horizon"]
    clean_ctxs = d["conds"][("clean", 0.0)]
    si = int(max(range(len(clean_ctxs)), key=lambda i: np.isfinite(clean_ctxs[i]).sum()))
    start = d["starts"][si]
    label_arr = np.asarray(list(d["test_data"].label)[si]["target"], dtype=np.float32)
    clean_ctx = np.asarray(clean_ctxs[si], dtype=np.float32)
    n = len(clean_ctx)
    scale = P.robust_scale(clean_ctx)
    clean_med = float(np.median(clean_ctx[np.isfinite(clean_ctx)]))   # signal baseline (corruption-free)
    W = min(n, max(6 * horizon, 150))            # fixed recent window -> horizon always visible
    keep = min(2 * horizon, W // 3)              # immediate pre-origin steps kept intact

    # one-series CLEAN test instance for si -> per-panel per-model MASE/WQL: forecast from the
    # CORRUPTED context is scored against the CLEAN held-out (MASE scale from the clean context),
    # via the same gluonts pipeline + 9-quantile grid as the aggregate eval.
    _clean_full = np.concatenate([np.asarray(clean_ctx, np.float32), label_arr])
    _, _tt_one = split([{"start": start - n, "target": _clean_full}], offset=-horizon)
    td_one = _tt_one.generate_instances(horizon, windows=1)

    own_pipes = pipes is None
    if own_pipes:
        pipes = _load_pipes()
    STYLE = {"chronos-2":  dict(color="C3", ls="-",  lw=2.2, zorder=6),      # C2 univariate = red solid
             "chronos-t5": dict(color="C0", ls=":",  lw=1.8, zorder=7)}      # T5 = blue dotted
    CL_STYLE = dict(color="C4", ls="--", lw=2.2, zorder=6)                   # C2 cross-learning = purple dashed
    ds_idx = _ALL_DS_IDX.get(dataset, 0)   # stable per-dataset seed (full Benchmark II order)

    def _gen(category, sev, rng):
        """Full corrupted context for one (category, severity). Magnitudes are x robust scale;
        missing severities are a fraction of the displayed window. (Spikes are handled separately
        below so intensity and density can be varied as clean controlled variables.)"""
        x = np.array(clean_ctx, dtype=np.float32, copy=True)
        if category == "clean":
            return x
        if category == "drift_ramp":
            return P.add_drift(clean_ctx, rng, slope=sev)          # ramp reaches the origin -> harmful
        reg0, reg1 = n - W, n - keep                                # "visible past" region (recent tail kept)
        span = max(1, reg1 - reg0)
        if category == "level_shift":
            seg = min(max(1, int(round(0.30 * W))), span)
            s = reg0 + (span - seg) // 2
            x[s:s + seg] = x[s:s + seg] + np.float32(sev * scale)
        elif category == "missing_random":
            g = min(max(1, int(round(sev * span))), span)
            s = reg0 + int(rng.integers(0, span - g + 1))          # random position in the past
            x[s:s + g] = np.nan
        elif category == "missing_boundary":
            g = min(max(1, int(round(sev * W))), W - 1)
            x[n - g:] = np.nan                                     # pinned to context|horizon junction
        return x

    def _draw(ax, ctx_full, title):
        ctx_show = ctx_full[-W:]
        xc = np.arange(-len(ctx_show), 0); xf = np.arange(0, horizon)
        # Shade EACH contiguous NaN run separately (the series may carry a few real missing
        # values besides the injected gap — a single span would merge them into one block).
        nan_idx = np.flatnonzero(~np.isfinite(ctx_show))
        if nan_idx.size:
            runs = np.split(nan_idx, np.where(np.diff(nan_idx) > 1)[0] + 1)
            for run in runs:
                ax.axvspan(xc[run[0]] - 0.5, xc[run[-1]] + 0.5, color="0.82", alpha=0.7, zorder=0)
            big = max(runs, key=len)
            if len(big) >= max(4, horizon // 4):   # label only a substantial (injected) gap
                ax.text(xc[big[len(big) // 2]], 0.82, "missing (NaN)",
                        transform=ax.get_xaxis_transform(), ha="center", va="top",
                        fontsize=13, color="0.35", style="italic")
        # clean group context (the neighbors cross-learning leans on). Cross-learning relates series
        # by SHAPE (InstanceNorm removes each series' own level/scale), so we z-normalize each neighbor
        # and re-anchor it to the CLEAN TARGET's shown-window mean/std -> the neighbors overlay the
        # target's line, showing the shared pattern the model exploits (not their raw, differing levels).
        cshow = np.asarray(clean_ctx, np.float32)[-len(ctx_show):]
        _f = cshow[np.isfinite(cshow)]
        if _f.size:
            mu_t, sd_t = float(np.mean(_f)), float(np.std(_f)) + 1e-8
            shown = 0
            for j, g in enumerate(clean_ctxs):
                if j == si or shown >= 6:
                    continue
                gg = np.asarray(g, np.float32)[-len(ctx_show):]
                ax.plot(np.arange(-len(gg), 0), _znorm(gg) * sd_t + mu_t, color="0.72", lw=0.6,
                        alpha=0.5, zorder=1, label="group context (shape-aligned)" if shown == 0 else None)
                shown += 1
        ax.plot(xc, ctx_show, color="0.45", lw=1.0, label="target context (corrupted)")
        ax.plot(xf, label_arr, color="C2", lw=2.0, zorder=5, label="actual (held-out)")
        focus = [label_arr]
        metrics = []
        for label, (pipe, kind) in pipes.items():
            fc = forecast(pipe, [ctx_full], [start], horizon, kind)[0]
            p10, p50, p90 = fc.forecast_array[0], fc.forecast_array[4], fc.forecast_array[8]
            st = STYLE[label]
            mlabel = "chronos-2 uni" if label == "chronos-2" else label   # C2 has two modes now
            ax.fill_between(xf, p10, p90, color=st["color"], alpha=0.15, zorder=2)
            ax.plot(xf, p50, label=f"{mlabel} p50", **st)
            focus.append(np.asarray(p50, dtype=np.float32))
            mase, wql = R2.evaluate([fc], td_one)          # corrupted-context forecast vs clean held-out
            metrics.append((mlabel, st["color"], mase, wql))
        # Chronos-2 CROSS-LEARNING: same corrupted target, but forecast inside its CLEAN group,
        # so clean neighbors can (potentially) rescue it. Same clean held-out for scoring.
        fc_cl = forecast_cl_one(pipes["chronos-2"][0], clean_ctxs, si, ctx_full, start, horizon)
        p10, p50, p90 = fc_cl.forecast_array[0], fc_cl.forecast_array[4], fc_cl.forecast_array[8]
        ax.fill_between(xf, p10, p90, color=CL_STYLE["color"], alpha=0.15, zorder=3)
        ax.plot(xf, p50, label="chronos-2 CL p50", **CL_STYLE)
        focus.append(np.asarray(p50, dtype=np.float32))
        mase, wql = R2.evaluate([fc_cl], td_one)
        metrics.append(("chronos-2 CL", CL_STYLE["color"], mase, wql))
        allf = np.concatenate(focus)
        lo, hi = float(np.nanmin(allf)), float(np.nanmax(allf))
        # Anchor the scale to the clean SIGNAL baseline (median +/- 3*MAD), NOT the corrupted
        # context, so spikes / the level-shift segment / the ramp clip instead of compressing the
        # forecast-vs-actual comparison. The forecast (p50) still extends the range when it is
        # genuinely dragged off (drift), so that case stays visible.
        lo = min(lo, clean_med - 3 * scale); hi = max(hi, clean_med + 3 * scale)
        pad = 0.15 * (hi - lo + 1e-9)
        lo, hi = lo - pad, hi + pad
        ax.set_ylim(lo, hi)
        ax.axvline(0, color="grey", ls="--", lw=0.8)
        # per-model MASE/WQL for THIS panel, in the LOWER-left corner (clears the top-center legend
        # and the "missing (NaN)" label), color-matched to each model's forecast. inf/nan -> "n/a"
        # (MASE inf = flat context so seasonal-naive scale is 0; WQL nan = all-zero held-out actual).
        def _fmt(v):
            return f"{v:.3f}" if np.isfinite(v) else "n/a"
        for i, (label, color, mase, wql) in enumerate(metrics):
            ax.text(0.012, 0.035 + (len(metrics) - 1 - i) * 0.115,
                    f"{label}:  MASE {_fmt(mase)}   WQL {_fmt(wql)}",
                    transform=ax.transAxes, ha="left", va="bottom", fontsize=13, color=color, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, alpha=0.85), zorder=10)
        ax.set_title(title, fontsize=16)
        ax.set_xlabel("steps from forecast origin", fontsize=14); ax.tick_params(labelsize=13)
        return lo, hi                              # per-panel focus range (for optional y-sharing)

    finite = np.flatnonzero(np.isfinite(clean_ctx))   # spikes land only on observed points
    outdir = OUT / "examples" / dataset
    outdir.mkdir(parents=True, exist_ok=True)
    (OUT / f"fig_examples_{dataset}.png").unlink(missing_ok=True)   # drop the old combined figure
    for cat_i, (cat, nice, sevs, fmt) in enumerate(EXAMPLE_FIGS):
        # 4x1 stacked, full-width panels so the (right-side) forecast horizon is not cramped
        # on long-but-narrow series like nn5.
        fig, axes = plt.subplots(4, 1, figsize=(15, 13.5))
        axx = axes.ravel()
        ranges = [_draw(axx[0], _gen("clean", 0.0, None), "clean context")]
        if cat == "spikes_intensity":
            # CONTROL: fixed positions + signs (density SPIKE_FIX_FRAC); only magnitude grows.
            rng = np.random.default_rng(np.random.SeedSequence([SEED, 7000 + ds_idx, cat_i]))
            k = max(1, int(round(SPIKE_FIX_FRAC * finite.size)))
            idx = rng.choice(finite, size=min(k, finite.size), replace=False)
            sgn = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=idx.size)
            for kk, m in enumerate(sevs):
                x = np.array(clean_ctx, dtype=np.float32, copy=True)
                x[idx] = x[idx] + sgn * np.float32(m * scale)
                ranges.append(_draw(axx[1 + kk], x, f"intensity {fmt.format(m)}  (density {SPIKE_FIX_FRAC:.0%})"))
        elif cat == "spikes_density":
            # CONTROL: fixed magnitude SPIKE_FIX_MAG; nested spike sets (more points added).
            rng = np.random.default_rng(np.random.SeedSequence([SEED, 7000 + ds_idx, cat_i]))
            kmax = max(1, int(round(max(sevs) * finite.size)))
            idx_all = rng.choice(finite, size=min(kmax, finite.size), replace=False)
            sgn_all = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=idx_all.size)
            for kk, frac in enumerate(sevs):
                kf = min(max(1, int(round(frac * finite.size))), idx_all.size)
                sub, sg = idx_all[:kf], sgn_all[:kf]
                x = np.array(clean_ctx, dtype=np.float32, copy=True)
                x[sub] = x[sub] + sg * np.float32(SPIKE_FIX_MAG * scale)
                ranges.append(_draw(axx[1 + kk], x, f"density {fmt.format(frac)}  (x{SPIKE_FIX_MAG:g})"))
        else:
            for k, sev in enumerate(sevs):
                rng = np.random.default_rng(np.random.SeedSequence([SEED, 7000 + ds_idx, cat_i, int(round(sev * 1000))]))
                ranges.append(_draw(axx[1 + k], _gen(cat, sev, rng), f"severity {fmt.format(sev)}"))
        # Share one y-axis across the 4 panels EXCEPT for drift_ramp, where the forecast itself
        # scales with severity (a shared axis would crush the low-severity panels). For the other
        # families the forecast envelope is stable across severities, so a shared axis (driven by
        # actual+forecast, with the corruption clipping) makes the panels directly comparable.
        if cat != "drift_ramp":
            ylo = min(r[0] for r in ranges); yhi = max(r[1] for r in ranges)
            for ax in axx:
                ax.set_ylim(ylo, yhi)
        handles, labels = axx[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.978),
                   ncol=4, fontsize=15, frameon=True)
        fig.suptitle(f"{dataset} — {nice}: clean vs increasing severity "
                     f"(forecast vs clean held-out actual)", fontsize=17, y=0.997)
        fig.tight_layout(rect=[0, 0, 1, 0.955])
        fig.savefig(outdir / f"{cat}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
    print(f"  saved {len(EXAMPLE_FIGS)} example figures -> {outdir}")
    if own_pipes:
        for _label, (pipe, _) in pipes.items():
            del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def report_only():
    """Rewrite EDGE_CASE_REPORT.md + fig_degradation_curves.png from the saved CSV.

    Lets the report narrative / degradation curve be regenerated without re-running the
    two models. The per-series example figure needs live inference, so it is left as-is.
    """
    df = pd.read_csv(OUT / "edge_case_results.csv")
    if "MASE_degr" not in df.columns:
        df = _add_degradation(df)
    _write_report(df)
    _plot_degradation(df)
    _plot_absolute(df)
    print(f"Regenerated report + degradation curve -> {OUT}")


def examples_only(datasets=EXAMPLE_DATASETS):
    """Regenerate fig_examples_<dataset>.png for each dataset (needs live inference).

    Rebuilds just the corrupted contexts for the requested datasets (no models for that
    step) and loads the two pipelines once, reused across all of them.
    """
    df = pd.read_csv(OUT / "edge_case_results.csv")
    horizons = dict(EDGE_DATASETS)
    data = {}
    for dataset in datasets:
        horizon = horizons[dataset]
        test_data, contexts, starts = build_dataset(dataset, horizon)
        conds = {(fam, sev): perturb_contexts(dataset, contexts, fam, sev) for (fam, sev) in CONDITIONS}
        data[dataset] = dict(horizon=horizon, test_data=test_data, starts=starts,
                             n_series=len(contexts), conds=conds)
    pipes = _load_pipes()
    for dataset in datasets:
        _plot_examples(df, data, dataset, pipes=pipes)
        print(f"Regenerated example figures -> {OUT / 'examples' / dataset}")
    del pipes
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def examples_all():
    """Render the per-series example figures for ALL Benchmark II datasets (25), each into its
    own folder results/examples/<dataset>/. Only the clean series is needed (corruptions are
    generated per-figure), so no metric sweep / CSV is required; both models are loaded once and
    reused, datasets are built one at a time, and a failing dataset is skipped (not fatal)."""
    pipes = _load_pipes()
    ok, failed = [], []
    for k, (config, horizon) in enumerate(BENCHMARK_II, 1):
        print(f"[{k}/{len(BENCHMARK_II)}] {config} (H={horizon}) ...", flush=True)
        try:
            test_data, contexts, starts = build_dataset(config, horizon)
            data = {config: dict(horizon=horizon, test_data=test_data, starts=starts,
                                 n_series=len(contexts),
                                 conds={("clean", 0.0): contexts})}
            _plot_examples(None, data, config, pipes=pipes)
            ok.append(config)
        except Exception as e:  # noqa: BLE001 - keep going across datasets
            failed.append((config, f"{type(e).__name__}: {e}"))
            print(f"  [skip] {config}: {type(e).__name__}: {e}", flush=True)
    del pipes
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"\nExample figures: {len(ok)}/{len(BENCHMARK_II)} datasets -> {OUT / 'examples'}")
    if failed:
        print("Skipped:", failed)


def add_family(fam):
    """Incrementally compute ONE corruption family over all datasets and merge into the existing
    CSV (without recomputing the other families), then rewrite the report + degradation curve."""
    if fam not in SEVERITIES:
        raise ValueError(f"unknown family {fam!r}; known: {list(SEVERITIES)}")
    cuda = torch.cuda.is_available()
    base_cols = ["dataset", "model", "family", "severity", "MASE", "WQL", "n_series", "latency_s"]
    old = pd.read_csv(OUT / "edge_case_results.csv")[base_cols]
    old = old[old["family"] != fam]                        # idempotent: drop any prior rows for fam
    pipes = _load_pipes()
    rows = []
    for di, (config, horizon) in enumerate(EDGE_DATASETS, 1):
        test_data, contexts, starts = build_dataset(config, horizon)
        print(f"[{di}/{len(EDGE_DATASETS)}] {config} (n={len(contexts)}) — {fam}", flush=True)
        for sev in SEVERITIES[fam]:
            pc = perturb_contexts(config, contexts, fam, sev)
            for label, (pipe, kind) in pipes.items():
                t0 = time.perf_counter()
                fcs = forecast(pipe, pc, starts, horizon, kind)
                if cuda:
                    torch.cuda.synchronize()
                mase, wql = R2.evaluate(fcs, test_data)
                rows.append({"dataset": config, "model": label, "family": fam, "severity": sev,
                             "MASE": mase, "WQL": wql, "n_series": len(contexts),
                             "latency_s": round(time.perf_counter() - t0, 3)})
        del contexts, test_data, starts
        if cuda:
            torch.cuda.empty_cache()        # avoid fragmentation building up across datasets
    del pipes
    if cuda:
        torch.cuda.empty_cache()
    df = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
    df = _add_degradation(df)
    df.to_csv(OUT / "edge_case_results.csv", index=False)
    _write_report(df)
    _plot_degradation(df)
    _plot_absolute(df)
    print(f"\nAdded family '{fam}' -> CSV + report + degradation curve regenerated in {OUT}")


if __name__ == "__main__":
    if "--report-only" in sys.argv:
        report_only()
    elif "--add-family" in sys.argv:
        add_family(sys.argv[sys.argv.index("--add-family") + 1])
    elif "--examples-all" in sys.argv:
        examples_all()
    elif "--examples-only" in sys.argv:
        examples_only()
    else:
        run()
