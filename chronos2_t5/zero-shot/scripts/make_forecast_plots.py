"""Per-dataset forecast plots, Chronos-2 vs Chronos-T5, at each dataset's native horizon.

For every Benchmark II dataset it picks one representative series and forecasts the
dataset's own benchmark horizon (paper Table 3). Chronos-T5 is univariate-only; Chronos-2
is shown in BOTH modes, so its folder carries univariate AND cross-learning:

    forecasts/chronos2/univariate/       Chronos-2, each series forecast alone (like-for-like vs T5)
    forecasts/chronos2/cross_learning/   Chronos-2 cross-learning (p50 + 10-90%), with the
                                         univariate p50 overlaid dashed so you see what the
                                         group changed  (mirrors plot_crosslearning_io's output panel)
    forecasts/chronos_t5/                Chronos-T5 (univariate only)
plus a 5x5 overview grid per folder.

The cross-learning forecast is EVAL-FAITHFUL: it is the forecast Chronos-2 produces for the
representative series *because it sits in its cross-learning group*. Groups == batches
(batch_size=CROSS_LEARNING_BATCH) and series in different batches never mix, so we forecast
only the target's positional batch -- mathematically identical to the full-dataset eval call,
far cheaper. (See plot_crosslearning_io.py for the mechanism drawn in full.)

Reuses the sibling Chronos2 project's loaders (no duplicate dataset code). Needs a GPU.
Run:  python make_forecast_plots.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import datasets as hfds

HERE = Path(__file__).resolve().parent          # .../zero-shot/scripts
ZS = HERE.parent                                 # .../zero-shot (output root)
ROOT = HERE.parents[2]                            # repo root
sys.path.insert(0, str(ROOT / "Chronos2" / "src"))                # config, datasets_lib (src root)
sys.path.insert(0, str(ROOT / "Chronos2" / "src" / "zero_shot"))  # run_zeroshot_chronos2.py (shared harness)
import run_zeroshot_chronos2 as R2          # to_gluonts_univariate / HF_REPO / MAX_SERIES / DTYPE / MODEL_ID / CROSS_LEARNING_BATCH
import datasets_lib as D                    # BENCHMARK_II (dataset, native horizon)

from chronos import BaseChronosPipeline
from gluonts.dataset.split import split

hfds.logging.set_verbosity_error(); hfds.disable_progress_bars()
Qs = R2.QUANTILES                                          # 9-level grid -> eval-faithful WQL
DRAW = [Qs.index(0.1), Qs.index(0.5), Qs.index(0.9)]       # indices used to draw the band + median


def _arr(a):
    a = a.cpu().numpy() if hasattr(a, "cpu") else np.asarray(a)
    return a[0] if a.ndim == 3 else a                              # (H, 9) full quantile grid


N_GROUP_SHOW = 10          # how many of the target's cross-learning group series to overlay as context


def _znorm(x):
    x = np.asarray(x, np.float32)
    m, s = np.nanmean(x), np.nanstd(x)
    return (x - m) / (s + 1e-8)


def pick_series_index(gts):
    """Representative series = the one with the most finite points (eval order preserved)."""
    return max(range(len(gts)), key=lambda i: int(np.isfinite(gts[i]["target"]).sum()))


def series_metrics(entry, H, q_full):
    """Per-series (MASE, WQL) for the plotted series via the eval's gluonts pipeline."""
    one = [{"start": entry["start"], "target": entry["target"]}]
    _, tt = split(one, offset=-H)
    td = tt.generate_instances(H, windows=1)
    start = entry["start"] + (len(entry["target"]) - H)               # first forecast step
    return R2.evaluate([R2._quantile_forecast(np.asarray(q_full, np.float32), start)], td)


def forecast_uni(pipe, entry, H, kind):
    """Univariate forecast for one series (each model alone)."""
    series = entry["target"]
    ctx = series[:len(series) - H]
    kw = dict(prediction_length=H, quantile_levels=Qs)
    if kind == "t5":
        torch.manual_seed(0); kw["num_samples"] = 20
    else:
        kw["limit_prediction_length"] = False
    qz, _ = pipe.predict_quantiles([torch.tensor(ctx)], **kw)
    a = _arr(qz[0])
    mase, wql = series_metrics(entry, H, a)
    return ctx, series[len(series) - H:], a, mase, wql


def forecast_cl(pipe, targets, tidx, entry, H):
    """Cross-learning forecast for the representative series, IDENTICAL to eval.

    Group == the positional batch (batch_size=B) the target lands in; series in other batches
    never mix, so forecasting just that batch reproduces the target's eval forecast exactly."""
    B = R2.CROSS_LEARNING_BATCH
    g0 = (tidx // B) * B
    grp = targets[g0: g0 + B]                                         # the target's cross-learning group
    ctxs = [t[: len(t) - H] for t in grp]
    qx, _ = pipe.predict_quantiles(ctxs, prediction_length=H, quantile_levels=Qs,
                                   cross_learning=True, batch_size=B, limit_prediction_length=False)
    local = tidx - g0
    a = _arr(qx[local])                                               # target's forecast (local index)
    mase, wql = series_metrics(entry, H, a)
    others = [ctxs[k] for k in range(len(ctxs)) if k != local][:N_GROUP_SHOW]   # group context to overlay
    return a, mase, wql, others


def draw(ax, name, ctx, fut, q, H, mase=None, wql=None, ylim=None):
    show = ctx[-max(3 * H, 60):]
    xc = np.arange(-len(show), 0); xf = np.arange(0, H)
    p10, p50, p90 = q[:, DRAW[0]], q[:, DRAW[1]], q[:, DRAW[2]]
    ax.plot(xc, show, color="C0", lw=1.0, label="history (actual)")
    ax.plot(xf, fut, color="C2", lw=1.6, label="actual (held-out)")
    ax.plot(xf, p50, color="C3", lw=1.6, label="forecast (p50)")
    ax.fill_between(xf, p10, p90, color="C3", alpha=0.25, label="forecast 10-90%")
    ax.axvline(0, color="grey", ls="--", lw=0.8)
    ax.set_xlim(-len(show), H)                      # shared x (both models: same series/horizon)
    if ylim is not None:
        ax.set_ylim(*ylim)
    # score of THIS drawn series (all folders draw the same series -> comparable across models).
    sc = f"\nthis series: MASE={mase:.3f} WQL={wql:.3f}" if mase is not None else ""
    ax.set_title(f"{name}  (H={H}){sc}", fontsize=8); ax.tick_params(labelsize=8)


def draw_cl(ax, name, ctx, fut, q_cl, q_uni, H, mase_cl=None, wql_cl=None,
            mase_uni=None, wql_uni=None, group=None, ylim=None):
    """Cross-learning vs univariate for the target, plus the target's cross-learning GROUP overlaid
    (grey, shape-aligned) as the in-context that drove cross-learning -- all in one panel."""
    show = ctx[-max(3 * H, 60):]
    xc = np.arange(-len(show), 0); xf = np.arange(0, H)
    # in-context group series cross-learning attended to: z-normalized then rescaled to the target's
    # shown-window mean/std, so their SHAPES sit next to the target on its own (original-unit) axis.
    if group:
        fs = show[np.isfinite(show)]
        if fs.size:
            m_t, s_t = float(np.mean(fs)), float(np.std(fs)) + 1e-8
            for k, g in enumerate(group):
                disp = _znorm(np.asarray(g, np.float32)[-len(show):]) * s_t + m_t
                ax.plot(np.arange(-len(disp), 0), disp, color="grey", lw=0.6, alpha=0.30, zorder=1,
                        label="group context (z-norm, shape-aligned)" if k == 0 else None)
    ax.plot(xc, show, color="C0", lw=1.0, label="history (actual)")
    ax.plot(xf, fut, color="C2", lw=1.6, label="actual (held-out)")
    # univariate = red SOLID (+ red band); cross-learning = purple DASHED (+ purple band).
    # both 10-90% bands so the uncertainty width is comparable, not just the median.
    ax.fill_between(xf, q_uni[:, DRAW[0]], q_uni[:, DRAW[2]], color="C3", alpha=0.15, label="univariate 10-90%")
    ax.fill_between(xf, q_cl[:, DRAW[0]], q_cl[:, DRAW[2]], color="C4", alpha=0.18, label="cross-learning 10-90%")
    ax.plot(xf, q_uni[:, DRAW[1]], color="C3", lw=1.7, label="univariate (p50)")
    ax.plot(xf, q_cl[:, DRAW[1]], color="C4", ls="--", lw=1.9, label="cross-learning (p50)")
    ax.axvline(0, color="grey", ls="--", lw=0.8)
    ax.set_xlim(-len(show), H)
    if ylim is not None:
        ax.set_ylim(*ylim)
    # scores of THIS drawn series for both modes (same series is drawn for T5 too -> comparable).
    sc = (f"\nthis series — uni MASE={mase_uni:.3f} WQL={wql_uni:.3f}  |  "
          f"CL MASE={mase_cl:.3f} WQL={wql_cl:.3f}" if mase_cl is not None else "")
    ax.set_title(f"{name}  (H={H}){sc}", fontsize=7.5); ax.tick_params(labelsize=8)


def shared_ylim(ctx, fut, qs, H):
    """Per-dataset y-range shared by the plots that should be directly comparable.
    Built from the held-out actual, each given forecast's 10-90% band, and the shown
    history's 1-99 pctile (so a history spike doesn't dominate the scale)."""
    show = ctx[-max(3 * H, 60):]
    vals = [np.asarray(fut, np.float32).ravel()]
    s = show[np.isfinite(show)]
    if s.size:
        vals.append(np.percentile(s, [1, 99]))
    for q in qs:
        vals.append(q[:, DRAW[0]]); vals.append(q[:, DRAW[2]])   # band lo / hi (p10/p90) of each curve
    allv = np.concatenate([np.ravel(v) for v in vals])
    lo, hi = float(np.nanmin(allv)), float(np.nanmax(allv))
    pad = 0.08 * (hi - lo + 1e-9)
    return lo - pad, hi + pad


def render_folder(sub, title, items, drawer, ylims):
    """Save one PNG per dataset + a 5x5 overview grid, using `drawer(ax, *item, ylim=...)`."""
    sub.mkdir(parents=True, exist_ok=True)
    for it in items:
        cfg = it[0]
        f1, a1 = plt.subplots(figsize=(7.2, 3.2)); drawer(a1, *it, ylim=ylims[cfg])
        a1.set_xlabel("steps from forecast origin"); a1.legend(fontsize=8, loc="upper left")
        f1.tight_layout(); f1.savefig(sub / f"{cfg}.png", dpi=130, bbox_inches="tight"); plt.close(f1)
    fig, axes = plt.subplots(5, 5, figsize=(23, 13))
    for ax, it in zip(axes.ravel(), items): drawer(ax, *it, ylim=ylims[it[0]])
    for ax in axes.ravel()[len(items):]: ax.axis("off")
    h, l = axes.ravel()[0].get_legend_handles_labels()
    fig.suptitle(f"{title} — representative series per Benchmark II dataset, native horizon", fontsize=15, y=0.999)
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=4, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.925])
    fig.savefig(sub / "_overview_grid.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"saved {len(items)} plots -> {sub}")


def main():
    # Per-dataset representative series (+ full targets/index for the cross-learning group).
    series = {}
    for cfg, H in D.BENCHMARK_II:
        ds = hfds.load_dataset(R2.HF_REPO, cfg, split="train"); ds.set_format("numpy")
        gts = R2.to_gluonts_univariate(ds, R2.MAX_SERIES)
        tidx = pick_series_index(gts)
        targets = [np.asarray(e["target"], np.float32) for e in gts]
        entry = {"start": gts[tidx]["start"], "target": targets[tidx]}
        series[cfg] = dict(entry=entry, H=H, targets=targets, tidx=tidx)

    # Pass 1: forecast (one model in memory at a time). C2 -> univariate + cross-learning; T5 -> univariate.
    c2_uni, c2_cl, t5_uni = {}, {}, {}
    pipe = BaseChronosPipeline.from_pretrained(R2.MODEL_ID, device_map="cuda", torch_dtype=R2.DTYPE)
    for cfg, s in series.items():
        c2_uni[cfg] = forecast_uni(pipe, s["entry"], s["H"], "c2")
        c2_cl[cfg] = forecast_cl(pipe, s["targets"], s["tidx"], s["entry"], s["H"])
    del pipe
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-t5-small", device_map="cuda", torch_dtype=R2.DTYPE)
    for cfg, s in series.items():
        t5_uni[cfg] = forecast_uni(pipe, s["entry"], s["H"], "t5")
    del pipe
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Per-dataset y-range shared by ALL three folders (C2-uni, C2-CL, T5) so they line up for
    # direct comparison. Built from the univariate C2 & T5 bands + target (not the CL band), so the
    # CL band may occasionally extend past it -- intended, to keep the shared scale.
    ylims_uni = {cfg: shared_ylim(c2_uni[cfg][0], c2_uni[cfg][1], [c2_uni[cfg][2], t5_uni[cfg][2]], s["H"])
                 for cfg, s in series.items()}

    # Pass 2: render each folder. Titles carry the score of THE DRAWN series (the same representative
    # series is drawn in all three folders, so uni / CL / T5 numbers are directly comparable). The
    # dataset-level verdict lives in benchmark/ (skill score + win rate over all series).
    #   items for draw()    : (cfg, ctx, fut, q, H, mase, wql)
    #   items for draw_cl() : (cfg, ctx, fut, q_cl, q_uni, H, mase_cl, wql_cl, mase_uni, wql_uni, group)
    uni_c2_items = [(cfg, *c2_uni[cfg][:2], c2_uni[cfg][2], s["H"], c2_uni[cfg][3], c2_uni[cfg][4])
                    for cfg, s in series.items()]
    uni_t5_items = [(cfg, *t5_uni[cfg][:2], t5_uni[cfg][2], s["H"], t5_uni[cfg][3], t5_uni[cfg][4])
                    for cfg, s in series.items()]
    cl_items = [(cfg, c2_uni[cfg][0], c2_uni[cfg][1], c2_cl[cfg][0], c2_uni[cfg][2], s["H"],
                 c2_cl[cfg][1], c2_cl[cfg][2], c2_uni[cfg][3], c2_uni[cfg][4], c2_cl[cfg][3])
                for cfg, s in series.items()]

    render_folder(ZS / "forecasts" / "chronos2" / "univariate",
                  "Chronos-2 forecasts (univariate)", uni_c2_items, draw, ylims_uni)
    # cross-learning shares the univariate/T5 y-range per dataset so all three folders are directly comparable
    render_folder(ZS / "forecasts" / "chronos2" / "cross_learning",
                  "Chronos-2 forecasts (cross-learning; univariate p50 dashed)", cl_items, draw_cl, ylims_uni)
    render_folder(ZS / "forecasts" / "chronos_t5",
                  "Chronos-T5 forecasts (univariate)", uni_t5_items, draw, ylims_uni)


if __name__ == "__main__":
    main()