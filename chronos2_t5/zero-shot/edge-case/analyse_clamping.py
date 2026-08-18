"""E1 analysis -- does quantiser clamping explain Chronos-T5's non-monotonic degradation?

Reads the two measurement CSVs produced elsewhere in this folder and asks three questions,
in increasing order of how hard they are to explain away:

  Q1 (aggregate)    Does the clamped fraction rise where the degradation curve recovers?
                    Suggestive, but a single coincidence across one aggregated curve.

  Q2 (per-dataset)  Are the datasets that clamp the same datasets whose curve recovers?
                    25 independent observations instead of one. This is the real test:
                    the mechanism predicts a POSITIVE association between how much a
                    dataset clamps and how much its degradation curve falls back.

  Q3 (re-axis)      If degradation is plotted against REALISED excursion (what survives
                    the bounded token grid) rather than NOMINAL severity (what was
                    injected), does the relationship become monotone? Compared as two
                    Spearman rho's over the identical set of (dataset, severity) cells,
                    so nothing but the x-axis changes.

Nothing here re-runs a model: it joins `clamping_measurements.csv` (tokeniser arithmetic)
onto `edge_case_results.csv` (the existing sweep).

Outputs (results/):
  fig_clamping_mechanism.png   three panels, one per question
  CLAMPING_ANALYSIS.md         the numbers, written so the paper can quote them directly

Usage:  python analyse_clamping.py [--family spikes_intensity] [--metric MASE]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gmean, spearmanr

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
MODEL = "chronos-t5"          # the clamping model; Chronos-2 has no value tokeniser


def _arg(flag: str, default: str) -> str:
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def load(family: str, metric: str):
    clamp = pd.read_csv(OUT / "clamping_measurements.csv")
    degr = pd.read_csv(OUT / "edge_case_results.csv")

    c = clamp[clamp.family == family].copy()
    d = degr[(degr.model == MODEL) & (degr.family == family)].copy()
    col = f"{metric}_degr"
    # Keep only the degradation column: `n_series` exists in both frames and the sweep's
    # copy is the same per-dataset cap, so carrying it twice just creates suffixed columns.
    d = d[["dataset", "severity", col]].rename(columns={col: "degr"})

    m = c.merge(d, on=["dataset", "severity"], how="inner")
    m = m[np.isfinite(m.degr) & (m.degr > 0)]
    if m.empty:
        raise SystemExit(f"no overlapping cells for family={family!r} metric={metric!r}")
    return m


def q1_aggregate(m: pd.DataFrame) -> pd.DataFrame:
    """Degradation (gmean across datasets, as the sweep reports it) vs clamped fraction."""
    g = m.groupby("severity")
    return pd.DataFrame({
        "degr_gmean": g.degr.apply(lambda v: gmean(v[v > 0])),
        "clamped_frac": g.clamped_frac.mean(),
        "series_any_clamped": g.series_any_clamped.mean(),
        "nominal_p99": g.nominal_p99.mean(),
        "realized_p99": g.realized_p99.mean(),
        "scale_mean": g.scale_mean.mean(),
    }).reset_index()


def q2_per_dataset(m: pd.DataFrame) -> pd.DataFrame:
    """Per dataset: how much does the curve fall back, and how much does it clamp?

    recovery = peak degradation / degradation at the largest severity. >1 means the curve
    came back down after peaking -- the non-monotonicity we are trying to explain.
    """
    rows = []
    for ds, g in m.groupby("dataset"):
        g = g.sort_values("severity")
        peak_i = g.degr.idxmax()
        end = g.iloc[-1]
        rows.append({
            "dataset": ds,
            "n_series": int(g.n_series.iloc[0]),
            "peak_degr": float(g.degr.max()),
            "peak_severity": float(g.loc[peak_i, "severity"]),
            "end_degr": float(end.degr),
            "end_severity": float(end.severity),
            "recovery": float(g.degr.max() / end.degr),
            "max_clamped_frac": float(g.clamped_frac.max()),
            "max_any_clamped": float(g.series_any_clamped.max()),
            "mad_over_meanabs": float(g.nominal_p99.max() / g.severity.max()),  # scale mismatch
        })
    return pd.DataFrame(rows).sort_values("max_clamped_frac", ascending=False)


def q3_reaxis(m: pd.DataFrame):
    """Spearman of degradation against nominal severity vs against realised excursion."""
    rn = spearmanr(m.severity, m.degr)
    rr = spearmanr(m.realized_p99, m.degr)
    return rn, rr


def figure(agg, per_ds, m, family, metric, rn, rr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(19.5, 5.8))

    # -- Q1: the coincidence -------------------------------------------------
    ax = axes[0]
    ax.plot(agg.severity, agg.degr_gmean, "o-", color="C0", lw=2.4, ms=8,
            label=f"Chronos-T5 {metric} (x clean)")
    ax.axhline(1.0, color="grey", ls=":", lw=1.2)
    ax.set_xlabel("nominal severity (x robust scale)", fontsize=13)
    ax.set_ylabel(f"{metric} (x clean)", fontsize=13, color="C0")
    ax.tick_params(axis="y", labelcolor="C0")
    ax2 = ax.twinx()
    ax2.plot(agg.severity, agg.clamped_frac, "s--", color="C1", lw=2.2, ms=7,
             label="clamped fraction")
    ax2.plot(agg.severity, agg.series_any_clamped, "^:", color="C5", lw=1.8, ms=6,
             label="series with any clamping")
    ax2.set_ylabel("fraction clamped", fontsize=13, color="C1")
    ax2.tick_params(axis="y", labelcolor="C1")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=10, loc="upper left")
    ax.set_title("Q1  degradation vs clamping onset", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)

    # -- Q2: the per-dataset test -------------------------------------------
    ax = axes[1]
    x, y = per_ds.max_clamped_frac, per_ds.recovery
    ax.scatter(x, y, s=42, color="C0", alpha=0.85, edgecolor="0.3", zorder=3)
    ax.axhline(1.0, color="grey", ls=":", lw=1.2)
    rho = spearmanr(x, y)
    for _, r in per_ds.nlargest(4, "max_clamped_frac").iterrows():
        ax.annotate(r.dataset, (r.max_clamped_frac, r.recovery), fontsize=8,
                    xytext=(4, 3), textcoords="offset points", color="0.3")
    ax.set_xlabel("max clamped fraction (per dataset)", fontsize=13)
    ax.set_ylabel("recovery  =  peak degr / degr at max severity", fontsize=13)
    ax.set_title(f"Q2  per-dataset  (Spearman rho={rho.statistic:.2f}, p={rho.pvalue:.3f})",
                 fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)

    # -- Q3: re-axis ---------------------------------------------------------
    ax = axes[2]
    ax.scatter(m.severity, m.degr, s=22, color="C0", alpha=0.5,
               label=f"vs nominal  (rho={rn.statistic:.2f})")
    ax.scatter(m.realized_p99, m.degr, s=22, color="C2", alpha=0.6, marker="^",
               label=f"vs realised  (rho={rr.statistic:.2f})")
    ax.axhline(1.0, color="grey", ls=":", lw=1.2)
    ax.set_xscale("log")
    ax.set_xlabel("severity  (nominal, or realised excursion)", fontsize=13)
    ax.set_ylabel(f"{metric} (x clean)", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_title("Q3  same cells, two x-axes", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    p = OUT / f"fig_clamping_mechanism_{family}_{metric}.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return p


def main():
    family = _arg("--family", "spikes_intensity")
    metric = _arg("--metric", "MASE")
    m = load(family, metric)
    agg = q1_aggregate(m)
    per_ds = q2_per_dataset(m)
    rn, rr = q3_reaxis(m)
    rho2 = spearmanr(per_ds.max_clamped_frac, per_ds.recovery)
    fig_path = figure(agg, per_ds, m, family, metric, rn, rr)

    print(f"\n=== {family} / {metric} ===  {len(m)} cells over {m.dataset.nunique()} datasets")
    print("\nQ1  aggregate")
    print(agg.round(4).to_string(index=False))
    print(f"\nQ2  per-dataset: Spearman(max_clamped_frac, recovery) "
          f"rho={rho2.statistic:.3f}  p={rho2.pvalue:.4f}  n={len(per_ds)}")
    print(per_ds.round(4).head(12).to_string(index=False))
    print(f"\nQ3  Spearman(degradation, .) : nominal rho={rn.statistic:.3f} (p={rn.pvalue:.3g}) "
          f"| realised rho={rr.statistic:.3f} (p={rr.pvalue:.3g})")

    lines = [
        f"# E1 -- Clamping analysis ({family}, {metric})", "",
        f"Chronos-T5 = `amazon/chronos-t5-small`; value grid `[-15, +15]` over 4093 bins, ",
        f"clamp tokens 3 / 4095. {len(m)} (dataset, severity) cells over ",
        f"{m.dataset.nunique()} datasets. Degradation is the sweep's own ",
        f"`{metric}_degr` (per-dataset ratio to that model's clean score).", "",
        "## Q1 - aggregate: does clamping switch on where the curve recovers?", "",
        agg.round(4).to_markdown(index=False), "",
        "## Q2 - per-dataset: do the datasets that clamp recover?", "",
        f"Spearman(max clamped fraction, recovery) **rho = {rho2.statistic:.3f}**, ",
        f"p = {rho2.pvalue:.4f}, n = {len(per_ds)} datasets. ",
        "`recovery` = peak degradation / degradation at the largest severity; ",
        "a value above 1 means the curve came back down after peaking.", "",
        per_ds.round(4).to_markdown(index=False), "",
        "## Q3 - same cells, two x-axes", "",
        f"- vs **nominal** severity: Spearman rho = {rn.statistic:.3f} (p = {rn.pvalue:.3g})",
        f"- vs **realised** excursion: Spearman rho = {rr.statistic:.3f} (p = {rr.pvalue:.3g})", "",
        f"![mechanism]({fig_path.name})", "",
    ]
    md = OUT / f"CLAMPING_ANALYSIS_{family}_{metric}.md"
    md.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"\n-> {fig_path}\n-> {md}")


if __name__ == "__main__":
    main()
