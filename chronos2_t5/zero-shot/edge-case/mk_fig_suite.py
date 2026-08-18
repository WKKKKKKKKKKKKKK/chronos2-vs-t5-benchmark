"""Figure for the Setup section: what the seven corruption families do to one series.

The paper needs one image that shows the whole suite at a glance. The existing
`results/examples/<dataset>/<family>.png` figures are four panels each of ONE family, which
is right for an appendix and useless for a reader meeting the suite for the first time.

Design decisions, since they affect what the figure argues:

  * One dataset, one series, seven panels plus the clean reference. Holding the series fixed
    is the point -- the reader should see the corruption, not a change of subject.
  * Severities are chosen for legibility, not from the sweep grid, and the caption must say
    so. A figure is an illustration; the numbers come from the tables.
  * The forecast origin is drawn on every panel. Placement relative to it is one of the
    paper's findings, and a reader who cannot see where the context ends cannot see why
    `gap` and `gap_boundary` are different families.
  * `drift` and `regime_trend` are drawn adjacent with the same severity, because they are
    matched by construction at the origin and the figure should make that visible.

No GPU: this draws corrupted inputs, not forecasts.

Usage:  python mk_fig_suite.py [--dataset nn5] [--series 0]
Output: results/fig_corruption_suite.png (+ .pdf for LaTeX inclusion)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import run_edge_cases as RE      # noqa: E402
import perturbations as P        # noqa: E402

OUT = HERE / "results"

# (family, severity, panel title). Severities are illustrative -- mid-grid where the effect
# is visible without dominating the y-axis.
PANELS = [
    ("spikes_intensity", 12.0, "spike magnitude  (12x scale, 5% of points)"),
    ("spikes_density",   0.20, "spike density  (20% of points, 20x scale)"),
    ("drift",             8.0, "global drift  (ramp over the whole context)"),
    ("regime_trend",      8.0, "persistent trend change  (last 25%, same end-offset)"),
    ("drift_step",        8.0, "transient level shift  (random 30% block)"),
    ("gap",               0.20, "dropout, random position  (20% blanked)"),
    ("gap_boundary",      0.20, "dropout at the origin  (20% blanked)"),
]


def _arg(flag: str, default: str) -> str:
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def main():
    ds = _arg("--dataset", "nn5")
    idx = int(_arg("--series", "0"))
    horizon = dict(RE.EDGE_DATASETS)[ds]

    _, contexts, _ = RE.build_dataset(ds, horizon)
    ctx = np.asarray(contexts[idx], dtype=np.float32)
    n = ctx.size
    scale = P.robust_scale(ctx)
    print(f"{ds}[{idx}]: {n} context steps, horizon {horizon}, MAD scale {scale:.4g}")

    fig, axes = plt.subplots(4, 2, figsize=(11.0, 9.6), sharex=True)
    axes = axes.ravel()

    # panel 0: the clean reference every other panel is a perturbation of
    axes[0].plot(ctx, lw=0.9, color="0.25")
    axes[0].set_title("clean context", fontsize=10, fontweight="bold", loc="left")

    for i, (ax, (fam, sev, title)) in enumerate(zip(axes[1:], PANELS)):
        # Seed by panel index, not hash(): hash() is per-process randomised, so a
        # hash-seeded figure is not reproducible run to run.
        rng = np.random.default_rng(np.random.SeedSequence([0, 900 + i]))
        pc = P.apply(fam, ctx, rng, sev)
        ax.plot(ctx, lw=0.7, color="0.72", zorder=1)          # clean, for reference
        ax.plot(pc, lw=0.9, color="C3", zorder=2)             # corrupted

        # Shade only the points this corruption BLANKED. The clean series may already
        # contain NaNs, so a plain isfinite() test on the corrupted array shades
        # pre-existing gaps too -- which is what an earlier version of this figure did,
        # covering every panel edge to edge.
        blanked = ~np.isfinite(pc) & np.isfinite(ctx)
        if blanked.any():
            idx_b = np.flatnonzero(blanked)
            ax.axvspan(idx_b[0], idx_b[-1], color="C3", alpha=0.15, zorder=0)
            ax.text(0.5 * (idx_b[0] + idx_b[-1]), ax.get_ylim()[1], "blanked",
                    ha="center", va="top", fontsize=7.5, color="C3")
        ax.set_title(title, fontsize=9.5, loc="left")

    for k, ax in enumerate(axes):
        ax.axvline(n - 1, color="C0", ls="--", lw=1.1, zorder=3)
        ax.tick_params(labelsize=8)
        ax.margins(x=0.01)
        if k == 0:
            ax.annotate("forecast origin", xy=(n - 1, ax.get_ylim()[1]),
                        xytext=(-6, -12), textcoords="offset points",
                        ha="right", fontsize=8, color="C0")
    for ax in axes[-2:]:
        ax.set_xlabel("context step", fontsize=9)

    fig.suptitle(f"The seven corruption families, one series of {ds}. "
                 "Grey = clean, red = corrupted, dashed = forecast origin.",
                 fontsize=10.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    for ext in ("png", "pdf"):
        p = OUT / f"fig_corruption_suite.{ext}"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"-> {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
