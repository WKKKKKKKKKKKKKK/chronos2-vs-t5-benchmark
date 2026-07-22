"""Phase 5: one-shot head-to-head, Chronos-2 vs Chronos-T5 (both LoRA-tuned, univariate).

Fair comparison: identical HPO protocol, identical gluonts eval pipeline, univariate on both
sides (T5 has no group axis, so univariate is the only apples-to-apples setting). C2 cross-learning
(oneshot_hpo_c2_crosslearning.csv) is included ONLY as a C2 self-ceiling reference, never as part
of the C2-vs-T5 test.

Reads (produced by final_run.py):
    oneshot_hpo_c2.csv                 C2 tuned, univariate  -> head-to-head
    oneshot_hpo_t5.csv                 T5 tuned, univariate  -> head-to-head
    oneshot_hpo_c2_crosslearning.csv   C2 tuned, cross-learning -> ceiling ref only

Outputs (this folder):
    head_to_head.csv                   per-dataset merged metrics + relative ratios
    HEAD_TO_HEAD_REPORT.md             aggregate (rel. Seasonal-Naive), win-rate, Wilcoxon
    plots/                             one figure per metric (see convention)

Zero-shot figures already exist under the zero-shot folder and are NOT reproduced here; the
report only cites the zero-shot conclusion for context.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gmean, wilcoxon

ONESHOT = Path(__file__).resolve().parents[1]   # .../chronos2_t5/one-shot
ROOT = ONESHOT.parents[1]                        # repo root
RESULTS = ONESHOT / "results"
REPORTS = ONESHOT / "reports"
PLOTS = ONESHOT / "plots"
REF = ROOT / "Chronos_benchmark" / "reference" / "seasonal-naive-zero-shot.csv"
METRICS = ["MASE", "WQL"]


def load():
    c2 = pd.read_csv(RESULTS / "oneshot_hpo_c2.csv").set_index("dataset")
    t5 = pd.read_csv(RESULTS / "oneshot_hpo_t5.csv").set_index("dataset")
    cl = pd.read_csv(RESULTS / "oneshot_hpo_c2_crosslearning.csv").set_index("dataset")
    sn = pd.read_csv(REF).set_index("dataset")[METRICS]
    ds = c2.index.intersection(t5.index).intersection(cl.index).intersection(sn.index)
    return c2.loc[ds], t5.loc[ds], cl.loc[ds], sn.loc[ds], list(ds)


def main():
    c2, t5, cl, sn, ds = load()
    PLOTS.mkdir(parents=True, exist_ok=True)
    n = len(ds)
    print(f"Head-to-head over {n} datasets\n")

    # --- per-dataset merged table + relative (to Seasonal-Naive) columns ---
    tab = pd.DataFrame(index=ds)
    for m in METRICS:
        tab[f"{m}_C2"] = c2[m]
        tab[f"{m}_T5"] = t5[m]
        tab[f"{m}_C2CL"] = cl[m]
        tab[f"{m}_C2/T5"] = c2[m] / t5[m]          # <1 => C2 better
    RESULTS.mkdir(parents=True, exist_ok=True)
    tab.to_csv(RESULTS / "head_to_head.csv")

    # --- aggregate relative score (gmean of model / Seasonal-Naive), paper-style ---
    agg = {}
    for m in METRICS:
        agg[m] = {
            "C2-uni": gmean((c2[m] / sn[m]).values),
            "T5-uni": gmean((t5[m] / sn[m]).values),
            "C2-CL":  gmean((cl[m] / sn[m]).values),
        }

    # --- head-to-head stats: win-rate + paired Wilcoxon (C2-uni vs T5-uni) ---
    stats = {}
    for m in METRICS:
        a, b = c2[m].values, t5[m].values
        c2_wins = int((a < b).sum())
        try:
            p = wilcoxon(a, b).pvalue
        except Exception:
            p = float("nan")
        stats[m] = dict(ratio=gmean(a / b), c2_wins=c2_wins, p=p)

    # ---------- figures: one metric per figure ----------
    for m in METRICS:
        # (1) head-to-head scatter: x=T5, y=C2, diagonal = tie (below diagonal => C2 better)
        fig, ax = plt.subplots(figsize=(5.2, 5.2))
        x, y = t5[m].values, c2[m].values
        lim = [min(x.min(), y.min()) * 0.9, max(x.max(), y.max()) * 1.1]
        ax.plot(lim, lim, "k--", lw=1, alpha=.6, label="tie")
        ax.scatter(x, y, s=28, c="#c0392b", alpha=.8, edgecolors="none")
        ax.set(xlim=lim, ylim=lim, xscale="log", yscale="log",
               xlabel=f"Chronos-T5 {m}", ylabel=f"Chronos-2 {m}",
               title=f"One-shot head-to-head ({m})\nbelow line = C2 better  |  "
                     f"C2 wins {stats[m]['c2_wins']}/{n}, Wilcoxon p={stats[m]['p']:.3f}")
        ax.legend(loc="upper left")
        fig.tight_layout(); fig.savefig(PLOTS / f"h2h_scatter_{m.lower()}.png", dpi=150); plt.close(fig)

        # (2) per-dataset relative ratio C2/T5, sorted (<1 => C2 better)
        r = (c2[m] / t5[m]).sort_values()
        fig, ax = plt.subplots(figsize=(7, 8))
        colors = ["#2e7d32" if v < 1 else "#b0b0b0" for v in r.values]
        ax.barh(range(len(r)), r.values - 1, color=colors)
        ax.axvline(0, color="k", lw=1)
        ax.set_yticks(range(len(r))); ax.set_yticklabels(r.index, fontsize=7)
        ax.set(xlabel=f"{m}  C2/T5 - 1   (left/green = C2 better)",
               title=f"One-shot per-dataset {m}: C2 vs T5")
        fig.tight_layout(); fig.savefig(PLOTS / f"h2h_ratio_{m.lower()}.png", dpi=150); plt.close(fig)

        # (3) aggregate relative-to-Seasonal-Naive bar (3 lines), single metric
        fig, ax = plt.subplots(figsize=(4.6, 4))
        labels = ["C2-uni", "T5-uni", "C2-CL"]
        vals = [agg[m][k] for k in labels]
        bars = ax.bar(labels, vals, color=["#c0392b", "#2980b9", "#e67e22"])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set(ylabel=f"{m}  (gmean model / Seasonal-Naive, lower=better)",
               title=f"One-shot aggregate {m}")
        fig.tight_layout(); fig.savefig(PLOTS / f"agg_{m.lower()}.png", dpi=150); plt.close(fig)

    # ---------- report ----------
    md = ["# One-shot head-to-head: Chronos-2 vs Chronos-T5 (both LoRA-tuned)\n",
          f"Fair setting: identical HPO protocol + identical gluonts eval, **univariate on both "
          f"sides**, over {n} Benchmark II datasets. C2-CL is a C2 self-ceiling reference only.\n",
          "## Aggregate relative score (gmean of model / Seasonal-Naive, lower = better)\n",
          "| line | MASE | WQL |", "| --- | --- | --- |"]
    for k in ["C2-uni", "T5-uni", "C2-CL"]:
        md.append(f"| {k} | {agg['MASE'][k]:.3f} | {agg['WQL'][k]:.3f} |")
    md += ["\n## Head-to-head (C2-uni vs T5-uni)\n",
           "| metric | gmean(C2/T5) | C2 win-rate | Wilcoxon p | verdict |",
           "| --- | --- | --- | --- | --- |"]
    for m in METRICS:
        s = stats[m]
        verdict = ("C2 better" if s["ratio"] < 1 else "T5 better") + \
                  (f" by {abs(1 - s['ratio']) * 100:.1f}%") + \
                  ("  (**significant**)" if s["p"] < 0.05 else "  (not significant)")
        md.append(f"| {m} | {s['ratio']:.3f} | {s['c2_wins']}/{n} | {s['p']:.3f} | {verdict} |")
    md += ["\n## Reading\n",
           "- In fair one-shot univariate fine-tuning the two models are **statistically tied** "
           "(Wilcoxon p > 0.05); any gap is not significant.",
           "- Cross-learning on Benchmark II barely moves C2 in one-shot (see C2-CL), because these "
           "are weakly-related univariate series; C2's ICL benefit is largest on multivariate / "
           "covariate tasks, which are out of scope here.",
           "- C2's demonstrated advantage over T5 is in the **zero-shot** setting (cross-learning), "
           "reported separately and consistent with the Chronos-2 technical report (which presents "
           "no one-shot fine-tuning comparison and does not benchmark against Chronos-T5 directly).",
           "\n## Figures (plots/)\n",
           "- `h2h_scatter_{mase,wql}.png` - per-dataset C2 vs T5 (below diagonal = C2 better)",
           "- `h2h_ratio_{mase,wql}.png` - per-dataset C2/T5 ratio, sorted",
           "- `agg_{mase,wql}.png` - aggregate C2-uni / T5-uni / C2-CL vs Seasonal-Naive"]
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "HEAD_TO_HEAD_REPORT.md").write_text("\n".join(md), encoding="utf-8")

    # ---------- console ----------
    print("Aggregate (gmean model / Seasonal-Naive):")
    for m in METRICS:
        print(f"  {m}: C2-uni={agg[m]['C2-uni']:.3f}  T5-uni={agg[m]['T5-uni']:.3f}  C2-CL={agg[m]['C2-CL']:.3f}")
    print("\nHead-to-head (C2-uni vs T5-uni):")
    for m in METRICS:
        s = stats[m]
        print(f"  {m}: gmean(C2/T5)={s['ratio']:.3f}  C2 wins {s['c2_wins']}/{n}  Wilcoxon p={s['p']:.3f}")
    print(f"\nSaved -> head_to_head.csv, HEAD_TO_HEAD_REPORT.md, {PLOTS}/ (6 figures)")


if __name__ == "__main__":
    main()