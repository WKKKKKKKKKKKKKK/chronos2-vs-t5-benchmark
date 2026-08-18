"""E3 analysis -- does the non-monotonic spike-intensity curve survive independent seeds?

C1 (Chronos-T5's degradation peaks mid-range and recovers, while Chronos-2's rises
monotonically) currently rests on a single corruption draw per cell, because `severity`
enters the seed in `run_edge_cases._rng`. This script pools seed 0 (the original sweep)
with seeds 1-3 from `run_seeds.py` and asks three questions:

  A  Shape per seed. Does each seed independently show peak-then-recover for
     Chronos-T5 and monotone growth for Chronos-2?

  B  Monotonicity, unbiased. Spearman(severity, degradation) computed per
     (seed, dataset) curve. Monotone growth gives rho near +1. Comparing the
     distribution of rho between models avoids the selection bias of "find the peak,
     then measure how far it fell" -- picking a maximum guarantees a fall.

  C  Out-of-sample confirmation. Seed 0 located the aggregate peak at severity 12.
     Seeds 1-3 never informed that choice, so testing degr(sev=12) > degr(sev=40) on
     seeds 1-3 alone is a clean held-out test of the shape, not a re-description of it.

Outputs (results/):
  fig_seed_curves.png        per-model curves, mean +/- s.d. over seeds
  SEED_ANALYSIS.md           the numbers
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gmean, spearmanr, wilcoxon

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
FAMS = ["spikes_intensity", "spikes_density"]
MODELS = ["chronos-t5", "chronos-2"]
PEAK_SEV = 12.0      # located on seed 0; used as the held-out contrast for seeds 1-3


def load() -> pd.DataFrame:
    """Seed 0 (original sweep) + seeds 1-3, degradation recomputed uniformly per seed."""
    s0 = pd.read_csv(OUT / "edge_case_results.csv")
    s0 = s0[s0.family.isin(FAMS + ["clean"])].copy()
    s0["seed"] = 0
    cols = ["seed", "dataset", "model", "family", "severity", "MASE", "WQL"]
    s0 = s0[cols]

    sN = pd.read_csv(OUT / "edge_case_seeds.csv")[cols]
    df = pd.concat([s0, sN], ignore_index=True)

    # degradation = metric / that (seed, model, dataset)'s own clean score
    clean = (df[df.family == "clean"]
             .set_index(["seed", "model", "dataset"])[["MASE", "WQL"]]
             .rename(columns={"MASE": "MASE_clean", "WQL": "WQL_clean"}))
    df = df.join(clean, on=["seed", "model", "dataset"])
    df["MASE_degr"] = df.MASE / df.MASE_clean
    df["WQL_degr"] = df.WQL / df.WQL_clean
    return df[df.family != "clean"]


def q_a_shape(df, fam, metric):
    """Aggregate curve per seed: gmean over datasets."""
    d = df[df.family == fam]
    col = f"{metric}_degr"
    rows = {}
    for model in MODELS:
        m = d[d.model == model]
        piv = (m.groupby(["seed", "severity"])[col]
               .apply(lambda v: gmean(v[np.isfinite(v) & (v > 0)]))
               .unstack("seed"))
        rows[model] = piv
    return rows


def q_b_monotonicity(df, fam, metric):
    """Spearman(severity, degradation) for every (seed, dataset) curve."""
    d = df[df.family == fam]
    col = f"{metric}_degr"
    out = []
    for (seed, model, ds), g in d.groupby(["seed", "model", "dataset"]):
        g = g.sort_values("severity")
        y = g[col].to_numpy()
        if len(g) < 4 or not np.all(np.isfinite(y)):
            continue
        out.append({"seed": seed, "model": model, "dataset": ds,
                    "rho": spearmanr(g.severity, y).statistic})
    return pd.DataFrame(out)


def q_c_heldout(df, fam, metric, peak=PEAK_SEV):
    """Held-out test on seeds 1-3: is degradation at the seed-0 peak above that at max severity?"""
    d = df[(df.family == fam) & (df.seed > 0)]
    col = f"{metric}_degr"
    smax = d.severity.max()
    res = {}
    for model in MODELS:
        m = d[d.model == model]
        a = m[m.severity == peak].set_index(["seed", "dataset"])[col]
        b = m[m.severity == smax].set_index(["seed", "dataset"])[col]
        j = pd.concat([a.rename("peak"), b.rename("end")], axis=1).dropna()
        j = j[(j.peak > 0) & (j.end > 0)]
        if len(j) < 6:
            res[model] = None
            continue
        w = wilcoxon(j.peak, j.end, alternative="greater")
        res[model] = {"n": len(j), "median_peak": float(j.peak.median()),
                      "median_end": float(j.end.median()),
                      "stat": float(w.statistic), "p": float(w.pvalue)}
    return res, smax


def figure(curves, df, metric):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    colors = {"chronos-t5": "C0", "chronos-2": "C3"}
    for ax, fam in zip(axes, FAMS):
        for model in MODELS:
            piv = curves[fam][model]
            sev = piv.index.to_numpy()
            mu, sd = piv.mean(axis=1).to_numpy(), piv.std(axis=1, ddof=1).to_numpy()
            ax.plot(sev, mu, "o-", color=colors[model], lw=2.3, ms=7,
                    label=f"{model} (mean of {piv.shape[1]} seeds)")
            ax.fill_between(sev, mu - sd, mu + sd, color=colors[model], alpha=0.18)
        ax.axhline(1.0, color="grey", ls=":", lw=1.2)
        ax.set_xlabel("severity", fontsize=13)
        ax.set_ylabel(f"{metric} (x clean)", fontsize=13)
        ax.set_title(fam.replace("_", " "), fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    p = OUT / f"fig_seed_curves_{metric}.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return p


def main():
    metric = sys.argv[sys.argv.index("--metric") + 1] if "--metric" in sys.argv else "MASE"
    df = load()
    seeds = sorted(df.seed.unique())
    print(f"seeds: {seeds}   datasets: {df.dataset.nunique()}   rows: {len(df)}")

    curves = {f: q_a_shape(df, f, metric) for f in FAMS}
    lines = [f"# E3 -- multi-seed replication ({metric})", "",
             f"Seeds {seeds}; {df.dataset.nunique()} datasets. Degradation is each "
             f"(seed, model, dataset)'s metric divided by its own clean-context score, "
             f"aggregated across datasets by geometric mean.", ""]

    print(f"\n=== A. spikes_intensity aggregate curve, per seed ({metric}) ===")
    lines += ["## A. Aggregate curve per seed (spikes_intensity)", ""]
    for model in MODELS:
        piv = curves["spikes_intensity"][model].round(3)
        print(f"\n{model}")
        print(piv.to_string())
        lines += [f"### {model}", "", piv.to_markdown(), ""]

    print(f"\n=== B. monotonicity: Spearman(severity, degradation) per curve ===")
    lines += ["## B. Monotonicity per (seed, dataset) curve", "",
              "Spearman rho of degradation against severity; +1 = perfectly monotone growth.", ""]
    rho = q_b_monotonicity(df, "spikes_intensity", metric)
    tab = (rho.groupby("model").rho
           .agg(mean="mean", median="median", sd="std", n="size",
                frac_pos=lambda v: float((v > 0).mean()))
           .round(3))
    print(tab.to_string())
    lines += [tab.to_markdown(), ""]
    a = rho[rho.model == "chronos-2"].rho.to_numpy()
    b = rho[rho.model == "chronos-t5"].rho.to_numpy()
    from scipy.stats import mannwhitneyu
    mw = mannwhitneyu(a, b, alternative="greater")
    print(f"  C2 rho > T5 rho : Mann-Whitney U={mw.statistic:.0f}, p={mw.pvalue:.3g}")
    lines += [f"Chronos-2 curves are more monotone than Chronos-T5 curves: "
              f"Mann-Whitney U = {mw.statistic:.0f}, p = {mw.pvalue:.3g}.", ""]

    print(f"\n=== C. held-out test on seeds 1-3 (peak sev={PEAK_SEV} located on seed 0) ===")
    res, smax = q_c_heldout(df, "spikes_intensity", metric)
    lines += [f"## C. Held-out recovery test (seeds 1-3 only)", "",
              f"One-sided Wilcoxon: degradation at severity {PEAK_SEV:g} greater than at "
              f"{smax:g}. Severity {PEAK_SEV:g} was located on seed 0, which is excluded here.", ""]
    for model, r in res.items():
        if r is None:
            print(f"  {model}: insufficient pairs")
            continue
        print(f"  {model:<11} n={r['n']:3d}  median {r['median_peak']:.3f} -> "
              f"{r['median_end']:.3f}   W={r['stat']:.0f}  p={r['p']:.4g}")
        lines += [f"- **{model}**: n = {r['n']}, median degradation "
                  f"{r['median_peak']:.3f} at sev {PEAK_SEV:g} vs {r['median_end']:.3f} at "
                  f"sev {smax:g}; W = {r['stat']:.0f}, p = {r['p']:.4g}"]

    p = figure(curves, df, metric)
    lines += ["", f"![seed curves]({p.name})", ""]
    md = OUT / f"SEED_ANALYSIS_{metric}.md"
    md.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"\n-> {p}\n-> {md}")


if __name__ == "__main__":
    main()
