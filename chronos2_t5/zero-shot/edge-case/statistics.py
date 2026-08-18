"""E6 -- one source of truth for every statistical claim in the paper.

Three things this fixes, in increasing order of how badly they would have bitten:

1. ANALYSIS UNIT. The multi-seed run gives 4 draws per dataset, i.e. 100 curves per
   model -- but those are 25 datasets x 4 seeds, and the seeds of one dataset are
   correlated. Treating them as 100 independent observations inflates significance.
   Every test below therefore collapses seeds within a dataset FIRST and uses the
   dataset as the unit (n = 25), which is also the unit the degradation ratio is
   defined on.

2. PAIRING. Both models see byte-identical corrupted contexts on the same datasets, so
   model comparisons are paired. A paired Wilcoxon signed-rank test over datasets is
   both more appropriate and more powerful than the unpaired Mann-Whitney used in the
   first pass.

3. MULTIPLICITY. The refutation in E1 enumerates six correlations (three candidate
   predictors x two metrics). Reporting one of them as "significant" without correction
   is exactly the practice this paper criticises elsewhere. Holm and Benjamini-Hochberg
   are applied within each declared family.

Bootstrap CIs resample the 25 datasets with replacement (2000 draws), matching the
convention already used for win rates in the technical report.

Output (results/STATISTICS.md): every test, with raw p, Holm p, BH p, and the effect
size, ready to be quoted directly in the paper.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gmean, spearmanr, wilcoxon

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
RNG = np.random.default_rng(0)
NBOOT = 2000
FAMS = ["spikes_intensity", "spikes_density"]
MODELS = ["chronos-t5", "chronos-2"]


# ---------------------------------------------------------------- corrections
def holm(ps: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values (monotone, capped at 1)."""
    m = len(ps)
    order = np.argsort(ps)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * ps[i])
        adj[i] = min(running, 1.0)
    return adj


def bh(ps: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (step-up, monotone, capped at 1)."""
    m = len(ps)
    order = np.argsort(ps)
    adj = np.empty(m)
    running = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        running = min(running, m / (rank + 1) * ps[i])
        adj[i] = min(running, 1.0)
    return adj


def boot_ci(x, stat=np.mean, n=NBOOT, alpha=0.05):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return (np.nan, np.nan)
    draws = [stat(x[RNG.integers(0, x.size, x.size)]) for _ in range(n)]
    return tuple(np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


# ---------------------------------------------------------------- data
def load_seeds() -> pd.DataFrame:
    s0 = pd.read_csv(OUT / "edge_case_results.csv")
    s0 = s0[s0.family.isin(FAMS + ["clean"])].copy()
    s0["seed"] = 0
    cols = ["seed", "dataset", "model", "family", "severity", "MASE", "WQL"]
    df = pd.concat([s0[cols], pd.read_csv(OUT / "edge_case_seeds.csv")[cols]],
                   ignore_index=True)
    clean = (df[df.family == "clean"]
             .set_index(["seed", "model", "dataset"])[["MASE", "WQL"]]
             .rename(columns={"MASE": "MASE_clean", "WQL": "WQL_clean"}))
    df = df.join(clean, on=["seed", "model", "dataset"])
    df["MASE_degr"] = df.MASE / df.MASE_clean
    df["WQL_degr"] = df.WQL / df.WQL_clean
    return df[df.family != "clean"]


def per_dataset_rho(df, fam, metric) -> pd.DataFrame:
    """Spearman(severity, degradation) per curve, then averaged over seeds within a
    dataset so the analysis unit is the dataset."""
    d = df[df.family == fam]
    col = f"{metric}_degr"
    rows = []
    for (seed, model, ds), g in d.groupby(["seed", "model", "dataset"]):
        g = g.sort_values("severity")
        y = g[col].to_numpy()
        if len(g) < 4 or not np.all(np.isfinite(y)):
            continue
        rows.append({"seed": seed, "model": model, "dataset": ds,
                     "rho": spearmanr(g.severity, y).statistic})
    per_seed = pd.DataFrame(rows)
    return per_seed.groupby(["model", "dataset"], as_index=False).rho.mean()


# ---------------------------------------------------------------- families
def family_c1(df, metric):
    """C1: does each model's error track severity? Paired over datasets, n = 25."""
    tests = []
    for fam in FAMS:
        agg = per_dataset_rho(df, fam, metric)
        piv = agg.pivot(index="dataset", columns="model", values="rho").dropna()
        t5, c2 = piv["chronos-t5"].to_numpy(), piv["chronos-2"].to_numpy()

        w = wilcoxon(c2, t5, alternative="greater")
        tests.append({
            "family": f"C1/{fam} [{metric}]",
            "test": "Chronos-2 rho > Chronos-T5 rho (paired)",
            "n": len(piv), "effect": f"mean rho {c2.mean():+.3f} vs {t5.mean():+.3f}",
            "stat": float(w.statistic), "p": float(w.pvalue)})

        for name, v in (("chronos-2", c2), ("chronos-t5", t5)):
            w1 = wilcoxon(v, alternative="greater")
            lo, hi = boot_ci(v)
            tests.append({
                "family": f"C1/{fam} [{metric}]", "test": f"{name} rho > 0",
                "n": len(v), "effect": f"mean rho {v.mean():+.3f} [95% CI {lo:+.3f}, {hi:+.3f}]",
                "stat": float(w1.statistic), "p": float(w1.pvalue)})
    return tests


def family_c1_heldout(df, metric, peak=12.0):
    """C1 held-out: seeds 1-3 only, contrast located on seed 0. Seeds averaged first."""
    d = df[(df.family == "spikes_intensity") & (df.seed > 0)]
    col = f"{metric}_degr"
    smax = d.severity.max()
    tests = []
    for model in MODELS:
        m = d[d.model == model]
        a = m[m.severity == peak].groupby("dataset")[col].mean()
        b = m[m.severity == smax].groupby("dataset")[col].mean()
        j = pd.concat([a.rename("peak"), b.rename("end")], axis=1).dropna()
        j = j[(j.peak > 0) & (j.end > 0)]
        w = wilcoxon(j.peak, j.end, alternative="greater")
        tests.append({
            "family": f"C1/held-out [{metric}]", "n": len(j),
            "test": f"{model}: degradation at sev {peak:g} > at sev {smax:g} (seeds 1-3)",
            "effect": f"median {j.peak.median():.3f} -> {j.end.median():.3f}",
            "stat": float(w.statistic), "p": float(w.pvalue)})
    return tests


def family_c2():
    """C2 refutation: three candidate predictors of recovery x two metrics."""
    clamp = pd.read_csv(OUT / "clamping_measurements.csv")
    degr = pd.read_csv(OUT / "edge_case_results.csv")
    clean_scale = clamp[clamp.family == "clean"].set_index("dataset").scale_mean
    si = clamp[clamp.family == "spikes_intensity"]

    tests = []
    for metric in ("MASE", "WQL"):
        d = degr[(degr.model == "chronos-t5") & (degr.family == "spikes_intensity")]
        rows = []
        for ds, g in d.groupby("dataset"):
            g = g.sort_values("severity")
            y = g[f"{metric}_degr"].to_numpy()
            if not np.all(np.isfinite(y)) or y[-1] <= 0:
                continue
            c = si[si.dataset == ds].sort_values("severity")
            rows.append({"recovery": y.max() / y[-1],
                         "clamped": c.clamped_frac.max(),
                         "scale_infl": c.scale_mean.iloc[-1] / clean_scale[ds],
                         "excursion": c.nominal_p99.iloc[-1] / c.nominal_p99.iloc[0]})
        t = pd.DataFrame(rows)
        for col, label in (("clamped", "max clamped fraction (clamping hypothesis)"),
                           ("scale_infl", "mean-scale inflation (scaling hypothesis)"),
                           ("excursion", "growth of excursion reaching the model")):
            r = spearmanr(t[col], t.recovery)
            tests.append({"family": "C2/refutation", "n": len(t),
                          "test": f"{metric}: Spearman(recovery, {label})",
                          "effect": f"rho = {r.statistic:+.3f}",
                          "stat": float(r.statistic), "p": float(r.pvalue)})
    return tests


# ---------------------------------------------------------------- report
def main():
    df = load_seeds()
    # Both metrics, WQL first because it is the study's PRIMARY metric. An earlier pass
    # computed C1 on MASE only, which quietly made every headline number the secondary
    # metric's. The two are correlated measurements of one hypothesis, not two
    # hypotheses, so each gets its own correction family rather than being pooled:
    # WQL is the claim, MASE is the robustness check.
    tests = []
    for metric in ("WQL", "MASE"):
        tests += family_c1(df, metric) + family_c1_heldout(df, metric)
    tests += family_c2()
    T = pd.DataFrame(tests)

    # Corrections applied WITHIN each declared family, not across the whole table:
    # the families test logically separate questions.
    T["p_holm"] = np.nan
    T["p_bh"] = np.nan
    for fam, g in T.groupby("family"):
        T.loc[g.index, "p_holm"] = holm(g.p.to_numpy())
        T.loc[g.index, "p_bh"] = bh(g.p.to_numpy())

    show = T[["family", "test", "n", "effect", "p", "p_holm", "p_bh"]].copy()
    for c in ("p", "p_holm", "p_bh"):
        show[c] = show[c].map(lambda v: f"{v:.3g}")
    print(show.to_string(index=False))

    lines = [
        "# Statistical appendix", "",
        "**Analysis unit.** Every test below uses the *dataset* as the unit of analysis "
        f"(n = 25). The multi-seed run provides four independent corruption draws per "
        "dataset; those are averaged within a dataset before testing, because seeds of "
        "the same dataset are not independent observations. An earlier pass treated the "
        "100 (seed, dataset) curves as independent, which inflates significance.", "",
        "**Pairing.** Both models receive byte-identical corrupted contexts on the same "
        "datasets, so model comparisons use a paired Wilcoxon signed-rank test rather "
        "than an unpaired test.", "",
        "**Multiplicity.** Holm-Bonferroni and Benjamini-Hochberg adjusted p-values are "
        "reported alongside raw p, corrected within each family of tests.", "",
        "**Metrics.** WQL (9-quantile grid) is the primary metric and carries the claim; "
        "MASE is reported as a robustness check. They are two correlated measurements of "
        "one hypothesis rather than two hypotheses, so each is corrected within its own "
        "family and no correction is applied across them. Both are shown for every C1 "
        "test so the reader can see that the conclusion does not depend on the choice.", "",
        "**Intervals.** Bootstrap 95% CIs resample the 25 datasets with replacement "
        f"({NBOOT} draws).", "",
        show.to_markdown(index=False), "",
    ]
    md = OUT / "STATISTICS.md"
    md.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"\n-> {md}")


if __name__ == "__main__":
    main()
