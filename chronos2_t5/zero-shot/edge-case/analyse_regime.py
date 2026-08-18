"""E10 -- does the undersensitivity finding survive outside the spike families?

C1 says Chronos-T5's error tracks spike *density* but not spike *magnitude*, while
Chronos-2 tracks both. That rests entirely on one corruption semantics: "some past
observations are wrong". `regime_trend` changes the semantics -- the generating process
changes partway through the context and is still in the new mode at the forecast origin,
so the model must judge whether to extrapolate it rather than filter it out.

Three questions, in the order that decides what gets written:

  Q1  Same test as C1: Spearman(severity, degradation) per (seed, dataset) curve,
      averaged within dataset, paired across models over the 25 datasets. Does the
      Chronos-2 / Chronos-T5 split reappear?
  Q2  The matched pair. `drift` and `regime_trend` displace the final context point by
      exactly the same amount at equal severity; they differ only in whether the
      displacement accumulated over the whole history (no breakpoint) or over the last
      25% (breakpoint, 4x steeper). Comparing degradation between them isolates the
      breakpoint at fixed origin displacement.
  Q3  Absolute cost, so a flat slope is not confused with a harmless corruption.

`drift` comes from the seed-0 sweep (edge_case_results.csv, no `seed` column); the paired
comparison in Q2 is therefore restricted to seed 0 for both families and is reported as
such. Q1 uses all four seeds of the regime run.

Analysis unit is the dataset (n = 25) throughout; seeds are averaged within a dataset
before any test, CIs bootstrap the datasets, and the paired contrast is Wilcoxon
signed-rank -- the same conventions as `statistics.py`, which remains the single source
of truth for anything that ends up in the paper.

Outputs (results/): fig_regime_<metric>.png, REGIME_ANALYSIS_<metric>.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
REGIME_CSV = OUT / "edge_case_regime.csv"      # overridable with --csv
SWEEP_CSV = OUT / "edge_case_results.csv"
SPIKE_CSV = OUT / "edge_case_seeds.csv"        # E3 multi-seed spikes, seeds 1-3
MODELS = ["chronos-2", "chronos-t5"]
NBOOT = 2000
RNG = np.random.default_rng(0)


def boot_ci(x, stat=np.mean, n=NBOOT):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return (np.nan, np.nan)
    d = [stat(x[RNG.integers(0, x.size, x.size)]) for _ in range(n)]
    return tuple(np.percentile(d, [2.5, 97.5]))


def load_regime(metric: str) -> pd.DataFrame:
    """Degradation = metric / that (seed, model, dataset)'s own clean score."""
    df = pd.read_csv(REGIME_CSV)
    clean = (df[df.family == "clean"]
             .set_index(["seed", "model", "dataset"])[metric].rename("clean"))
    d = df[df.family == "regime_trend"].join(clean, on=["seed", "model", "dataset"])
    d["degr"] = d[metric] / d["clean"]
    return d


def q1_slope(d: pd.DataFrame) -> pd.DataFrame:
    """Spearman(severity, degradation) per curve, then averaged within dataset."""
    rows = []
    for (seed, model, ds), g in d.groupby(["seed", "model", "dataset"]):
        g = g.dropna(subset=["degr"])
        if g.severity.nunique() < 3:
            continue
        rows.append({"seed": seed, "model": model, "dataset": ds,
                     "rho": spearmanr(g.severity, g.degr).statistic})
    per_seed = pd.DataFrame(rows)
    return per_seed.groupby(["model", "dataset"], as_index=False).rho.mean()


def q2_matched_pair(metric: str) -> pd.DataFrame | None:
    """drift vs regime_trend at equal severity, seed 0 only (drift has no other seeds).

    At equal severity the two displace the final context point identically; any
    difference in degradation is attributable to the breakpoint and the 4x local slope.
    """
    if not SWEEP_CSV.exists():
        return None
    sw = pd.read_csv(SWEEP_CSV)
    dr = sw[sw.family == "drift"][["dataset", "model", "severity", f"{metric}_degr"]]
    rg = load_regime(metric)
    rg = rg[rg.seed == 0][["dataset", "model", "severity", "degr"]]
    shared = sorted(set(dr.severity) & set(rg.severity))
    if not shared:
        return None
    j = (dr[dr.severity.isin(shared)]
         .merge(rg[rg.severity.isin(shared)], on=["dataset", "model", "severity"]))
    j = j.rename(columns={f"{metric}_degr": "drift", "degr": "regime"})
    # one number per (model, dataset): mean over the shared severities
    agg = j.groupby(["model", "dataset"], as_index=False)[["drift", "regime"]].mean()
    rows = []
    for model in MODELS:
        m = agg[agg.model == model]
        if len(m) < 3:
            continue
        w = wilcoxon(m.regime, m.drift)
        lo, hi = boot_ci(m.regime - m.drift)
        rows.append({"model": model, "n": len(m),
                     "drift_mean": m.drift.mean(), "regime_mean": m.regime.mean(),
                     "diff": (m.regime - m.drift).mean(),
                     "ci_lo": lo, "ci_hi": hi, "p": float(w.pvalue),
                     "severities": ",".join(f"{s:g}" for s in shared)})
    return pd.DataFrame(rows)


def q4_effect_matched(d: pd.DataFrame, metric: str, band=(1.02, 1.55)):
    """The control the first regime run could not provide.

    That run swept severities whose mildest level already cost 2.3x, while spike
    magnitude tops out at 1.5x. So "T5 tracks severity here but not on spikes" was
    confounded with "here the corruption is an order of magnitude more damaging".
    This restricts the regime sweep to the severities whose mean degradation lands
    inside the spike band and recomputes the slope there, against the spike slope
    computed by the same code path.

    Both sides use seeds 1-3, the seeds `edge_case_seeds.csv` contains.
    """
    if not SPIKE_CSV.exists():
        return None, None, None
    lvl = d[d.seed.isin([1, 2, 3])].groupby("severity").degr.mean()
    keep = lvl[(lvl >= band[0]) & (lvl <= band[1])].index.tolist()
    if len(keep) < 3:
        return None, keep, lvl

    reg = q1_slope(d[d.seed.isin([1, 2, 3]) & d.severity.isin(keep)])
    reg["family"] = "regime_trend (effect-matched)"

    sp = pd.read_csv(SPIKE_CSV)
    clean = (sp[sp.family == "clean"]
             .set_index(["seed", "model", "dataset"])[metric].rename("clean"))
    s = sp[sp.family == "spikes_intensity"].join(clean, on=["seed", "model", "dataset"])
    s["degr"] = s[metric] / s["clean"]
    spk = q1_slope(s)
    spk["family"] = "spikes_intensity"

    both = pd.concat([reg, spk], ignore_index=True)
    rows = []
    for fam, gf in both.groupby("family", sort=False):
        for model in MODELS:
            v = gf[gf.model == model].rho.to_numpy()
            if v.size < 3:
                continue
            lo, hi = boot_ci(v)
            rows.append({"family": fam, "model": model, "n": len(v),
                         "mean_rho": v.mean(), "ci_lo": lo, "ci_hi": hi,
                         "datasets_positive": f"{int((v > 0).sum())}/{len(v)}"})
    return pd.DataFrame(rows), keep, lvl


def main():
    metric = sys.argv[sys.argv.index("--metric") + 1] if "--metric" in sys.argv else "WQL"
    global REGIME_CSV
    if "--csv" in sys.argv:
        REGIME_CSV = OUT / sys.argv[sys.argv.index("--csv") + 1]
    # Tag every output with the input it came from. Without this the low-severity run
    # silently overwrites the high-severity run's figure and report, since both are
    # "regime" at the same metric.
    tag = REGIME_CSV.stem.replace("edge_case_regime", "").strip("_") or "high"
    if not REGIME_CSV.exists():
        sys.exit(f"missing {REGIME_CSV} -- run run_seeds.py --families regime_trend first")
    d = load_regime(metric)
    print(f"{d.dataset.nunique()} datasets x {d.seed.nunique()} seeds x "
          f"{d.severity.nunique()} severities\n")

    # ---- Q1: does the C1 split reappear? ----------------------------------
    agg = q1_slope(d)
    piv = agg.pivot(index="dataset", columns="model", values="rho").dropna()
    lines_q1 = []
    for model in MODELS:
        v = piv[model].to_numpy()
        lo, hi = boot_ci(v)
        lines_q1.append({"model": model, "n": len(v), "mean_rho": v.mean(),
                         "ci_lo": lo, "ci_hi": hi,
                         "datasets_positive": f"{int((v > 0).sum())}/{len(v)}"})
    Q1 = pd.DataFrame(lines_q1)
    # Both models can saturate at rho = 1 on every dataset, which makes every paired
    # difference exactly zero and Wilcoxon degenerate (it returns NaN). That is a result,
    # not a failure: the two models are indistinguishable because both track perfectly.
    diffs = (piv["chronos-2"] - piv["chronos-t5"]).to_numpy()
    if np.allclose(diffs, 0.0):
        p_paired, paired_note = float("nan"), (
            "not testable -- every per-dataset difference is exactly zero "
            "(both models saturate at rho = 1)")
    else:
        p_paired = float(wilcoxon(piv["chronos-2"], piv["chronos-t5"],
                                  alternative="greater").pvalue)
        paired_note = f"p = {p_paired:.3g}"
    print("Q1 -- Spearman(severity, degradation), seeds averaged within dataset")
    print(Q1.round(4).to_string(index=False))
    print(f"  paired (C2 > T5), n={len(piv)}: {paired_note}\n")

    t5_ci = (Q1.loc[Q1.model == "chronos-t5", "ci_lo"].iloc[0],
             Q1.loc[Q1.model == "chronos-t5", "ci_hi"].iloc[0])
    t5_tracks = bool(t5_ci[0] > 0)
    verdict = ("C1 GENERALISES: Chronos-T5 is decoupled from severity here too"
               if not t5_tracks else
               "C1 IS BOUNDED: Chronos-T5 does track severity on this family")
    print(f"VERDICT: {verdict}\n")

    # ---- Q2: matched pair vs drift ----------------------------------------
    Q2 = q2_matched_pair(metric)
    if Q2 is not None and len(Q2):
        print("Q2 -- regime_trend vs drift at equal origin displacement (seed 0)")
        print(Q2.round(4).to_string(index=False))
        print()

    # ---- Q3: absolute cost -------------------------------------------------
    Q3 = (d.groupby(["model", "severity"], as_index=False)
            .degr.mean().pivot(index="severity", columns="model", values="degr"))
    print("Q3 -- mean degradation by severity (x clean)")
    print(Q3.round(4).to_string())

    # ---- Q4: effect-size-matched control -----------------------------------
    Q4, keep, lvl = q4_effect_matched(d, metric)
    if Q4 is not None:
        print(f"\nQ4 -- slope restricted to severities in the spike band "
              f"(degradation 1.02-1.55x): {keep}")
        print(Q4.round(4).to_string(index=False))
        t5r = Q4[(Q4.family.str.startswith("regime")) & (Q4.model == "chronos-t5")]
        t5s = Q4[(Q4.family == "spikes_intensity") & (Q4.model == "chronos-t5")]
        if len(t5r) and len(t5s):
            closed = bool(t5r.ci_lo.iloc[0] > 0 and t5s.ci_hi.iloc[0] < 0.3)
            print(f"  confound closed: {closed} -- at matched effect size Chronos-T5's "
                  f"slope is {t5r.mean_rho.iloc[0]:+.3f} on regime vs "
                  f"{t5s.mean_rho.iloc[0]:+.3f} on spike magnitude")
    elif keep is not None:
        print(f"\nQ4 -- skipped: only {len(keep)} severities fall in the spike band "
              f"({keep}). Mean degradation per severity:\n{lvl.round(3).to_string()}")

    # ---- figure ------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
    for model, c in zip(MODELS, ("C0", "C3")):
        g = d[d.model == model].groupby(["severity", "seed"]).degr.mean().unstack()
        axes[0].errorbar(g.index, g.mean(axis=1), yerr=g.std(axis=1), marker="o",
                         capsize=3, lw=2, color=c, label=model)
    axes[0].axhline(1.0, color="grey", ls=":", lw=1.2)
    axes[0].set_xlabel("severity (MAD-units of displacement at the forecast origin)")
    axes[0].set_ylabel(f"{metric} (x clean)")
    axes[0].set_title("Persistent trend change: does error track severity?",
                      fontsize=12, fontweight="bold")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    xs = np.arange(len(MODELS))
    for i, model in enumerate(MODELS):
        v = piv[model].to_numpy()
        axes[1].scatter(np.full(len(v), i) + RNG.normal(0, 0.05, len(v)), v,
                        s=26, alpha=0.55, color="C0", zorder=3)
        lo, hi = boot_ci(v)
        axes[1].plot([i - 0.22, i + 0.22], [v.mean()] * 2, color="C3", lw=2.6, zorder=4)
        axes[1].plot([i, i], [lo, hi], color="C3", lw=1.6, zorder=4)
    axes[1].axhline(0.0, color="grey", ls=":", lw=1.2)
    axes[1].set_xticks(xs); axes[1].set_xticklabels(MODELS)
    axes[1].set_ylabel(r"Spearman $\rho$(severity, degradation)")
    axes[1].set_title("Per-dataset slope", fontsize=12, fontweight="bold")
    axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    p = OUT / f"fig_regime_{tag}_{metric}.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)

    md = [f"# E10 -- persistent trend change ({tag} severity grid, {metric})", "",
          f"Source: `{REGIME_CSV.name}`, severities "
          f"{sorted(d.severity.unique())}.", "",
          "`regime_trend` changes the trend in the last 25% of the context and keeps it "
          "changed through the forecast origin. Unlike the other six families it does not "
          "say \"past observations are wrong\" but \"the process changed and is still in "
          "the new mode\", which the model must judge rather than filter.", "",
          "Analysis unit is the dataset (n = 25); seeds averaged within dataset before "
          f"testing; CIs bootstrap the datasets ({NBOOT} draws).", "",
          "## Q1 -- does the C1 split reappear?", "",
          Q1.round(4).to_markdown(index=False), "",
          f"Paired Wilcoxon (Chronos-2 rho > Chronos-T5 rho), n = {len(piv)}: "
          f"{paired_note}", "", f"**{verdict}**", ""]
    if Q2 is not None and len(Q2):
        md += ["## Q2 -- matched against `drift` at equal origin displacement", "",
               "Both families displace the final context point by the same amount at a "
               "given severity; they differ only in whether a breakpoint is present. "
               "Seed 0 only, because the seed-0 sweep is the only run containing `drift`.",
               "", Q2.round(4).to_markdown(index=False), ""]
    md += ["## Q3 -- absolute degradation by severity", "",
           Q3.round(4).to_markdown(), ""]
    if Q4 is not None:
        md += ["## Q4 -- effect-size-matched control", "",
               "The first regime sweep started at 2.3x degradation while spike magnitude "
               "tops out at 1.5x, so a difference in slope was confounded with a "
               "difference in how damaging the corruption is. Here the regime sweep is "
               f"restricted to severities {keep}, whose mean degradation falls inside the "
               "spike band, and both slopes are computed by the same code path on seeds "
               "1-3.", "", Q4.round(4).to_markdown(index=False), ""]
    md += [f"![regime]({p.name})", ""]
    mdp = OUT / f"REGIME_ANALYSIS_{tag}_{metric}.md"
    mdp.write_text("\n".join(md), encoding="utf-8", newline="\n")
    print(f"\n-> {p}\n-> {mdp}")


if __name__ == "__main__":
    main()
