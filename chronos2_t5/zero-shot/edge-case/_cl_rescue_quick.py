"""QUICK cross-learning robustness probe: does a CLEAN group of related series rescue a
CORRUPTED target? For each dataset we corrupt ONE target at a time, keep its group-mates
clean, and forecast it three ways — C2-CL (target in clean group), C2-uni (target alone),
T5 (target alone). Reuses run_edge_cases helpers so corruption + eval are identical to the
main study. Writes _cl_rescue_quick.csv."""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from gluonts.dataset.split import split
import datasets as hfds

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_edge_cases as E          # sets up C2 src path, imports R2, P, helpers
R2, P = E.R2, E.P

hfds.logging.set_verbosity_error(); hfds.disable_progress_bars()

# FULL benchmark: all 25 datasets (some have unrelated series -> honest full-picture average)
DATASETS = [cfg for cfg, _ in E.EDGE_DATASETS]
SEV = {"spikes_intensity": 20.0, "spikes_density": 0.20, "drift": 12.0,
       "drift_step": 12.0, "gap": 0.40, "gap_boundary": 0.40}
K = 10                              # target series scored per (dataset, family)
HORIZ = dict(E.EDGE_DATASETS)


def load(config, H):
    ds = hfds.load_dataset(R2.HF_REPO if hasattr(R2, "HF_REPO") else E.HF_REPO, config, split="train")
    ds.set_format("numpy")
    gts = R2.to_gluonts_univariate(ds, E.EDGE_MAX_SERIES)
    _, tt = split(gts, offset=-H)
    ti = list(tt.generate_instances(H, windows=1).input)
    contexts = [np.asarray(e["target"], np.float32) for e in ti]
    starts = [e["start"] + len(e["target"]) for e in ti]
    return gts, contexts, starts


def subset_td(gts, idx, H):
    sub = [gts[i] for i in idx]
    _, tt = split(sub, offset=-H)
    return tt.generate_instances(H, windows=1)


def main():
    pipes = E._load_pipes()
    c2, _ = pipes["chronos-2"]; t5, _ = pipes["chronos-t5"]
    rows = []
    for config in DATASETS:
        H = HORIZ[config]
        gts, contexts, starts = load(config, H)
        n = len(contexts)
        idx = sorted(set(np.linspace(0, n - 1, min(K, n)).astype(int).tolist()))
        td = subset_td(gts, idx, H)
        sidx = [starts[i] for i in idx]; cidx = [contexts[i] for i in idx]
        # clean references per config (same K targets)
        base = {
            "C2-uni": R2.evaluate(E.forecast(c2, cidx, sidx, H, "c2"), td),
            "T5":     R2.evaluate(E.forecast(t5, cidx, sidx, H, "t5"), td),
            "C2-CL":  R2.evaluate([E.forecast_cl_one(c2, contexts, i, contexts[i], starts[i], H) for i in idx], td),
        }
        print(f"{config}: n={n}, K={len(idx)}, H={H}", flush=True)
        for fam, sev in SEV.items():
            rng = E._rng(config, fam, sev)
            corr = [P.apply(fam, contexts[i], rng, sev) for i in idx]     # corrupt each target once
            fc = {
                "C2-uni": E.forecast(c2, corr, sidx, H, "c2"),
                "T5":     E.forecast(t5, corr, sidx, H, "t5"),
                "C2-CL":  [E.forecast_cl_one(c2, contexts, i, corr[k], starts[i], H) for k, i in enumerate(idx)],
            }
            for label, fcs in fc.items():
                mase, wql = R2.evaluate(fcs, td)
                bm, bw = base[label]
                rows.append({"dataset": config, "family": fam, "severity": sev, "model": label,
                             "MASE": mase, "WQL": wql, "MASE_degr": mase / bm, "WQL_degr": wql / bw,
                             "n": n, "K": len(idx)})
        del gts, contexts, starts
    out = HERE / "results" / "_cl_rescue_full.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("saved ->", out)


if __name__ == "__main__":
    main()