"""Add Chronos-2 CROSS-LEARNING to the main edge-case sweep, IDENTICAL protocol to
run_edge_cases (ALL series corrupted, full severity grid, all 25 datasets), so C2-uni /
C2-CL / T5 can be compared on equal footing. Writes _c2cl_full.csv (same columns as
edge_case_results.csv). Cross-learning groups the (corrupted) series in batches of 100."""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_edge_cases as E
R2 = E.R2
from chronos import BaseChronosPipeline


def forecast_cl_all(pipe, contexts, starts, horizon):
    """Forecast every (corrupted) context with cross_learning=True, grouped in batches of 100."""
    B = R2.CROSS_LEARNING_BATCH
    fc = []
    for b0 in range(0, len(contexts), B):
        grp = [torch.tensor(np.asarray(c, np.float32)) for c in contexts[b0:b0 + B]]
        q, _ = pipe.predict_quantiles(grp, prediction_length=horizon, quantile_levels=R2.QUANTILES,
                                      cross_learning=True, batch_size=B, limit_prediction_length=False)
        for qi, st in zip(q, starts[b0:b0 + B]):
            a = qi.cpu().numpy() if torch.is_tensor(qi) else np.asarray(qi)
            if a.ndim == 3: a = a[0]
            fc.append(R2._quantile_forecast(np.asarray(a, np.float32), st))
    return fc


def main():
    pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-2",
              device_map="cuda" if torch.cuda.is_available() else "cpu", torch_dtype=R2.DTYPE)
    cuda = torch.cuda.is_available()
    rows = []
    for di, (config, horizon) in enumerate(E.EDGE_DATASETS, 1):
        test_data, contexts, starts = E.build_dataset(config, horizon)
        print(f"[{di}/{len(E.EDGE_DATASETS)}] {config} (n={len(contexts)}, H={horizon})", flush=True)
        for (fam, sev) in E.CONDITIONS:
            pc = E.perturb_contexts(config, contexts, fam, sev)
            t0 = time.perf_counter()
            fcs = forecast_cl_all(pipe, pc, starts, horizon)
            if cuda: torch.cuda.synchronize()
            lat = time.perf_counter() - t0
            mase, wql = R2.evaluate(fcs, test_data)
            rows.append({"dataset": config, "model": "chronos-2-CL", "family": fam,
                         "severity": sev, "MASE": mase, "WQL": wql,
                         "n_series": len(contexts), "latency_s": round(lat, 3)})
        del contexts, test_data, starts
    df = pd.DataFrame(rows)
    df = E._add_degradation(df)
    out = HERE / "results" / "_c2cl_full.csv"
    df.to_csv(out, index=False)
    print("saved ->", out)


if __name__ == "__main__":
    main()