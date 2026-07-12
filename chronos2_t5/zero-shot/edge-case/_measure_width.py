"""Measure 80% prediction-interval width (q0.9 - q0.1) for C2-uni vs T5 on clean zero-shot,
all 25 datasets. Scale-free comparison via per-series ratio T5/C2. Tests the claim
'T5 has wider uncertainty'. Writes _interval_width.csv."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, torch
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import run_edge_cases as E
R2 = E.R2
Q = R2.QUANTILES; I10 = Q.index(0.1); I90 = Q.index(0.9)

def raw_fc(pipe, ctxs, H, kind):
    out = []
    for b0 in range(0, len(ctxs), E.BATCH):
        ch = ctxs[b0:b0+E.BATCH]; kw = dict(prediction_length=H, quantile_levels=Q)
        if kind == "t5": torch.manual_seed(E.SEED); kw["num_samples"] = 20
        else: kw["limit_prediction_length"] = False
        q,_ = pipe.predict_quantiles([torch.tensor(np.asarray(c,np.float32)) for c in ch], **kw)
        for qi in q:
            a = qi.cpu().numpy() if torch.is_tensor(qi) else np.asarray(qi)
            if a.ndim == 3: a = a[0]
            out.append(a)   # (H, nq)
    return out

pipes = E._load_pipes(); c2,_ = pipes["chronos-2"]; t5,_ = pipes["chronos-t5"]
rows = []
for di,(cfg,H) in enumerate(E.EDGE_DATASETS,1):
    _, ctxs, starts = E.build_dataset(cfg, H)
    q2 = raw_fc(c2, ctxs, H, "c2"); q5 = raw_fc(t5, ctxs, H, "t5")
    for i in range(len(ctxs)):
        w2 = np.mean(q2[i][:,I90]-q2[i][:,I10])
        w5 = np.mean(q5[i][:,I90]-q5[i][:,I10])
        if w2 > 1e-9:
            rows.append({"dataset":cfg, "w_c2":w2, "w_t5":w5, "ratio_t5_over_c2":w5/w2})
    print(f"[{di}/25] {cfg}", flush=True)
    del ctxs
df = pd.DataFrame(rows)
df.to_csv(HERE/"results"/"_interval_width.csv", index=False)
# summary
g = df.groupby("dataset")["ratio_t5_over_c2"].median()
print("\nmedian T5/C2 interval-width ratio per dataset:")
print(g.round(2).sort_values(ascending=False).to_string())
print(f"\ndatasets where T5 wider (median ratio>1): {(g>1).sum()}/{len(g)}")
print(f"overall median ratio T5/C2: {df['ratio_t5_over_c2'].median():.2f}")
print(f"series-level: T5 wider in {(df['w_t5']>df['w_c2']).mean()*100:.0f}% of series")