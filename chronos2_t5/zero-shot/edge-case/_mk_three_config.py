"""3-config robustness figure (all 25 datasets, identical all-corrupted protocol), TWO panels:
(top) pairwise win rate on ABSOLUTE WQL = who predicts better under corruption;
(bottom) pairwise win rate on RELATIVE degradation (WQL/clean) = who degrades least ('robustness').
Each cell carries a bootstrap 95% CI (resample the 25 datasets, 2000x)."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent   # chronos2_t5/zero-shot/edge-case
EC = HERE / "results"                     # data lives next to this script
df = pd.concat([pd.read_csv(EC/"edge_case_results.csv"), pd.read_csv(EC/"_c2cl_full.csv")], ignore_index=True)
df = df[df.family != "clean"].copy()
df["model"] = df["model"].replace({"chronos-2":"C2-uni","chronos-2-CL":"C2-CL","chronos-t5":"T5"})
cfgs = ["C2-CL","C2-uni","T5"]; n=len(cfgs)
NAVY="#1F355E"
cmap = LinearSegmentedColormap.from_list("pg", ["#7B4EA3","#F4F1F6","#1B6E5F"])

def pairwise(col):
    p = df.pivot_table(index=["dataset","family","severity"], columns="model", values=col).dropna(subset=cfgs)
    ds = p.index.get_level_values("dataset").values
    datasets = np.unique(ds); idxby={d: np.where(ds==d)[0] for d in datasets}
    rng = np.random.default_rng(0)
    boot_ds = [rng.choice(datasets, len(datasets), replace=True) for _ in range(2000)]
    boot_idx = [np.concatenate([idxby[d] for d in s]) for s in boot_ds]
    P=np.full((n,n),50.0); LO=np.full((n,n),50.0); HI=np.full((n,n),50.0)
    for i,a in enumerate(cfgs):
        for j,b in enumerate(cfgs):
            if i==j: continue
            w=(p[a].values<p[b].values).astype(float)+0.5*(p[a].values==p[b].values)
            P[i,j]=w.mean()*100
            bs=np.array([w[ii].mean()*100 for ii in boot_idx])
            LO[i,j],HI[i,j]=np.percentile(bs,[2.5,97.5])
    return P,LO,HI

panels=[("WQL","Absolute accuracy under corruption  (row beats column = lower WQL)"),
        ("WQL_degr","Relative degradation  (row beats column = degrades LESS = more 'robust')")]

fig,axes=plt.subplots(2,1,figsize=(5.6,7.7))
im=None
for ax,(col,title) in zip(axes,panels):
    P,LO,HI=pairwise(col)
    im=ax.imshow(P,cmap=cmap,vmin=15,vmax=85,aspect="auto")
    for i in range(n):
        for j in range(n):
            if i==j:
                ax.text(j,i,"—",ha="center",va="center",fontsize=13,color="#aaa")
            else:
                v=P[i,j]
                ax.text(j,i,f"{v:.0f}",ha="center",va="center",fontsize=15,fontweight="bold",
                        color="white" if (v>70 or v<30) else NAVY)
                ax.text(j,i+0.26,f"({LO[i,j]:.0f},{HI[i,j]:.0f})",ha="center",va="center",
                        fontsize=8,color="white" if (v>70 or v<30) else "#555")
    ax.set_xticks(range(n)); ax.set_xticklabels(cfgs,fontsize=10.5)
    ax.set_yticks(range(n)); ax.set_yticklabels(cfgs,fontsize=10.5)
    ax.xaxis.set_label_position("top"); ax.xaxis.tick_top()
    ax.set_title(title,fontsize=9.3,fontweight="bold",color=NAVY,pad=8)
    ax.set_xticks(np.arange(-.5,n,1),minor=True); ax.set_yticks(np.arange(-.5,n,1),minor=True)
    ax.grid(which="minor",color="white",lw=2); ax.tick_params(which="minor",length=0)
fig.suptitle("Under corruption (25 datasets): win rate with 95% CI\nC2 is more accurate · T5 degrades less (clamps outliers) · C2-uni ≈ C2-CL",
             fontsize=10.5,fontweight="bold",color=NAVY,y=1.0)
fig.tight_layout(rect=[0,0.06,1,0.95])
cax=fig.add_axes([0.25,0.035,0.5,0.018])
cb=fig.colorbar(im,cax=cax,orientation="horizontal")
cb.set_label("Win rate %  (green = row wins, purple = row loses)",fontsize=8.5,color=NAVY)
cb.ax.tick_params(labelsize=8)
out=HERE/"plots"/"robust_three_config.png"
fig.savefig(out,dpi=200,bbox_inches="tight"); plt.close(fig)
from PIL import Image
print("saved",out,Image.open(out).size)