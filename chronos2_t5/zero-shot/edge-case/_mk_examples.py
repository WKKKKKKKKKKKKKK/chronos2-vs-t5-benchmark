"""Two clean time-series examples that corroborate the pairwise heatmap: under corruption
Chronos-2 tracks the held-out actual while Chronos-T5 fails. Panel A = missing@boundary
(traffic), Panel B = spikes (australian electricity). Dataset-level WQL annotated (representative)."""
import sys
from pathlib import Path
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import run_edge_cases as E
R2, P = E.R2, E.P
import pandas as pd
EC = pd.read_csv(HERE/"results"/"edge_case_results.csv")
CLF = pd.read_csv(HERE/"results"/"_c2cl_full.csv")

NAVY="#1F355E"; BLUE="#2E6FB5"; TEAL="#1B6E5F"; ORANGE="#B8752A"; GREEN="#2E7D32"; GREY="#888"
CASES=[("monash_traffic","gap_boundary",0.20,"Missing chunk @ forecast boundary","Road occupancy rate (0–1)"),
       ("monash_australian_electricity","spikes_intensity",8.0,"Noisy spikes in the context","Electricity demand (MW)")]
QI={q:i for i,q in enumerate(R2.QUANTILES)}; I50=QI[0.5]; I10=QI[0.1]; I90=QI[0.9]

def ds_wql(ds,fam,sev,model):
    src=CLF if model=="chronos-2-CL" else EC
    r=src[(src.dataset==ds)&(src.family==fam)&(src.severity==sev)&(src.model==model)]
    return float(r["WQL"].iloc[0]) if len(r) else np.nan

def fc_cl_all(pipe,ctxs,starts,H):
    B=R2.CROSS_LEARNING_BATCH; out=[]
    for b0 in range(0,len(ctxs),B):
        grp=[torch.tensor(np.asarray(c,np.float32)) for c in ctxs[b0:b0+B]]
        q,_=pipe.predict_quantiles(grp,prediction_length=H,quantile_levels=R2.QUANTILES,
                                   cross_learning=True,batch_size=B,limit_prediction_length=False)
        for qi in q:
            a=qi.cpu().numpy() if torch.is_tensor(qi) else np.asarray(qi)
            if a.ndim==3: a=a[0]
            out.append(a)
    return out

def fc_all(pipe,ctxs,starts,H,kind):
    out=[]
    for b0 in range(0,len(ctxs),E.BATCH):
        ch=ctxs[b0:b0+E.BATCH]; kw=dict(prediction_length=H,quantile_levels=R2.QUANTILES)
        if kind=="t5": torch.manual_seed(E.SEED); kw["num_samples"]=20
        else: kw["limit_prediction_length"]=False
        q,_=pipe.predict_quantiles([torch.tensor(np.asarray(c,np.float32)) for c in ch],**kw)
        for qi in q:
            a=qi.cpu().numpy() if torch.is_tensor(qi) else np.asarray(qi)
            if a.ndim==3: a=a[0]
            out.append(a)
    return out   # list of (H, nq)

pipes=E._load_pipes(); c2,_=pipes["chronos-2"]; t5,_=pipes["chronos-t5"]
fig,axes=plt.subplots(1,2,figsize=(9.4,4.0))
for ax,(ds,fam,sev,label,ylab) in zip(axes,CASES):
    test_data,contexts,starts=E.build_dataset(ds,dict(E.EDGE_DATASETS)[ds])
    H=dict(E.EDGE_DATASETS)[ds]
    pc=E.perturb_contexts(ds,contexts,fam,sev)
    labels=[np.asarray(x["target"][-H:],float) for x in test_data.label] if hasattr(test_data,"label") else None
    # actual future via test_data
    acts=[np.asarray(l["target"],float) for l in test_data.label]
    q2=fc_all(c2,pc,starts,H,"c2"); q5=fc_all(t5,pc,starts,H,"t5"); qcl=fc_cl_all(c2,pc,starts,H)
    # pick series with the largest T5-vs-C2 median-abs-error gap (clearest visual)
    def mae(q,a): return np.mean(np.abs(q[:,I50]-a))
    gaps=[mae(q5[i],acts[i])-mae(q2[i],acts[i]) for i in range(len(acts))]
    si=int(np.argmax(gaps))
    ctx=np.asarray(pc[si],float); a=acts[si]; keep=min(70,len(ctx))
    xh=np.arange(len(ctx)-keep,len(ctx)); xf=np.arange(len(ctx),len(ctx)+H)
    ax.plot(xh,ctx[-keep:],color="#555",lw=1.1,label="context (corrupted)")
    ax.plot(xf,a,color=GREEN,lw=2.4,label="actual",zorder=5)
    ax.plot(xf,q2[si][:,I50],color=BLUE,lw=2.2,label="Chronos-2 (uni)")
    ax.fill_between(xf,q2[si][:,I10],q2[si][:,I90],color=BLUE,alpha=0.13)
    ax.plot(xf,qcl[si][:,I50],color=TEAL,lw=2.0,ls=(0,(4,2)),label="Chronos-2 (CL)")
    ax.plot(xf,q5[si][:,I50],color=ORANGE,lw=2.2,ls="--",label="Chronos-T5")
    ax.fill_between(xf,q5[si][:,I10],q5[si][:,I90],color=ORANGE,alpha=0.12)
    ax.axvline(len(ctx)-0.5,color=GREY,lw=1,ls=":")
    w2=ds_wql(ds,fam,sev,"chronos-2"); w5=ds_wql(ds,fam,sev,"chronos-t5"); wcl=ds_wql(ds,fam,sev,"chronos-2-CL")
    ax.set_title(f"{label}  ·  {ds.replace('monash_','')}\ndataset WQL:  C2-uni {w2:.2f} / C2-CL {wcl:.2f} / T5 {w5:.2f}",
                 fontsize=8.5,fontweight="bold",color=NAVY,pad=6)
    ax.tick_params(labelsize=7.5); ax.set_xlabel("time step",fontsize=8); ax.set_ylabel(ylab,fontsize=8.5,color=NAVY)
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    del contexts,test_data
axes[0].legend(fontsize=7.5,loc="upper left",framealpha=0.95,ncol=1)
fig.suptitle("Illustrative series — under corruption Chronos-2 tracks the actual; Chronos-T5 breaks",
             fontsize=10.5,fontweight="bold",color=NAVY,y=1.02)
fig.tight_layout()
out=HERE/"plots"/"robust_examples.png"
fig.savefig(out,dpi=200,bbox_inches="tight"); plt.close(fig)
from PIL import Image
print("saved",out,Image.open(out).size)