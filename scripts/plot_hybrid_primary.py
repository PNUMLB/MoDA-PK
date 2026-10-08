"""New Ours everywhere: architecture, matched trajectories, drug metrics and horizons."""
from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.backends.backend_pdf import PdfPages
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'src')]
from mhap_pk.data.traceable import load_profiles
S=Path(sys.argv[sys.argv.index('--summary')+1]).resolve() if '--summary' in sys.argv else R/'results/summaries/hybrid_primary_20261002';p=pd.read_parquet(S/'primary_predictions.parquet');x=p[(p.protocol=='first_k')&(p.k==2)]
mods=['ours','poppk_vi_map','dcm','unipk','deepct'];names=dict(zip(mods,['MoDE-PK (Ours)','PopPK + MAP','DCM','Uni-PK','DeepCt']));colors=dict(zip(mods,['#b2182b','#2166ac','#8856a7','#1b7837','#d95f02']))
label=lambda d:d.replace('_',' ').capitalize();keys=['source_id','drug_id','participant_id','profile_id']
if (S/'plot_support_doses_LOCAL_ONLY.json').exists():
 from types import SimpleNamespace
 ps=json.loads((S/'plot_support_doses_LOCAL_ONLY.json').read_text())['profiles']
 for a in ps:a['episodes']={2:SimpleNamespace(support=pd.DataFrame(a['support_k2']))}
else:ps,meta,_=load_profiles()
lookup={tuple(a[k] for k in keys):a for a in ps}
# Locked profiles from the preceding revision: no selection using the adopted model's test errors.
casefile=S/'main_curve_selection_LOCAL_ONLY.csv' if (S/'main_curve_selection_LOCAL_ONLY.csv').exists() else R/'results/summaries/molecular_identity_20261002/main_curve_selection_LOCAL_ONLY.csv';cases=pd.read_csv(casefile);cases=cases[cases.track=='Known-drug'];limits={};checks=[];score=[]
for _,c in cases.iterrows():
 g=x[(x[keys]==c[keys]).all(axis=1)];s=lookup[tuple(c[k] for k in keys)]['episodes'][2].support;v=np.exp(np.r_[g.groupby(['track','condition','observation_id']).y_pred_log.mean(),g.y_true_log,s.target_log_dv]);limits[c.drug_id]=(v.min()/1.5,v.max()*1.5)

def curves(cs,track,figsize):
 fig,axs=plt.subplots(int(np.ceil(len(cs)/2)),2,figsize=figsize,squeeze=False)
 for ax,(_,c) in zip(axs.flat,cs.iterrows()):
  g=x[(x.track==track)&(x[keys]==c[keys]).all(axis=1)];profile=lookup[tuple(c[k] for k in keys)];support=profile['episodes'][2].support.groupby('time').target_log_dv.mean();truth=g[(g.condition=='ours')&(g.seed==1)].sort_values('query_time');assert len(truth)>0
  ref=truth.set_index('observation_id')[['query_time','y_true_log','support_end_time']].sort_index()
  for model in mods:
   q=g[g.condition==model]
   if q.empty:continue
   assert set(q.seed)==set(range(1,11))
   for seed,h in q.groupby('seed'):pd.testing.assert_frame_equal(ref,h.set_index('observation_id')[ref.columns].sort_index())
   a=q.groupby('observation_id').agg(t=('query_time','first'),yp=('y_pred_log','mean'),yt=('y_true_log','first')).sort_values('t');ax.plot(a.t,np.exp(a.yp),color=colors[model],lw=2 if model=='ours' else 1.4,marker='.',ms=2,label=names[model]);score.append(dict(track=track,drug=c.drug_id,profile=c.profile_id,model=model,ensemble_log_rmse=float(np.sqrt(np.mean((a.yp-a.yt)**2)))))
  ax.scatter(support.index,np.exp(support),c='black',marker='s',s=28,zorder=6);ax.scatter(truth.query_time,np.exp(truth.y_true_log),facecolors='white',edgecolors='black',s=27,zorder=7);ax.axvline(support.index.max(),color='black',ls='--',lw=.8)
  for time,amount,route,duration in profile['events']:
   ax.axvline(time,color='gray',alpha=.35,lw=.7)
   if duration>0:ax.axvspan(time,time+duration,color='gray',alpha=.12)
  ax.set(yscale='log',xlabel='Time (h)',ylabel='Concentration (µmol/L)',title=label(c.drug_id));ax.grid(alpha=.15)
  if c.drug_id in limits:ax.set_ylim(limits[c.drug_id])
  checks.append(dict(track=track,drug_id=c.drug_id,profile_id=c.profile_id,models=g.condition.nunique(),seeds=10,queries=len(truth)))
 for ax in list(axs.flat)[len(cs):]:ax.axis('off')
 mm=mods if track=='Known-drug' else [m for m in mods if m!='poppk_vi_map']
 handles=[Line2D([0],[0],color=colors[m],lw=2,label=names[m]) for m in mm]+[Line2D([0],[0],color='black',marker='s',ls='',label='Observed history'),Line2D([0],[0],color='black',marker='o',mfc='white',ls='',label='Future measurements')]
 fig.legend(handles=handles,loc='upper center',ncol=4,fontsize=8);fig.suptitle(track+' | k=2 | ten-seed geometric ensemble',y=.94,fontsize=11);fig.tight_layout(rect=(0,0,1,.91));return fig
for track,name in [('Known-drug','Known_PK_curves'),('Joint drug–source','Joint_PK_curves')]:
 fig=curves(cases,track,(11,10));fig.savefig(S/(name+'.pdf'));fig.savefig(S/(name+'.png'),dpi=170);plt.close(fig)
cases.to_csv(S/'main_curve_selection_LOCAL_ONLY.csv',index=False);pd.DataFrame(checks).to_csv(S/'main_curve_checks.csv',index=False);pd.DataFrame(score).to_csv(S/'main_curve_scores.csv',index=False)
# k/count and fixed-origin use identical selected Ours; no legacy head scores.
c=pd.read_csv(S/'primary_cohort_summary.csv')
for protocol,name in [('first_k','Initial_history'),('common_origin','Fixed_origin')]:
 fig,axs=plt.subplots(1,2,figsize=(10,4))
 for ax,track in zip(axs,['Known-drug','Joint drug–source']):
  for m in mods:
   g=c[(c.condition==m)&(c.track==track)&(c.protocol==protocol)].sort_values('k')
   if not g.empty:ax.errorbar(g.k,g.sb_hrmse_mean,yerr=g.sb_hrmse_std,marker='o',ms=4,capsize=2,color=colors[m],label=names[m])
  ax.set(xlabel='Observed concentration times (k)',ylabel='SB-hRMSE',title=track);ax.set_xticks(range(5));ax.grid(alpha=.2)
 handles,labs=axs[0].get_legend_handles_labels();fig.legend(handles,labs,loc='lower center',ncol=5,fontsize=8);fig.tight_layout(rect=(0,.08,1,1));fig.savefig(S/(name+'.pdf'));plt.close(fig)
# All-drug horizon atlas, equal participant weighting within a drug.
h=pd.read_csv(S/'horizon_by_seed.csv');bins=['0–1','1–4','4–12','12–24','>24'];hs=h.groupby(['track','condition','drug_id','horizon_bin'],observed=True).log_rmse.agg(['mean','std']).reset_index()
with PdfPages(S/'Drug_horizon_atlas.pdf') as pdf:
 for drug in sorted(x.drug_id.unique()):
  fig,axs=plt.subplots(1,2,figsize=(11,4.5))
  for ax,track in zip(axs,['Known-drug','Joint drug–source']):
   for m in mods:
    t=hs[(hs.drug_id==drug)&(hs.track==track)&(hs.condition==m)].set_index('horizon_bin').reindex(bins)
    if t['mean'].notna().any():ax.errorbar(range(5),t['mean'],yerr=t['std'],color=colors[m],label=names[m],marker='o',capsize=2)
   counts=h[(h.drug_id==drug)&(h.track==track)&(h.condition=='ours')&(h.seed==1)].set_index('horizon_bin').reindex(bins)
   ax.set_xticks(range(5),[b+'\n'+(f'{int(counts.loc[b,"participants"])} / {int(counts.loc[b,"queries"])}' if pd.notna(counts.loc[b,'queries']) else '—') for b in bins]);ax.set(title=track,xlabel='Horizon (h); participants / queries',ylabel='Drug log-RMSE');ax.grid(alpha=.2)
  handles,labs=axs[0].get_legend_handles_labels();fig.legend(handles,labs,loc='lower center',ncol=5,fontsize=8);fig.suptitle(label(drug));fig.tight_layout(rect=(0,.07,1,.95));pdf.savefig(fig);plt.close(fig)
# Vector architecture: exact adopted inputs, no active relation branch.
fig,ax=plt.subplots(figsize=(10,6));ax.set(xlim=(0,11),ylim=(0,6));ax.axis('off')
def box(x,y,w,h,text,color='#e7eff7'):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08',facecolor=color,edgecolor='#40566c'));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=14)
def arrow(a,b):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',lw=1.3,color='#40566c'))
box(.2,4.8,2.7,.85,'Molecular structure\nFrozen Uni-Mol (512)')
box(.2,3.5,2.7,.85,'Drug identity (512)\nUnseen: mean training\nidentity embedding')
box(3.6,4.1,2.9,1,'Concatenate (1024)\nShared context (32)\nLinear + SiLU')
box(7.3,4.1,3.3,1,'Drug-conditioned prior\nFive effective\nlog-PK parameters')
arrow((2.98,5.22),(3.52,4.8));arrow((2.98,3.92),(3.52,4.35));arrow((6.58,4.6),(7.22,4.6))
box(.2,1.65,2.7,1.1,'Recorded dose events\nTime, amount, route\nInfusion duration')
box(3.6,1.65,2.9,1.1,'Compartment states\nAnalytic dose-event\npropagation')
box(7.3,1.65,3.3,1.1,'GRU observer (32)\nHistory + drug context\nCL/Vc and amount updates')
arrow((2.98,2.2),(3.52,2.2));arrow((8.95,4.02),(8.95,2.83));arrow((8.15,4.02),(5.65,2.83));arrow((5.05,4.02),(7.7,2.83));arrow((6.58,2.2),(7.22,2.2))
box(.2,.1,2.7,.8,'Observed concentrations\nAvailable history only','#f3eee1');arrow((2.98,.5),(7.22,1.8))
box(7.3,.1,3.3,.8,'Future concentration\nFinal parameters + state\nRecorded doses','#e7f3e6');arrow((8.95,1.57),(8.95,.98))
ax.annotate('Next observation',xy=(5.05,1.57),xytext=(8.3,1.1),arrowprops=dict(arrowstyle='->',connectionstyle='angle3,angleA=0,angleB=-90',color='#40566c'),fontsize=8,ha='center')
fig.tight_layout();fig.savefig(S/'Figure1.pdf');fig.savefig(S/'Figure1.png',dpi=170);plt.close(fig)
print('Updated Ours curves, histories, horizons and architecture')
