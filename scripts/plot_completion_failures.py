from pathlib import Path
import sys
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from mhap_pk.data.traceable import load_profiles
O=R/'results/summaries/completion_suite_20261002';p=pd.read_parquet(O/'stage_predictions_LOCAL_ONLY.parquet');b=pd.read_parquet(R/'results/summaries/ten_seeds_20261001/canonical_predictions.parquet');b=b[(b.k==2)&(b.protocol=='first_k')&(b.track=='Joint drug–source')];cases=pd.read_csv(R/'results/summaries/drug_first_editorial_20261002/case_selection_LOCAL_ONLY.csv');cases=cases[cases.track=='Joint drug–source'];keys=['source_id','drug_id','participant_id','profile_id'];ps,_,_=load_profiles();lookup={tuple(x[k] for k in keys):x for x in ps};fig,axs=plt.subplots(1,2,figsize=(11,5.1))
for ax,(_,c) in zip(axs,cases.iterrows()):
 g=p[(p[keys]==c[keys]).all(axis=1)&(p.fold!='known')];profile=lookup[tuple(c[k] for k in keys)];support=profile['episodes'][2].support.groupby('time').target_log_dv.mean();truth=g[g['mode']=='full'].query('seed==1').sort_values('query_time')
 for mode,name,col,ls in [('same_checkpoint_prior','Same-checkpoint prior','#777777','--'),('full','MoDE-PK','#b2182b','-'),('state_path_only','State pathway only','#2166ac','-'),('parameter_path_only','Parameter pathway only','#8856a7',':')]:
  x=g[g['mode']==mode].groupby('observation_id').agg(time=('query_time','first'),y=('y_pred_log','mean')).sort_values('time');ax.plot(x.time,np.exp(x.y),color=col,ls=ls,lw=1.8,label=name)
 mode='unipk' if c.drug_id=='warfarin' else 'deepct';x=b[(b[keys]==c[keys]).all(axis=1)&(b.condition==mode)].groupby('observation_id').agg(time=('query_time','first'),y=('y_pred_log','mean')).sort_values('time');ax.plot(x.time,np.exp(x.y),color='#1b7837' if mode=='unipk' else '#d95f02',lw=1.8,label='Uni-PK' if mode=='unipk' else 'DeepCt');ax.scatter(support.index,np.exp(support),c='black',marker='s',s=30,label='Observed history');ax.scatter(truth.query_time,np.exp(truth.y_true_log),facecolors='white',edgecolors='black',s=28,label='Future measurements');ax.axvline(support.index.max(),color='black',ls='--',lw=.6);ax.set(yscale='log',xlabel='Time (h)',ylabel='Concentration (µmol/L)',title=c.drug_id.capitalize());ax.grid(alpha=.2)
 for time,amount,route,duration in profile['events']:
  ax.axvline(time,color='gray',alpha=.4,lw=.7)
  if duration>0:ax.axvspan(time,time+duration,color='gray',alpha=.12)
handles={}
for ax in axs:
 for h,label in zip(*ax.get_legend_handles_labels()):handles.setdefault(label,h)
fig.legend(handles.values(),handles.keys(),loc='upper center',ncol=4,fontsize=8,frameon=False)
fig.tight_layout(rect=(0,0,1,.87));fig.savefig(O/'Failure_update_paths.pdf');fig.savefig(O/'Failure_update_paths.png',dpi=150)
