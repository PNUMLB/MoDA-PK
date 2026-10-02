from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];B=R/'results/runs/SCALE_SHAPE/20261002';S=R/'results/summaries/scale_shape_20261002';S.mkdir(exist_ok=False)
d=pd.read_parquet(B/'predictions.parquet');old=pd.read_parquet(R/'results/summaries/ten_seeds_20261001/canonical_predictions.parquet')
keys=['fold','seed','source_id','drug_id','participant_id','profile_id','observation_id']
o=old[(old.condition=='molecular_relation')&(old.k==2)&(old.protocol=='first_k')].copy()
a=d[d.condition=='mode_original'].merge(o,on=keys,validate='one_to_one',suffixes=('_new','_old'))
assert len(a)==len(o)==len(d[d.condition=='mode_original'])
assert np.max(abs(a.y_true_log_new-a.y_true_log_old))<1e-12
maxdiff=float(np.max(abs(a.y_pred_log_new-a.y_pred_log_old)));assert maxdiff<1e-9,maxdiff
assert not d.duplicated(['condition']+keys).any()
assert np.isfinite(d[['y_true_log','y_pred_log']]).all().all()
for c,g in d.groupby('condition'):assert len(g)==len(o)
d=d.merge(o[keys+['cutoff_time']],on=keys,validate='many_to_one');d['error']=d.y_pred_log-d.y_true_log;d['sq']=d.error**2;d['track']=np.where(d.fold=='known','Known-drug','Joint drug-source')
pkeys=['track','condition','seed','source_id','drug_id','participant_id','profile_id']
p=d.groupby(pkeys).agg(mse=('sq','mean'),bias=('error','mean'),n=('error','size')).reset_index();p['centered_mse']=p.mse-p.bias**2
base=p[p.condition=='mode_original'].drop(columns='condition');pair=p.merge(base,on=[k for k in pkeys if k!='condition'],suffixes=('','_original'),validate='many_to_one');pair['delta_mse']=pair.mse-pair.mse_original;pair.to_csv(S/'profile_paired_changes.csv',index=False)
# Profiles are descriptive; a second aggregate balances participants then studies.
neg=pair.groupby(['track','condition','seed']).agg(worsened_pct=('delta_mse',lambda x:100*(x>1e-10).mean()),mean_delta=('delta_mse','mean'));neg.to_csv(S/'profile_harm_by_seed.csv')
p.to_csv(S/'scale_shape_error_decomposition.csv',index=False)
d['horizon']=d.query_time-d.cutoff_time;d['horizon_bin']=pd.cut(d.horizon,[0,6,24,72,np.inf],labels=['0-6 h','6-24 h','24-72 h','>72 h'])
h=d.groupby(['track','condition','seed','drug_id','participant_id','horizon_bin'],observed=True).agg(mse=('sq','mean'),bias=('error','mean'),queries=('error','size')).reset_index();h.to_csv(S/'horizon_participant_metrics.csv',index=False)
for name in ['summary.csv','metrics_by_seed.csv','drug_summary.csv','drug_metrics_by_seed.csv']:(S/name).write_bytes((B/name).read_bytes())
f=pd.read_csv(B/'drug_summary.csv');labels={'mode_original':'MoDE-PK','prior_scale':'Prior + scale','mode_scale':'MoDE-PK + scale','mode_scale_shape':'MoDE-PK + scale/rate','prior_only':'Prior only'}
fig,axs=plt.subplots(1,2,figsize=(12,5))
for ax,(track,g) in zip(axs,f.groupby('track',sort=False)):
 t=g.pivot(index='drug_id',columns='condition',values='log_rmse');delta=t[['prior_scale','mode_scale','mode_scale_shape']].sub(t.mode_original,axis=0)
 im=ax.imshow(delta,cmap='RdBu_r',vmin=-3,vmax=3,aspect='auto');ax.set_yticks(range(len(delta)),delta.index);ax.set_xticks(range(3),[labels[c] for c in delta],rotation=25,ha='right');ax.set_title(track+'\nChange in drug log-RMSE (negative improves)')
 for i in range(len(delta)):
  for j in range(3):ax.text(j,i,f'{delta.iloc[i,j]:+.2f}',ha='center',va='center',fontsize=8)
fig.tight_layout();fig.savefig(S/'drug_changes.png',dpi=180);fig.savefig(S/'drug_changes.pdf');plt.close(fig)
# Median original profile MSE across seeds; same observed query IDs in all curves.
fig,axs=plt.subplots(1,2,figsize=(12,4))
selected=[]
for ax,drug in zip(axs,['warfarin','xanthohumol']):
 g=base[(base.track=='Joint drug-source')&(base.drug_id==drug)];m=g.groupby(['source_id','participant_id','profile_id']).mse.mean().sort_values();key=m.index[len(m)//2];selected.append(dict(drug=drug,key=list(key),rule='Median original profile MSE averaged over ten seeds; post hoc descriptive'))
 q=d[(d.drug_id==drug)&(d.source_id==key[0])&(d.participant_id==key[1])&(d.profile_id==key[2])&(d.track=='Joint drug-source')]
 truth=q[q.condition=='mode_original'].groupby('observation_id').agg(t=('query_time','first'),y=('y_true_log','first')).sort_values('t');ax.scatter(truth.t,truth.y,c='black',marker='x',s=45,label='Observed future')
 for c in ['mode_original','prior_scale','mode_scale','mode_scale_shape']:
  t=q[q.condition==c].groupby('observation_id').agg(t=('query_time','first'),y=('y_pred_log','mean')).sort_values('t');ax.plot(t.t,t.y,'o-',label=labels[c],ms=3)
 ax.set_title(drug);ax.set_xlabel('Time after recorded origin (h)');ax.set_ylabel('Natural log concentration (umol/L)');ax.legend(fontsize=7)
fig.suptitle('Query-point connections; mean log predictions across 10 seeds (not dense trajectories)',fontsize=10);fig.tight_layout();fig.savefig(S/'failure_cases.png',dpi=180);fig.savefig(S/'failure_cases.pdf');plt.close(fig)
with pd.ExcelWriter(S/'scale_shape_results.xlsx') as w:
 for name in ['summary','drug_summary','metrics_by_seed','drug_metrics_by_seed']:pd.read_csv(S/f'{name}.csv').to_excel(w,sheet_name=name,index=False)
 neg.reset_index().to_excel(w,sheet_name='profile_harm',index=False)
man=dict(status='completed',runs=100,seeds=list(range(1,11)),k=2,baseline_max_absolute_log_difference=maxdiff,baseline_query_rows=len(o),case_selection=selected,scope='Frozen-checkpoint exploratory support calibration; not new training or independent validation',files={str(x.relative_to(R)):hashlib.sha256(x.read_bytes()).hexdigest() for x in B.glob('*/predictions.parquet')})
(S/'analysis_manifest.json').write_text(json.dumps(man,indent=2))
print(pd.read_csv(S/'summary.csv').to_string(index=False));print(f[f.drug_id.isin(['warfarin','xanthohumol'])].to_string(index=False))
