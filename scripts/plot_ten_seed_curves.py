"""All-seed PK curves and separate geometric-ensemble evaluation from saved predictions."""
from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from mhap_pk.evaluation.drug_metrics import balanced_metrics
I=R/'results/summaries/training_scope_extension_20261001';C=R/'results/summaries/comparison_refresh_20260929';O=R/'results/summaries/ten_seed_curves_20261002'
KEYS=['source_id','drug_id','participant_id','profile_id'];Q=KEYS+['observation_id'];MODS=['single_drug','poppk_vi_map','dcm','unipk','deepct'];NAMES=['MoDE-PK (Ours)','PopPK + MAP','DCM','Uni-PK','DeepCt'];COLORS=['#b2182b','#2166ac','#8856a7','#1b7837','#d95f02'];SEEDS=set(range(1,11))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 O.mkdir(parents=True,exist_ok=False)
 p=pd.read_parquet(I/'single_drug_predictions.parquet',filters=[('k','==',2)]);assert set(p.protocol)=={'first_k'};assert set(p.condition)==set(MODS);assert not p.duplicated(['condition','seed']+Q).any();assert np.isfinite(p[['y_true_log','y_pred_log']]).all().all() and np.isfinite(np.exp(p[['y_true_log','y_pred_log']])).all().all()
 ref=None;checks=[]
 for (mode,seed),g in p.groupby(['condition','seed']):
  x=g.set_index(Q)[['query_time','y_true_log','support_end_time']].sort_index()
  if ref is None:ref=x
  else:pd.testing.assert_frame_equal(ref,x)
  checks.append(dict(condition=mode,seed=int(seed),queries=len(g),same_queries_truth_times=True))
 assert p.groupby('condition').seed.apply(set).map(lambda x:x==SEEDS).all()
 group=['condition']+Q;ensemble=p.groupby(group,sort=False).agg(query_time=('query_time','first'),support_end_time=('support_end_time','first'),y_true_log=('y_true_log','first'),y_pred_log=('y_pred_log','mean'),seed_count=('seed','nunique')).reset_index();assert ensemble.seed_count.eq(10).all()
 ensemble.to_parquet(O/'ensemble_predictions_LOCAL_ONLY.parquet',index=False)
 rows=[]
 for (mode,drug,seed),g in p.groupby(['condition','drug_id','seed']):rows.append(dict(condition=mode,drug_id=drug,seed=seed,**balanced_metrics(g.y_true_log,g.y_pred_log,g.source_id.astype(str)+'::'+g.participant_id.astype(str))))
 seed_metrics=pd.DataFrame(rows);seed_metrics.to_csv(O/'individual_seed_drug_metrics.csv',index=False)
 rows=[]
 for (mode,drug),g in ensemble.groupby(['condition','drug_id']):rows.append(dict(condition=mode,drug_id=drug,**balanced_metrics(g.y_true_log,g.y_pred_log,g.source_id.astype(str)+'::'+g.participant_id.astype(str))))
 em=pd.DataFrame(rows);em.to_csv(O/'ensemble_drug_metrics.csv',index=False)
 avg=seed_metrics.groupby(['condition','drug_id']).agg(mean_seed_log_rmse=('log_rmse','mean'),sd_seed_log_rmse=('log_rmse','std'),mean_seed_log_mae=('log_mae','mean'),mean_seed_twofold=('within_twofold','mean')).reset_index();comp=em.merge(avg,on=['condition','drug_id'],validate='one_to_one');comp['ensemble_minus_mean_seed_rmse']=comp.log_rmse-comp.mean_seed_log_rmse;comp.to_csv(O/'ensemble_vs_individual.csv',index=False)
 old=pd.read_csv(I/'single_drug_summary.csv');old=old[old.k==2].set_index(['condition','drug_id']);new=avg.set_index(['condition','drug_id']);parity=float(abs(old.rmse_mean-new.mean_seed_log_rmse).max());assert parity<1e-12
 chosen=pd.read_csv(I/'single_curve_selection.csv');assert len(chosen)==11;chosen.to_csv(O/'case_selection_LOCAL_ONLY.csv',index=False);su=pd.read_parquet(C/'curve_support.parquet');dose=pd.read_csv(C/'curve_doses.csv');metrics=[];export=[];supports=[];doses=[]
 plt.rcParams.update({'font.size':10,'axes.titlesize':11,'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9})
 def row(axs,r):
  key=r[KEYS];g=p[(p[KEYS]==key).all(axis=1)];e=ensemble[(ensemble[KEYS]==key).all(axis=1)];s=su[(su.track=='Known-drug')&(su[KEYS]==key).all(axis=1)].groupby('time').target_log_dv.mean().sort_index();dv=dose[(dose.track=='Known-drug')&(dose[KEYS]==key).all(axis=1)];assert len(s)==2 and g.drug_id.nunique()==1
  truth=e[e.condition=='single_drug'].sort_values(['query_time','observation_id']);allc=np.exp(np.r_[g.y_pred_log.values,truth.y_true_log.values,s.values]);ylim=(allc.min()/1.5,allc.max()*1.5);xmin=min(s.index.min(),dv.time.min() if len(dv) else s.index.min());xmax=truth.query_time.max();pad=(xmax-xmin)*.04
  for ax,mode,name,color in zip(axs,MODS,NAMES,COLORS):
   h=g[g.condition==mode];m=e[e.condition==mode].sort_values(['query_time','observation_id']);assert set(h.seed)==SEEDS
   for seed,hh in h.groupby('seed'):
    hh=hh.sort_values(['query_time','observation_id']);ax.plot(hh.query_time,np.exp(hh.y_pred_log),color=color,alpha=.25,lw=.7,marker='.',ms=2,zorder=2)
   ax.plot(m.query_time,np.exp(m.y_pred_log),color=color,lw=2,marker='.',ms=3,zorder=4)
   ax.scatter(s.index,np.exp(s),c='black',marker='s',s=22,zorder=7);ax.scatter(truth.query_time,np.exp(truth.y_true_log),facecolor='white',edgecolor='black',s=20,linewidth=.7,zorder=6);ax.axvline(s.index.max(),color='black',ls='--',lw=.7)
   for event in dv.itertuples():
    ax.axvline(event.time,color='gray',lw=.5,alpha=.6)
    if event.duration>0:ax.axvspan(event.time,event.time+event.duration,color='gray',alpha=.12)
   ax.set_yscale('log');ax.set_ylim(ylim);ax.set_xlim(xmin-pad,xmax+pad);ax.grid(alpha=.12);ax.set_xlabel('Time (h)');err=np.sqrt(np.mean((m.y_pred_log-m.y_true_log)**2));ax.text(.04,.04,f'Ensemble log-RMSE\n{err:.3f}',transform=ax.transAxes,fontsize=8,bbox=dict(facecolor='white',edgecolor='none',alpha=.75))
   metrics.append(dict(drug_id=r.drug_id,condition=mode,ensemble_profile_log_rmse=err,query_records=len(m),distinct_future_times=m.query_time.nunique(),seeds=10))
  axs[0].set_ylabel(r.drug_id.capitalize()+'\nConcentration (µmol/L)');export.append(g);supports.append(s.reset_index().assign(drug_id=r.drug_id));doses.append(dv)
 legend=[Line2D([0],[0],color='gray',alpha=.5,lw=.8,label='Individual seed (1–10)'),Line2D([0],[0],color='black',lw=2,label='Geometric ensemble'),Line2D([0],[0],color='black',marker='s',ls='',label='Observed history'),Line2D([0],[0],color='black',mfc='white',marker='o',ls='',label='Future measurements')]
 fig,axs=plt.subplots(4,5,figsize=(12.8,10.3),layout='constrained')
 for ax,name in zip(axs[0],NAMES):ax.set_title(name,fontweight='bold')
 for axes,drug in zip(axs,['buprenorphine','empagliflozin','mavoglurant','remifentanil']):row(axes,chosen[chosen.drug_id==drug].iloc[0])
 fig.legend(handles=legend,loc='outside upper center',ncol=4,fontsize=9);fig.savefig(O/'Ten_seed_curves.pdf');fig.savefig(O/'Ten_seed_curves.png',dpi=180);plt.close(fig)
 # Complete atlas, same fixed samples, each drug readable on its own page.
 with PdfPages(O/'Ten_seed_curve_atlas.pdf') as pdf:
  for _,r in chosen.iterrows():
   fig,axs=plt.subplots(2,3,figsize=(11,7),layout='constrained');flat=axs.ravel()
   for ax,name in zip(flat,NAMES):ax.set_title(name,fontweight='bold')
   row(flat[:5],r);flat[-1].axis('off');flat[-1].legend(handles=legend,loc='center',frameon=False);fig.suptitle(f'{r.drug_id.capitalize()} | same profile, k=2, ten seeds\nLines connect stored query predictions; no interpolated solver trajectory',fontsize=12);pdf.savefig(fig);plt.close(fig)
 pd.DataFrame(metrics).drop_duplicates().to_csv(O/'case_ensemble_metrics.csv',index=False);pd.concat(export).drop_duplicates(['condition','seed']+Q).to_parquet(O/'curve_predictions_LOCAL_ONLY.parquet',index=False);pd.concat(supports).drop_duplicates().to_csv(O/'curve_support_LOCAL_ONLY.csv',index=False);pd.concat(doses).drop_duplicates().to_csv(O/'curve_doses_LOCAL_ONLY.csv',index=False);pd.DataFrame(checks).to_csv(O/'query_checks.csv',index=False)
 with pd.ExcelWriter(O/'ten_seed_curve_metrics.xlsx') as w:
  em.to_excel(w,sheet_name='ensemble_drug',index=False);seed_metrics.to_excel(w,sheet_name='individual_seed',index=False);comp.to_excel(w,sheet_name='ensemble_vs_seed_mean',index=False);pd.DataFrame(metrics).drop_duplicates().to_excel(w,sheet_name='case_metrics',index=False)
 manifest=dict(status='completed',new_training=False,seeds=sorted(SEEDS),condition='Single-drug Known; initial k=2',model_names=dict(zip(MODS,NAMES)),case_selection='Unchanged previous seed-1 median-error rule; no reselection for ensemble performance',aggregation='Mean natural-log prediction per identical query then exponentiate; geometric ensemble',uncertainty='Individual-seed spread only; not a confidence or prediction interval',main_table_parity_max_abs=parity,query_contract_checks=len(checks),ensemble_query_rows=len(ensemble),input_hashes={str(x.relative_to(R)):sha(x) for x in [I/'single_drug_predictions.parquet',I/'single_curve_selection.csv',C/'curve_support.parquet',C/'curve_doses.csv']},code_sha256=sha(Path(__file__)))
 (O/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2),flush=True)
if __name__=='__main__':main()
