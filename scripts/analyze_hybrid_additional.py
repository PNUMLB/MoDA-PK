"""Drug-wise transfer, negative history effects, tail errors and prediction atlas."""
from pathlib import Path
import json,sys
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'));
from mhap_pk.evaluation.quality_metrics import extended_metrics
S=R/'results/summaries/hybrid_primary_20261002';assert json.loads((S/'analysis_manifest.json').read_text())['status']=='completed'
mods=['ours','poppk_vi_map','dcm','unipk','deepct'];names=dict(zip(mods,['MoDE-PK (Ours)','PopPK + MAP','DCM','Uni-PK','DeepCt']));colors=dict(zip(mods,['#b2182b','#2166ac','#8856a7','#1b7837','#d95f02']));label=lambda d:'R-lipoic acid' if d=='r_lipoic_acid' else d.capitalize()
low=pd.read_csv(S/'pooling_drug_summary.csv');low=low[low.k==2];drugs=sorted(low.drug_id.unique());fig,axs=plt.subplots(4,3,figsize=(11,11))
for ax,drug in zip(axs.flat,drugs):
 for scope,col in [('target','#2166ac'),('pooled','#b2182b')]:
  g=low[(low.drug_id==drug)&(low.scope==scope)].sort_values('cap');ax.errorbar(g.cap,g.log_rmse_mean,yerr=g.log_rmse_std,label='Target only' if scope=='target' else 'With other-drug records',marker='o',capsize=2,color=col,ms=3)
 ax.set(title=label(drug),xlabel='Target training participants',ylabel='Drug log-RMSE',xscale='log');ax.set_xticks(sorted(low[low.drug_id==drug].cap.unique()),[str(int(x)) for x in sorted(low[low.drug_id==drug].cap.unique())]);ax.grid(alpha=.15)
axs.flat[-1].axis('off');handles,labs=axs.flat[0].get_legend_handles_labels();axs.flat[-1].legend(handles,labs,loc='center',frameon=False);fig.suptitle('Matched target data; same concatenation model; k=2; ten-seed mean ± SD',fontsize=11);fig.tight_layout(rect=(0,0,1,.96));fig.savefig(S/'Pooling_by_drug.pdf');plt.close(fig)
# All drugs have real query points, no artificial trajectory smoothing.
p=pd.read_parquet(S/'primary_predictions.parquet',filters=[('protocol','==','first_k'),('k','==',2)]);p['residual']=p.y_pred_log-p.y_true_log
keys=['source_id','drug_id','participant_id','profile_id'];rows=[]
for (track,model,seed,drug),g in p.groupby(['track','condition','seed','drug_id']):
 rows.append(dict(track=track,model=model,seed=seed,drug_id=drug,query_count=len(g),p90_absolute_log_error=float(abs(g.residual).quantile(.9)),p95_absolute_log_error=float(abs(g.residual).quantile(.95)),query_median_signed_log_error=float(g.residual.median())))
pd.DataFrame(rows).to_csv(S/'drug_tail_errors_by_seed.csv',index=False)
# Participant-equal aggregation of same-checkpoint history effect, profile separate.
h=pd.read_csv(S/'history_effect_profiles.csv');print('History columns',h.columns.tolist(),flush=True)
with PdfPages(S/'Drug_prediction_atlas.pdf') as pdf:
 for track in ['Known-drug','Joint drug–source']:
  for drug in drugs:
   g=p[(p.track==track)&(p.drug_id==drug)];fig,axs=plt.subplots(1,len([m for m in mods if m in set(g.condition)]),figsize=(12,3.5),squeeze=False);lo=min(g.y_true_log.min(),g.y_pred_log.quantile(.01));hi=max(g.y_true_log.max(),g.y_pred_log.quantile(.99))
   for ax,m in zip(axs.flat,[m for m in mods if m in set(g.condition)]):
    q=g[g.condition==m];z=q.groupby(keys+['observation_id']).agg(y=('y_true_log','first'),p=('y_pred_log','mean'));ax.scatter(z.y,z.p,s=10,alpha=.55,color=colors[m]);ax.plot([lo,hi],[lo,hi],'k--',lw=.8);ax.set(title=names[m],xlabel='Observed log concentration',ylabel='Mean log prediction');ax.grid(alpha=.15)
   fig.suptitle(label(drug)+' | '+track+' | ten-seed geometric ensemble; all future queries',fontsize=11);fig.tight_layout(rect=(0,0,1,.92));pdf.savefig(fig);plt.close(fig)
# Registered model change compared at the same retained query rows and seed, not just summary joins.
old=pd.read_parquet(R/'results/summaries/ten_seeds_20261001/canonical_predictions.parquet',filters=[('condition','==','molecular_relation'),('protocol','==','first_k'),('k','==',2)])
new=p[p.condition=='ours'];ik=['track','seed']+keys+['observation_id'];a=new.set_index(ik);b=old.set_index(ik);pd.testing.assert_frame_equal(a[['query_time','y_true_log']].sort_index(),b[['query_time','y_true_log']].sort_index());j=a[['y_pred_log','y_true_log']].join(b.y_pred_log.rename('old_prediction'),validate='one_to_one').reset_index();j['delta_squared_error']=(j.y_pred_log-j.y_true_log)**2-(j.old_prediction-j.y_true_log)**2
revision=[]
for label_,df_ in [('Earlier molecular + relations',old),('Current molecular + identity',new)]:
 for (track,seed),g in df_.groupby(['track','seed']):revision.append(dict(configuration=label_,track=track,seed=seed,**extended_metrics(g)))
pd.DataFrame(revision).to_csv(S/'primary_revision_by_seed.csv',index=False)
part=j.groupby(['track','seed','source_id','drug_id','participant_id']).delta_squared_error.mean().reset_index();summary=part.groupby(['track','seed','source_id','drug_id']).agg(mean_delta_mse=('delta_squared_error','mean'),worsened_participant_fraction=('delta_squared_error',lambda x:float((x>1e-10).mean())),participants=('participant_id','nunique')).reset_index();summary.to_csv(S/'primary_change_paired_drug.csv',index=False)
# Actual history effect: same checkpoint, same query, participant-balanced and profile-level rates separated.
z0=pd.read_parquet(S/'primary_predictions.parquet',filters=[('protocol','==','first_k'),('k','==',0),('condition','==','ours')]).set_index(ik)
z2=new.set_index(ik);pd.testing.assert_frame_equal(z0[['query_time','y_true_log']].sort_index(),z2[['query_time','y_true_log']].sort_index())
hh=z2[['y_true_log','y_pred_log']].join(z0.y_pred_log.rename('prior_prediction'),validate='one_to_one').reset_index();hh['delta_mse']=(hh.y_pred_log-hh.y_true_log)**2-(hh.prior_prediction-hh.y_true_log)**2
part=hh.groupby(['track','seed','source_id','drug_id','participant_id']).delta_mse.mean().reset_index();pp=hh.groupby(['track','seed']+keys).delta_mse.mean().reset_index()
summary=part.groupby(['track','seed','source_id','drug_id']).agg(participant_mean_delta_mse=('delta_mse','mean'),worsened_participant_fraction=('delta_mse',lambda x:float((x>1e-10).mean())),participants=('participant_id','nunique')).reset_index()
profiles=pp.groupby(['track','seed','source_id','drug_id']).agg(worsened_profile_fraction=('delta_mse',lambda x:float((x>1e-10).mean())),profile_mean_delta_mse=('delta_mse','mean'),profiles=('profile_id','size')).reset_index();summary=summary.merge(profiles,on=['track','seed','source_id','drug_id'],validate='one_to_one');summary.to_csv(S/'history_effect_drug_by_seed.csv',index=False)
fig,axs=plt.subplots(1,2,figsize=(11,5))
for ax,track in zip(axs,['Known-drug','Joint drug–source']):
 g=summary[summary.track==track].groupby('drug_id').participant_mean_delta_mse.agg(['mean','std']).reindex(drugs);ax.barh(range(len(drugs)),g['mean'],xerr=g['std'],color=np.where(g['mean']<=0,'#2166ac','#b2182b'),alpha=.8);ax.axvline(0,color='black',lw=.8);ax.set_yticks(range(len(drugs)),[label(d) for d in drugs]);ax.set(title=track,xlabel='MSE(k=2) − MSE(k=0)');ax.grid(axis='x',alpha=.15)
fig.suptitle('Same-checkpoint observation effect; participant-balanced within each drug',fontsize=11);fig.tight_layout(rect=(0,0,1,.95));fig.savefig(S/'History_effect_by_drug.pdf');plt.close(fig)
with pd.ExcelWriter(S/'MoDE_PK_hybrid_results.xlsx',engine='openpyxl',mode='a',if_sheet_exists='replace') as w:
 for name in ['primary_revision_by_seed','history_effect_drug_by_seed','primary_change_paired_drug','drug_tail_errors_by_seed','budget_cohort_summary','budget_drug_by_seed','primary_cohort_by_seed','primary_drug_by_seed','representation_original_space','stage_drug_metrics','primary_source_bootstrap']:
  pd.read_csv(S/(name+'.csv')).to_excel(w,sheet_name=name[:31],index=False)
(S/'additional_manifest.json').write_text(json.dumps(dict(status='completed',model_change_query_rows=len(j),old='Uni-Mol + both relation terms',new='Uni-Mol + identity, relations OFF',old_vs_new_changes_multiple_factors=True,model_change_after_internal_inspection=True),indent=2));print('Additional hybrid analyses complete')
