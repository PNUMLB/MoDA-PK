"""Distance-preservation and drug-wise predictive diagnostics; frozen models and fixed data."""
from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from mhap_pk.evaluation.drug_metrics import balanced_metrics
from scipy.spatial.distance import pdist,squareform
from scipy.stats import spearmanr
from sklearn.manifold import MDS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
O=R/'results/summaries/drug_geometry_20261002';G=R/'results/summaries/profile_comparison_20261002'
MODS=['molecular_relation','poppk_vi_map','dcm','unipk','deepct'];N=dict(zip(MODS,['MoDE-PK (Ours)','PopPK + MAP','DCM','Uni-PK','DeepCt']));C=dict(zip(MODS,['#b2182b','#2166ac','#8856a7','#1b7837','#d95f02']))
DN=lambda s:'R-lipoic acid' if s=='r_lipoic_acid' else s.replace('_',' ').capitalize()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def save(fig,name):
 fig.savefig(O/(name+'.pdf'),bbox_inches='tight');fig.savefig(O/(name+'.png'),dpi=180,bbox_inches='tight');plt.close(fig)
def metrics(d,keys):
 rows=[]
 for vals,g in d.groupby(keys,observed=True,sort=True):
  vals=vals if isinstance(vals,tuple) else (vals,)
  people=g.source_id.astype(str)+'::'+g.participant_id.astype(str)
  rows.append(dict(zip(keys,vals))|balanced_metrics(g.y_true_log,g.y_pred_log,people))
 return pd.DataFrame(rows)
def summary(d,keys):
 names=['log_rmse','log_mae','within_twofold','signed_log_bias','linear_rmse','linear_mae','r2_log','r2_linear']
 s=d.groupby(keys)[names].agg(['mean','std','count']);s.columns=['_'.join(t) for t in s.columns];return s.reset_index()
def main():
 O.mkdir(exist_ok=False,parents=True)
 a=pd.read_parquet(R/'results/summaries/training_scope_extension_20261001/single_drug_predictions.parquet');a['scope']='Single-drug Known';a.condition=a.condition.replace({'single_drug':'molecular_relation'})
 b=pd.read_parquet(R/'results/summaries/ten_seeds_20261001/canonical_predictions.parquet');b=b[(b.track=='Joint drug–source')&b.condition.isin(MODS)].copy();b['scope']='Joint drug–source'
 p=pd.concat([a,b],ignore_index=True);p=p[(p.k==2)&(p.protocol=='first_k')&p.condition.isin(MODS)].copy()
 assert not p.duplicated(['scope','condition','seed','source_id','drug_id','participant_id','profile_id','observation_id']).any()
 # Every model has identical query/label keys per evaluation scope and seed.
 for (scope,seed),g in p.groupby(['scope','seed']):
  refs=[]
  for mod,h in g.groupby('condition'):
   refs.append(h.set_index(['source_id','drug_id','participant_id','profile_id','observation_id']).y_true_log.sort_index())
  assert all(refs[0].equals(t) for t in refs[1:])
 by=metrics(p,['scope','drug_id','condition','seed']);su=summary(by,['scope','drug_id','condition']);by.to_csv(O/'drug_metrics_by_seed.csv',index=False);su.to_csv(O/'drug_metrics_summary.csv',index=False)
 by[['scope','drug_id','condition','seed','participants','queries']].to_csv(O/'counts.csv',index=False)
 p['horizon_h']=p.query_time-p.support_end_time;assert (p.horizon_h>0).all();p['horizon_bin']=pd.cut(p.horizon_h,[0,1,4,12,24,np.inf],labels=['0–1','1–4','4–12','12–24','>24'])
 h=metrics(p,['scope','drug_id','condition','seed','horizon_bin']);hs=summary(h,['scope','drug_id','condition','horizon_bin']);h.to_csv(O/'horizon_by_seed.csv',index=False);hs.to_csv(O/'horizon_summary.csv',index=False)
 with pd.ExcelWriter(O/'drug_metrics.xlsx') as w:
  by.to_excel(w,sheet_name='seed_metrics',index=False);su.to_excel(w,sheet_name='drug_summary',index=False);h.to_excel(w,sheet_name='horizon_counts_metrics',index=False)
 drugs=sorted(p.drug_id.unique());scopes=['Single-drug Known','Joint drug–source']
 fig,axs=plt.subplots(2,2,figsize=(11,10))
 for i,scope in enumerate(scopes):
  mods=[m for m in MODS if m in su[su.scope==scope].condition.unique()]
  for j,(metric,title,cmap) in enumerate([('log_mae_mean','Participant-balanced log-MAE ↓','YlOrRd'),('within_twofold_mean','Within-twofold (%) ↑','YlGn')]):
   ax=axs[i,j];tab=su[su.scope==scope].pivot(index='drug_id',columns='condition',values=metric).reindex(index=drugs,columns=mods);im=ax.imshow(tab,cmap=cmap,aspect='auto',vmin=0,vmax=100 if j else None)
   for r in range(len(drugs)):
    for c in range(len(mods)):ax.text(c,r,f'{tab.iloc[r,c]:.1f}' if j else f'{tab.iloc[r,c]:.2f}',ha='center',va='center',fontsize=8,color='black')
   ax.set_xticks(range(len(mods)),[N[m] for m in mods],rotation=25,ha='right',fontsize=8);ax.set_yticks(range(len(drugs)),[DN(x) for x in drugs],fontsize=8);ax.set_title(scope+'\n'+title,fontsize=11);fig.colorbar(im,ax=ax,shrink=.6)
 fig.tight_layout();save(fig,'Drug_complementary_metrics')
 # Drug-wise future-label diagnostics, no seed ensembling.
 with PdfPages(O/'Drug_prediction_atlas.pdf') as pdf:
  for scope in scopes:
   for drug in drugs:
    g=p[(p.scope==scope)&(p.drug_id==drug)&(p.seed==1)];fig,axs=plt.subplots(2,3,figsize=(11,7))
    for ax,mod in zip(axs.flat,MODS):
     q=g[g.condition==mod]
     if q.empty:ax.axis('off');continue
     col=np.log1p(q.horizon_h);im=ax.scatter(q.y_true_log,q.y_pred_log,c=col,s=13,cmap='viridis');lo=min(q.y_true_log.min(),q.y_pred_log.min());hi=max(q.y_true_log.max(),q.y_pred_log.max());ax.plot([lo,hi],[lo,hi],'k--',lw=.7);ax.set_xlabel('Observed log concentration');ax.set_ylabel('Predicted log concentration');ax.set_title(N[mod],fontsize=11);fig.colorbar(im,ax=ax,label='log(1 + horizon / h)',shrink=.7)
    axs.flat[-1].axis('off');fig.suptitle(f'{DN(drug)} | {scope} | seed 1 | k=2');fig.tight_layout(rect=(0,0,1,.95));pdf.savefig(fig);plt.close(fig)
 with PdfPages(O/'Drug_horizon_atlas.pdf') as pdf:
  for drug in drugs:
   fig,axs=plt.subplots(1,2,figsize=(10,4))
   for ax,scope in zip(axs,scopes):
    for mod in MODS:
     t=hs[(hs.drug_id==drug)&(hs.scope==scope)&(hs.condition==mod)].set_index('horizon_bin').reindex(['0–1','1–4','4–12','12–24','>24'])
     if t.log_rmse_mean.notna().any():ax.errorbar(range(5),t.log_rmse_mean,yerr=t.log_rmse_std,color=C[mod],label=N[mod],marker='o',capsize=2)
    ax.set_xticks(range(5),['0–1','1–4','4–12','12–24','>24']);ax.set_xlabel('Horizon after final observation (h)');ax.set_ylabel('Participant-balanced log-RMSE');ax.set_title(scope)
   axs[0].legend(fontsize=7);fig.suptitle(DN(drug));fig.tight_layout();pdf.savefig(fig);plt.close(fig)
 # Metric MDS on exactly the previously extracted model-specific vectors.
 coords=pd.read_csv(G/'coordinates.csv');metadata=coords[coords.projection=='t-SNE'];models=list(metadata.model.unique());palette=dict(zip(drugs,plt.get_cmap('tab20')(np.linspace(0,1,len(drugs)))));xyall=[];diagnostics=[];distrecords=[]
 fig,axs=plt.subplots(2,3,figsize=(11,7));sf,saxs=plt.subplots(2,3,figsize=(11,7));handles=[]
 for ax,sax,model in zip(axs.flat,saxs.flat,models):
  x=np.load(G/f'{model}_features.npz')['test_scaled'];md=metadata[metadata.model==model].reset_index(drop=True);dist=pdist(x);dm=squareform(dist)
  mds=MDS(n_components=2,metric=True,n_init=8,max_iter=1000,eps=1e-9,random_state=17,dissimilarity='precomputed',n_jobs=1);xy=mds.fit_transform(dm);low=pdist(xy);stress=np.sqrt(np.sum((dist-low)**2)/np.sum(dist**2));rho=float(spearmanr(dist,low).statistic)
  diagnostics.append(dict(model=model,profiles=len(x),dimension=x.shape[1],normalized_distance_stress=stress,distance_spearman=rho))
  md['mds_x']=xy[:,0];md['mds_y']=xy[:,1];xyall.append(md)
  ii,jj=np.triu_indices(len(x),1);distrecords.append(pd.DataFrame(dict(model=model,profile_a=md.profile_uid.values[ii],profile_b=md.profile_uid.values[jj],original_distance=dist,projected_distance=low)))
  for drug in drugs:
   mask=md.drug_id==drug;handle=ax.scatter(xy[mask,0],xy[mask,1],c=[palette[drug]],s=25,edgecolors='white',linewidths=.3,label=DN(drug))
   if model==models[0]:handles.append(handle)
  ax.set_title(N.get(model,'PopPK + MAP')+f'\nstress={stress:.3f}',fontsize=11);ax.set_aspect('equal',adjustable='datalim');ax.set_xlabel('MDS 1');ax.set_ylabel('MDS 2')
  sax.scatter(dist,low,s=3,alpha=.15);mx=max(max(dist),max(low));sax.plot([0,mx],[0,mx],'k--',lw=.7);sax.set_title(N.get(model,'PopPK + MAP')+f' | ρ={rho:.3f}');sax.set_xlabel('Original standardized-feature distance');sax.set_ylabel('2D distance')
 axs.flat[-1].axis('off');axs.flat[-1].legend(handles=handles,loc='center',fontsize=10,frameon=False);saxs.flat[-1].axis('off');fig.suptitle('Metric MDS | same 101 test profiles | k=2');fig.tight_layout(rect=(0,0,1,.96));save(fig,'Profile_MDS');sf.tight_layout();save(sf,'MDS_distance_preservation')
 pd.concat(xyall).to_csv(O/'mds_coordinates.csv',index=False);pd.DataFrame(diagnostics).to_csv(O/'mds_diagnostics.csv',index=False);pd.concat(distrecords).to_parquet(O/'original_and_projected_distances.parquet',index=False)
 # Original-space within-drug distances vs known history; future error is descriptive only.
 import torch
 from mhap_pk.data.traceable import load_profiles,indices_for_split
 from mhap_pk.data import representations,make_batch
 from mhap_pk.evaluation.history_protocols import initial_observation_inputs
 ps,meta,_=load_profiles();tr,_,te=indices_for_split(meta,'known');z,ids,_=representations(meta,tr,'unimol',17);batch=make_batch(ps,meta,te,z,ids);args=initial_observation_inputs(batch,2);rr=[]
 for j,idx in enumerate(batch.indices):
  prof=ps[idx];mask=args[4][j];tt=args[2][j,mask];yy=args[3][j,mask];uid=json.dumps([str(prof[k]) for k in ['source_id','drug_id','participant_id','profile_id']],ensure_ascii=False)
  rr.append(dict(profile_uid=uid,last_observed_log=float(yy[-1]),last_observed_time=float(tt[-1]),support_span=float(tt[-1]-tt[0])))
 context=pd.DataFrame(rr);context.to_csv(O/'observed_history_context.csv',index=False);cor=[]
 for model in models:
  md=metadata[metadata.model==model].reset_index(drop=True).merge(context,on='profile_uid',validate='one_to_one',sort=False);x=np.load(G/f'{model}_features.npz')['test_scaled']
  for drug,g in md.groupby('drug_id'):
   ix=g.index.to_numpy();d=pdist(x[ix])
   for feature in ['last_observed_log','last_observed_time','support_span']:
    target=pdist(g[[feature]].values);reason='';rho=np.nan
    if len(ix)<5:reason='fewer than 5 profiles'
    elif np.ptp(target)==0 or np.ptp(d)==0:reason='constant pairwise differences'
    else:rho=float(spearmanr(d,target).statistic)
    cor.append(dict(model=model,drug_id=drug,profiles=len(ix),pairs=len(d),feature=feature,distance_rank_correlation=rho,undefined_reason=reason))
 cd=pd.DataFrame(cor);cd.to_csv(O/'within_drug_distance_associations.csv',index=False)
 fig,axs=plt.subplots(1,3,figsize=(13,6))
 for ax,feature,title in zip(axs,['last_observed_log','last_observed_time','support_span'],['Observed log concentration','Final observation time','Observation span']):
  tab=cd[cd.feature==feature].pivot(index='drug_id',columns='model',values='distance_rank_correlation').reindex(index=drugs,columns=models);im=ax.imshow(tab,cmap='coolwarm',vmin=-1,vmax=1,aspect='auto')
  for i in range(len(drugs)):
   for j in range(len(models)):
    v=tab.iloc[i,j];ax.text(j,i,'NA' if pd.isna(v) else f'{v:.2f}',ha='center',va='center',fontsize=7)
  ax.set_xticks(range(len(models)),[N.get(m,'PopPK + MAP') for m in models],rotation=35,ha='right',fontsize=8);ax.set_yticks(range(len(drugs)),[DN(d) for d in drugs],fontsize=8);ax.set_title(title);fig.colorbar(im,ax=ax,shrink=.6)
 fig.suptitle('Within-drug original-space distance associations | descriptive, dependent pairs');fig.tight_layout(rect=(0,0,1,.94));save(fig,'Within_drug_geometry')
 # Keep original first-k scores exactly, remove fixed-origin from main figure only.
 scores=pd.read_csv(R/'results/summaries/ten_seeds_20261001/cohort_summary.csv');fig,axs=plt.subplots(1,2,figsize=(10,4))
 for ax,track in zip(axs,['Known-drug','Joint drug–source']):
  for mod in MODS:
   g=scores[(scores.track==track)&(scores.condition==mod)].sort_values('k')
   if len(g):ax.errorbar(g.k,g.sb_hrmse_mean,yerr=g.sb_hrmse_std,marker='o',color=C[mod],label=N[mod],capsize=2)
  ax.set_title(track);ax.set_xticks(range(5));ax.set_xlabel('Initial observed concentration times k');ax.set_ylabel('SB-hRMSE');ax.grid(alpha=.15)
 axs[0].legend(fontsize=8);fig.tight_layout();save(fig,'Initial_history')
 # Compact supplement tables, full precision/SD/counts in workbook.
 for scope,fn in zip(scopes,['known_additional','joint_additional']):
  text=r'\begingroup\scriptsize\setlength{\tabcolsep}{3pt}'+'\n'+r'\begin{longtable}{llrrrrrr}'+'\n'+r'\caption{'+scope+r': participant-balanced additional metrics at $k=2$, averaged over ten seeds. RMSE is in $\mu$mol/L; $R^2$ is computed across queries with equal participant weights. NA denotes undefined $R^2$.}\\'+'\n'+r'\toprule Drug & Model & log-MAE & Twofold (\%) & RMSE & $R^2_C$ & $R^2_{\log C}$ & Bias\\\midrule\endfirsthead'+'\n'+r'\toprule Drug & Model & log-MAE & Twofold (\%) & RMSE & $R^2_C$ & $R^2_{\log C}$ & Bias\\\midrule\endhead'+'\n'
  for drug in drugs:
   for mod in MODS:
    v=su[(su.scope==scope)&(su.drug_id==drug)&(su.condition==mod)]
    if v.empty:continue
    vals=[v.iloc[0][k+'_mean'] for k in ['log_mae','within_twofold','linear_rmse','r2_linear','r2_log','signed_log_bias']];fmt=lambda x:'NA' if pd.isna(x) else (f'{x:.2e}' if abs(x)>999 or (0<abs(x)<.01) else f'{x:.3f}')
    text+=' & '.join([DN(drug),N[mod]]+[fmt(x) for x in vals])+r'\\'+'\n'
  text+=r'\bottomrule\end{longtable}\endgroup'+'\n';(O/(fn+'.tex')).write_text(text)
 manifest=dict(status='completed',new_training=False,metric='participant-balanced-drug-v1',seed_set=list(range(1,11)),primary_k=2,geometry_scope='Pooled Known seed1 only; model-specific features',distance='Euclidean in training-standardized features; no physiological interpretation',mds=dict(metric=True,n_init=8,max_iter=1000,eps=1e-9,seed=17),limitations=['Pairwise associations are descriptive; pairs not independent','Between-model coordinate systems not comparable','Observed concentration encoding does not prove future forecasting ability','No arbitrary clustering labels or chosen number of clusters'],inputs={str(f):sha(f) for f in [G/'profile_features.parquet',R/'results/summaries/ten_seeds_20261001/canonical_predictions.parquet',R/'results/summaries/training_scope_extension_20261001/single_drug_predictions.parquet']},source_sha256=sha(Path(__file__)),files={f.name:sha(f) for f in O.iterdir() if f.is_file()})
 (O/'manifest.json').write_text(json.dumps(manifest,indent=2));print(pd.DataFrame(diagnostics).to_string(index=False));print('DONE',flush=True)
if __name__=='__main__':main()
