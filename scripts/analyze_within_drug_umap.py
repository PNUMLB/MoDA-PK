"""Within-drug UMAP of matched retrained input controls; no model selection."""
from pathlib import Path
import sys,json,hashlib,itertools
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import trustworthiness
from scipy.spatial.distance import pdist,cdist
from scipy.stats import spearmanr
import umap
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'scripts'))
from analyze_molecular_conditioning import parent
I=R/'results/summaries/molecular_conditioning_20261002';O=R/'results/summaries/within_drug_umap_20261002'
MODES=['no_molecule','molecular'];LABELS={'no_molecule':'Without molecular features','molecular':'With molecular features'}
KEYS=['source_id','drug_id','participant_id','profile_id']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def metric(x,y):
 a=pdist(x);b=pdist(y);return float(spearmanr(a,b).statistic) if np.ptp(a)>1e-12 and np.ptp(b)>1e-12 else None
def project(x,nn=5,seed=17,md=.1):
 return umap.UMAP(n_neighbors=min(nn,len(x)-1),min_dist=md,n_components=2,random_state=seed,n_jobs=1,init='random',metric='euclidean',n_epochs=500).fit_transform(x)
def neighbors(x,k=3):
 d=cdist(x,x);np.fill_diagonal(d,np.inf);return np.argsort(d,axis=1)[:,:k]
def main():
 O.mkdir(parents=True,exist_ok=False)
 # Lock the descriptive rule before calculating projections or examining colors.
 plan=dict(models=MODES,relation_losses='OFF in both retrained conditions',k=2,main_drugs='All drugs with at least 10 test profiles; rule uses count only',main_fit_seed=1,projection=dict(n_neighbors=5,min_dist=.1,seed=17,n_epochs=500,init='random'),sensitivity=dict(n_neighbors=[3,5,10],min_dist=[.1,.5],seeds=[17,29,43]),scope='Known-drug only; pooled training; descriptive internal analysis',small_groups='Do not project groups with fewer than 10 profiles; provide actual/predicted concentration diagnostics instead',future_labels_used_for_projection=False)
 (O/'analysis_plan.json').write_text(json.dumps(plan,indent=2))
 f=pd.read_parquet(I/'features_LOCAL_ONLY.parquet');f=f[f.model.isin(MODES)].copy();context=pd.read_csv(R/'results/summaries/drug_geometry_20261002/observed_history_context.csv').set_index('profile_uid');assert context.index.is_unique
 md=f[(f.model=='molecular')&(f.seed==1)&(f.role=='test')][['profile_uid','drug_id']].copy();counts=md.groupby('drug_id').size();drugs=sorted(counts[counts>=10].index);prediction=[];pstats=[];check=[]
 for mode in MODES:
  p=parent('known',1,mode)/'predictions.parquet';d=pd.read_parquet(p);d=d[(d.protocol=='first_k')&(d.k==2)].copy();assert not d.duplicated(KEYS+['observation_id']).any()
  d['profile_uid']=d.apply(lambda r:json.dumps([str(r[k]) for k in KEYS]),axis=1);d['horizon']=d.query_time-d.support_end_time;assert (d.horizon>0).all();d['se']=(d.y_pred_lnc-d.y_true_lnc)**2;d['model']=mode;prediction.append(d)
  for uid,g in d.groupby('profile_uid'):pstats.append(dict(model=mode,profile_uid=uid,future_log_rmse=np.sqrt(g.se.mean()),mean_horizon=g.horizon.mean(),query_count=len(g),participant_id=g.participant_id.iloc[0],source_id=g.source_id.iloc[0]))
  check.append(dict(model=mode,path=str(p.relative_to(R)),sha256=sha(p)))
 pred=pd.concat(prediction,ignore_index=True);pstats=pd.DataFrame(pstats);a=prediction[0].set_index(KEYS+['observation_id'])[['query_time','y_true_lnc']].sort_index();b=prediction[1].set_index(KEYS+['observation_id'])[['query_time','y_true_lnc']].sort_index();assert a.equals(b)
 ftest=f[(f.seed==1)&(f.role=='test')].merge(context,on='profile_uid',validate='many_to_one').merge(pstats,on=['model','profile_uid'],validate='one_to_one');coords=[];diagnostics=[];associations=[];senscoords=[];matrices={}
 for mode in MODES:
  for seed in range(1,11):
   train=f[(f.model==mode)&(f.seed==seed)&(f.role=='train')];test=f[(f.model==mode)&(f.seed==seed)&(f.role=='test')].merge(context,on='profile_uid',validate='one_to_one');sc=StandardScaler().fit(np.stack(train.vector.map(json.loads)));x=sc.transform(np.stack(test.vector.map(json.loads)))
   for drug in drugs:
    ix=np.flatnonzero(test.drug_id.to_numpy()==drug);xx=x[ix];sub=test.iloc[ix]
    for feature in ['last_observed_log','last_observed_time','support_span']:associations.append(dict(model=mode,fit_seed=seed,drug_id=drug,feature=feature,rho=metric(xx,sub[[feature]].to_numpy()),profiles=len(sub),scope='dependent profile pairs; descriptive, no significance test'))
    if seed!=1:continue
    matrices[(mode,drug)]=xx
    for nn,mdist,rs in itertools.product([3,5,10],[.1,.5],[17,29,43]):
     xy=project(xx,nn,rs,mdist);n1=neighbors(xx);n2=neighbors(xy);overlap=float(np.mean([len(set(a)&set(b))/3 for a,b in zip(n1,n2)]));tw=float(trustworthiness(xx,xy,n_neighbors=3))
     diagnostics.append(dict(model=mode,drug_id=drug,neighbors=min(nn,len(xx)-1),min_dist=mdist,projection_seed=rs,neighbor_overlap3=overlap,trustworthiness3=tw,profiles=len(xx)))
     for uid,(u,v) in zip(sub.profile_uid,xy):senscoords.append(dict(model=mode,drug_id=drug,profile_uid=uid,neighbors=min(nn,len(xx)-1),min_dist=mdist,projection_seed=rs,x=u,y=v))
     if (nn,mdist,rs)==(5,.1,17):
      for uid,(u,v) in zip(sub.profile_uid,xy):coords.append(dict(model=mode,drug_id=drug,profile_uid=uid,x=u,y=v))
    print('PROJECTED',mode,drug,flush=True)
 c=pd.DataFrame(coords).merge(ftest.drop(columns=['drug_id','vector','seed','role']),on=['model','profile_uid'],validate='one_to_one');c.to_csv(O/'coordinates_LOCAL_ONLY.csv',index=False);pd.DataFrame(senscoords).to_csv(O/'sensitivity_coordinates_LOCAL_ONLY.csv',index=False);pd.DataFrame(diagnostics).to_csv(O/'projection_diagnostics.csv',index=False);pd.DataFrame(associations).to_csv(O/'original_space_associations.csv',index=False)
 group=[]
 for drug,n in counts.items():
  t=ftest[(ftest.drug_id==drug)&(ftest.model=='molecular')];group.append(dict(drug_id=drug,profiles=int(n),participants=len(t[['source_id','participant_id']].drop_duplicates()),query_count=int(t.query_count.sum()),projected=n>=10,reason='at least 10 profiles' if n>=10 else 'too few profiles for this descriptive projection rule'))
 pd.DataFrame(group).to_csv(O/'drug_counts.csv',index=False)
 effect=pd.read_csv(I/'molecular_effect_summary.csv');effect=effect[effect.track=='Known-drug'].set_index('drug_id')
 fig,axs=plt.subplots(len(drugs),4,figsize=(12,3.1*len(drugs)),layout='constrained')
 for ri,drug in enumerate(drugs):
  g=c[c.drug_id==drug];l=g.last_observed_log;v=g.future_log_rmse;limc=(l.min(),l.max());lime=(0,max(v.max(),1e-6));count=next(x for x in group if x['drug_id']==drug)
  for ci,(mode,feature) in enumerate([('no_molecule','last_observed_log'),('molecular','last_observed_log'),('no_molecule','future_log_rmse'),('molecular','future_log_rmse')]):
   t=g[g.model==mode];ax=axs[ri,ci];lo,hi=limc if feature=='last_observed_log' else lime;im=ax.scatter(t.x,t.y,c=t[feature],cmap='viridis' if ci<2 else 'magma',vmin=lo,vmax=hi,s=35,edgecolors='black',linewidths=.3);ax.set_xticks([]);ax.set_yticks([])
   if ri==0:ax.set_title(LABELS[mode]+'\n'+('Observed concentration' if ci<2 else 'Future prediction error'),fontsize=9)
   if ci==0:ax.set_ylabel(f'{drug.capitalize()}\n{count["profiles"]} profiles / {count["participants"]} participants',fontsize=10)
   if ci in [1,3]:fig.colorbar(im,ax=axs[ri,ci-1:ci+1].tolist(),shrink=.72,label='Last observed log concentration' if ci==1 else 'Profile future log-RMSE')
  delta=effect.loc[drug,'delta_log_rmse'];axs[ri,2].set_xlabel(f'10-seed drug RMSE difference: {delta:+.3f}\n(molecular minus no-molecule)',fontsize=8)
 fig.suptitle('Within-drug history representations | matched molecular-input controls\nSame profile positions reused across the two colorings; relation losses OFF',fontsize=12);fig.savefig(O/'Within_drug_UMAP.pdf');fig.savefig(O/'Within_drug_UMAP.png',dpi=160);plt.close(fig)
 # Supplement: same coordinates, observed time and future horizon instead of concentration/error.
 with PdfPages(O/'Within_drug_context_atlas.pdf') as pdf:
  for drug in drugs:
   g=c[c.drug_id==drug];fig,axs=plt.subplots(2,3,figsize=(11,7),layout='constrained')
   for ci,feature in enumerate(['last_observed_time','support_span','mean_horizon']):
    lo,hi=g[feature].min(),g[feature].max()
    for ri,mode in enumerate(MODES):
     t=g[g.model==mode];ax=axs[ri,ci]
     if hi-lo<1e-12:ax.scatter(t.x,t.y,color='gray',s=30);ax.set_xlabel(f'Constant: {lo:g} h')
     else:im=ax.scatter(t.x,t.y,c=t[feature],vmin=lo,vmax=hi,cmap='viridis',s=30);fig.colorbar(im,ax=ax,label='hours')
     ax.set_title(feature.replace('_',' '));ax.set_xticks([]);ax.set_yticks([])
     if ci==0:ax.set_ylabel(LABELS[mode])
   fig.suptitle(drug.capitalize()+': same UMAP coordinates, different time annotations');pdf.savefig(fig);plt.close(fig)
 with PdfPages(O/'Small_group_prediction_atlas.pdf') as pdf:
  for drug in sorted(counts[counts<10].index):
   g=pred[pred.drug_id==drug];fig,axs=plt.subplots(1,2,figsize=(9,4),layout='constrained');lo=min(g.y_true_lnc.min(),g.y_pred_lnc.min());hi=max(g.y_true_lnc.max(),g.y_pred_lnc.max())
   for ax,mode in zip(axs,MODES):
    t=g[g.model==mode];im=ax.scatter(t.y_true_lnc,t.y_pred_lnc,c=t.horizon,cmap='viridis',vmin=g.horizon.min(),vmax=g.horizon.max(),s=18);ax.plot([lo,hi],[lo,hi],'k--',lw=.8);ax.set_xlim(lo-.1,hi+.1);ax.set_ylim(lo-.1,hi+.1);ax.set_title(LABELS[mode]);ax.set_xlabel('Observed future log concentration');ax.set_ylabel('Predicted future log concentration')
   fig.colorbar(im,ax=axs.tolist(),label='Forecast horizon (hours)');fig.suptitle(f'{drug.capitalize()} | {counts[drug]} test profiles: no UMAP\nQuery points are repeated measurements, not independent participants');pdf.savefig(fig);plt.close(fig)
 # Sensitivity matrices avoid choosing a flattering projection. Each page fixes min_dist.
 sc=pd.DataFrame(senscoords)
 with PdfPages(O/'UMAP_sensitivity_atlas.pdf') as pdf:
  for drug,mode,mdist in itertools.product(drugs,MODES,[.1,.5]):
   fig,axs=plt.subplots(3,3,figsize=(8,8),layout='constrained');g=c[(c.drug_id==drug)&(c.model==mode)].set_index('profile_uid');lo,hi=g.last_observed_log.min(),g.last_observed_log.max()
   for i,nn in enumerate([3,5,10]):
    for j,rs in enumerate([17,29,43]):
     t=sc[(sc.drug_id==drug)&(sc.model==mode)&(sc.min_dist==mdist)&(sc.neighbors==min(nn,len(g)-1))&(sc.projection_seed==rs)];ax=axs[i,j];ax.scatter(t.x,t.y,c=g.loc[t.profile_uid,'last_observed_log'],vmin=lo,vmax=hi,cmap='viridis',s=22);ax.set_title(f'neighbors={nn}, seed={rs}',fontsize=9);ax.set_xticks([]);ax.set_yticks([])
   fig.suptitle(f'{drug} | {LABELS[mode]} | min_dist={mdist}\nColor: last observed log concentration');pdf.savefig(fig);plt.close(fig)
 (O/'manifest.json').write_text(json.dumps(dict(status='completed',new_training=False,models=MODES,main_drugs=drugs,profiles_total=int(counts.sum()),profiles_projected=int(counts.loc[drugs].sum()),input_sha256={str(p.relative_to(R)):sha(p) for p in [I/'features_LOCAL_ONLY.parquet',I/'molecular_effect_summary.csv',R/'results/summaries/drug_geometry_20261002/observed_history_context.csv']},predictions=check,source_sha256=sha(Path(__file__)),future_labels_used_only_for_colors_and_error=True,diagnostic_projection_count=len(diagnostics),fit_seeds_for_original_space_association=list(range(1,11))),indent=2))
 print('COMPLETE',drugs,len(diagnostics),flush=True)
if __name__=='__main__':main()
