"""Re-extract the adopted model and matched branch-control representations."""
from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import trustworthiness
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
import umap
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'src')]
from mhap_pk.data.traceable import load_profiles,indices_for_split
from mhap_pk.data import representations,make_batch
from mhap_pk.evaluation.history_protocols import initial_observation_inputs
from mhap_pk.models.priors.hybrid_controls import build_control
S=R/'results/summaries/hybrid_primary_20261002';B=R/'results/runs/HYBRID_PRIMARY/20261002';torch.set_num_threads(1)
ps,meta,_=load_profiles();tr,va,te=indices_for_split(meta,'known');z,ids,sc=representations(meta,tr,'unimol',17);names={'full':'MoDE-PK (Ours)','without_molecule':'Without molecular input','without_identity':'Without identity input'};features=[];coords=[];checks=[];original=[]
for variant in names:
 for seed in range(1,11):
  root=(R/'results/runs/MOLECULAR_IDENTITY/20261002' if variant=='full' else B/'panel'/variant)/f'known__s{seed}__molecular';assert json.loads((root/'run_manifest.json').read_text())['status']=='completed'
  a=json.loads((root/'artifacts.json').read_text());assert a['train']==tr.tolist() and a['scaler']==sc
  m=build_control(variant,len(sc['fitted_drugs']),torch.tensor(json.loads((root/'population.json').read_text())['log_parameters'],dtype=torch.double));m.load_state_dict(torch.load(root/'selected.pt',weights_only=True));m.eval();hs={};details={}
  for role,ix in [('train',tr),('test',te)]:
   batch=make_batch(ps,meta,ix,z,ids);args=initial_observation_inputs(batch,2)
   with torch.no_grad():o=m(*args)
   h=o['state']['hidden'].numpy();assert np.isfinite(h).all();hs[role]=h;details[role]=[]
   for i,idx in enumerate(ix):
    p=ps[idx];end=float(args[2][i,args[4][i]].max());row=dict(model=variant,seed=seed,role=role,**{k:p[k] for k in ['source_id','drug_id','participant_id','profile_id']},last_observed_log=float(args[3][i,args[4][i]][-1]),support_end=end,profile_log_rmse=float(((o['mean'][i,batch.query_mask[i]]-batch.query_y[i,batch.query_mask[i]])**2).mean().sqrt()));details[role].append(row);features.append(row|dict(vector=json.dumps(h[i].tolist())))
   if role=='test':
    actual=np.concatenate([o['mean'][i,batch.query_mask[i]].numpy() for i in range(len(ix))]);ref=pd.read_parquet(root/'predictions.parquet');ref=ref[(ref.k==2)&(ref.protocol=='first_k')].set_index('observation_id');wanted=ref.loc[[str(q) for idx in ix for q in ps[idx]['query'].observation_id]].y_pred_lnc.to_numpy();err=float(abs(actual-wanted).max());assert err<1e-12;checks.append(dict(model=variant,seed=seed,max_prediction_difference=err,checkpoint=str((root/'selected.pt').relative_to(R)),checkpoint_sha256=hashlib.sha256((root/'selected.pt').read_bytes()).hexdigest()))
  scaled=StandardScaler().fit(hs['train']).transform(hs['test']);info=pd.DataFrame(details['test'])
  for drug,index in info.groupby('drug_id').groups.items():
   ii=np.array(list(index));xx=scaled[ii];n=len(ii)
   if n<10:continue
   distances=pdist(xx)
   def corr(v):
    b=pdist(np.asarray(v)[:,None]);return float(spearmanr(distances,b).statistic) if np.ptp(distances)>1e-12 and np.ptp(b)>1e-12 else np.nan
   original.append(dict(model=variant,seed=seed,drug_id=drug,profiles=n,distance_association_last_concentration=corr(info.loc[ii,'last_observed_log']),distance_association_future_error=corr(info.loc[ii,'profile_log_rmse'])))
   if seed==1:
    u=umap.UMAP(n_neighbors=5,min_dist=.1,random_state=17,init='random',n_components=2,n_epochs=500,n_jobs=1).fit_transform(xx);trust=float(trustworthiness(xx,u,n_neighbors=3))
    for i,xy in zip(ii,u):coords.append(info.iloc[i].to_dict()|dict(x=float(xy[0]),y=float(xy[1]),trustworthiness=trust))
f=pd.DataFrame(features);f.to_parquet(S/'observer_features_LOCAL_ONLY.parquet',index=False);c=pd.DataFrame(coords);c.to_csv(S/'umap_coordinates_LOCAL_ONLY.csv',index=False);pd.DataFrame(checks).to_csv(S/'representation_replay_checks.csv',index=False);pd.DataFrame(original).to_csv(S/'representation_original_space.csv',index=False)
drugs=sorted(c.drug_id.unique())
for color,title,name in [('last_observed_log','Last observed log concentration','UMAP_observed'),('profile_log_rmse','Future profile log-RMSE','UMAP_future_error')]:
 fig,axs=plt.subplots(len(drugs),3,figsize=(11,3*len(drugs)),squeeze=False)
 for row,drug in enumerate(drugs):
  g=c[c.drug_id==drug];lo=g[color].min();hi=g[color].max()
  for col,model in enumerate(names):
   h=g[g.model==model];ax=axs[row,col];im=ax.scatter(h.x,h.y,c=h[color],cmap='viridis',vmin=lo,vmax=hi,s=30,edgecolors='white',linewidths=.3);ax.set_title(drug.replace('_',' ').capitalize()+' | '+names[model],fontsize=9);ax.set_xticks([]);ax.set_yticks([]);ax.set_xlabel(f'{len(h)} profiles; trustworthiness={h.trustworthiness.iloc[0]:.2f}',fontsize=8)
  fig.colorbar(im,ax=list(axs[row]),fraction=.02,pad=.02,label=title)
 fig.subplots_adjust(left=.03,right=.88,top=.94,bottom=.04,wspace=.2,hspace=.35);fig.suptitle('Known-drug | same profiles and observer layer | independent UMAP projections',fontsize=11);fig.savefig(S/(name+'.pdf'));fig.savefig(S/(name+'.png'),dpi=160);plt.close(fig)
(S/'representation_manifest.json').write_text(json.dumps(dict(status='completed',fits=30,parity_checks=len(checks),primary='full hybrid',controls=list(names)[1:],scope='Known only; training-standardized last GRU hidden states; initial k2; relationOFF',projection_seed=17,model_seed_for_projection=1,eligible_drugs=drugs,projection_uses_future_labels=False,future_labels_for_colors_only=True,axes_independently_fitted=True),indent=2));print('Hybrid UMAP and original-space analyses complete')
