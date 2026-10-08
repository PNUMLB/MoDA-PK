"""Replace Ours in the cross-model projection; verify baseline feature provenance."""
from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import trustworthiness
import umap
R=Path(__file__).resolve().parents[1];S=R/'results/summaries/hybrid_primary_20261002';OLD=R/'results/summaries/profile_comparison_20261002';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
f=pd.read_parquet(OLD/'profile_features.parquet');f=f[f.model!='molecular_relation'].copy();manifest=json.loads((OLD/'manifest.json').read_text());inventory=pd.read_csv(S/'primary_inventory.csv');verified=[]
for mode,paths in manifest['checkpoints'].items():
 if mode=='molecular_relation':continue
 for path,h in paths.items():
  cp=Path(path);assert sha(cp)==h
  if mode!='poppk':
   matches=inventory[(inventory.model==mode)&(inventory.run_id==str(cp.parent.relative_to(R)))];assert len(matches)==1
   selection=json.loads((cp.parent/'selection.json').read_text());assert selection['checkpoint_sha256']==h
   assert sha(cp.parent/'predictions.parquet')==matches.iloc[0].prediction_sha256
  else:
   # Drug-specific population checkpoint hashes are in the existing source manifest.
   run=cp.parent;raw=run/'predictions.parquet';rm=json.loads((run/'run_manifest.json').read_text());assert sha(raw)==rm['prediction_sha256']
  verified.append(dict(model=mode,path=str(cp.relative_to(R)),sha256=h))
a=pd.read_parquet(S/'observer_features_LOCAL_ONLY.parquet');a=a[(a.model=='full')&(a.seed==1)].copy();a['model']='ours';keys=['source_id','drug_id','participant_id','profile_id'];a['profile_uid']=a.apply(lambda row:json.dumps([str(row[k]) for k in keys],ensure_ascii=False),axis=1);f=pd.concat([a[f.columns],f],ignore_index=True)
for role,g in f.groupby('role'):
 ref=set(g[g.model=='ours'].profile_uid)
 for m,h in g.groupby('model'):assert set(h.profile_uid)==ref and h.profile_uid.is_unique
coords=[];diagnostics=[];names={'ours':'MoDE-PK (Ours)','dcm':'DCM: parameters + offset','deepct':'DeepCt: parameters + offset','unipk':'Uni-PK: context + offset','poppk':'PopPK: MAP parameters'}
for m in names:
 train=f[(f.model==m)&(f.role=='train')].sort_values('profile_uid');test=f[(f.model==m)&(f.role=='test')].sort_values('profile_uid');x=StandardScaler().fit(np.stack(train.vector.map(json.loads))).transform(np.stack(test.vector.map(json.loads)));u=umap.UMAP(n_neighbors=15,min_dist=.1,n_components=2,random_state=17,init='random',n_epochs=500,n_jobs=1).fit_transform(x);c=test.drop(columns='vector').copy();c['x']=u[:,0];c['y']=u[:,1];coords.append(c);diagnostics.append(dict(model=m,dimensions=x.shape[1],profiles=len(test),trustworthiness5=float(trustworthiness(x,u,n_neighbors=5))))
c=pd.concat(coords,ignore_index=True);c.to_csv(S/'cross_model_umap_LOCAL_ONLY.csv',index=False);pd.DataFrame(diagnostics).to_csv(S/'cross_model_projection_quality.csv',index=False);f.to_parquet(S/'cross_model_features_LOCAL_ONLY.parquet',index=False)
drugs=sorted(c.drug_id.unique());palette=dict(zip(drugs,plt.get_cmap('tab20')(np.linspace(0,1,len(drugs)))));fig,axs=plt.subplots(2,3,figsize=(11,7));handles=[]
for ax,m in zip(axs.flat,names):
 for drug in drugs:
  g=c[(c.model==m)&(c.drug_id==drug)];h=ax.scatter(g.x,g.y,c=[palette[drug]],s=26,edgecolors='white',linewidths=.35,label=drug.replace('_',' ').capitalize())
  if m=='ours':handles.append(h)
 ax.set_title(names[m],fontsize=10);ax.set_xticks([]);ax.set_yticks([])
axs.flat[-1].axis('off');axs.flat[-1].legend(handles=handles,loc='center',frameon=False,fontsize=10,title='Drug');fig.suptitle('Known-drug | same 101 profiles | independently projected model-specific features',fontsize=11);fig.tight_layout(rect=(0,0,1,.95));fig.savefig(S/'Cross_model_UMAP.pdf');fig.savefig(S/'Cross_model_UMAP.png',dpi=170);plt.close(fig)
(S/'cross_model_representation_manifest.json').write_text(json.dumps(dict(status='completed',baseline_checkpoints=verified,old_features_sha256=sha(OLD/'profile_features.parquet'),old_manifest_sha256=sha(OLD/'manifest.json'),ours_features_sha256=sha(S/'observer_features_LOCAL_ONLY.parquet'),feature_definitions={k:v for k,v in manifest['features'].items() if k!='molecular_relation'}|dict(ours='32-dimensional concatenation-model GRU state'),scope='Known seed1 k2; all101 testprofiles; five heterogeneous predictive feature definitions; not shared axes or efficacy ranking'),indent=2));print('Cross-model UMAP updated and baseline checkpoint hashes verified')
