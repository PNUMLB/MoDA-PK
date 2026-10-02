"""Descriptive, model-specific profile features; no new training or label-based selection."""
from pathlib import Path
import sys,json,hashlib,inspect
import numpy as np,pandas as pd,torch
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'src'),str(R/'scripts')]
from mhap_pk.data.traceable import load_profiles,indices_for_split
from mhap_pk.data import representations,make_batch
from mhap_pk.evaluation.history_protocols import initial_observation_inputs
from mhap_pk.baselines.population_vi import PopulationVI,map_forecast
from mhap_pk.relational import floor_curve
from complete_comparison_refresh import build
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE,trustworthiness
import umap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=R/'results/summaries/profile_comparison_20261002'
BASE=R/'results/runs/TRACEABLE/evaluation_20260925'
PATHS={m:BASE/f'known__s1__{m}' for m in ['molecular_relation','dcm','unipk']}
PATHS['deepct']=R/'results/runs/COMPARISON_REFRESH/20260929/corrected120/known__s1__deepct'
PATHS['poppk']=R/'results/runs/TRACEABLE_POPPK/evaluation_20260925/full__s1'
NAMES={'molecular_relation':'MoDE-PK (Ours)','dcm':'Deterministic DCM','deepct':'DeepCt','unipk':'Uni-PK','poppk':'PopPK + MAP'}
FEATURES={'molecular_relation':'32-dimensional recurrent observer hidden state','dcm':'5 effective log-PK parameters + observed-history log offset','deepct':'13 log compartment parameters + observed-history log offset','unipk':'64-dimensional molecular dynamics context at zero concentration + observed-history log offset','poppk':'5 effective log-PK parameters after support-only MAP'}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 torch.set_num_threads(1);O.mkdir(parents=True,exist_ok=False)
 ps,meta,_=load_profiles();tr,va,te=indices_for_split(meta,'known');z,ids,sc=representations(meta,tr,'unimol',17)
 rows=[];checks=[];checkpoints={};matrices={}
 for mode,path in PATHS.items():
  print('EXTRACT',mode,flush=True);h={}
  if mode!='poppk':
   art=json.loads((path/'artifacts.json').read_text());assert art['train']==tr.tolist() and art['test']==te.tolist() and art['scaler']==sc
   pop=torch.tensor(json.loads((path/'population.json').read_text())['log_parameters'],dtype=torch.double)
   m=build('molecular' if mode=='molecular_relation' else mode,len(sc['fitted_drugs']),pop);m.load_state_dict(torch.load(path/'selected.pt',weights_only=True));m.eval();checkpoints[mode]={str(path/'selected.pt'):sha(path/'selected.pt')}
  else:
   art=json.loads((path/'artifacts.json').read_text());cfg=json.loads((path/'config.resolved.json').read_text());models={};checkpoints[mode]={}
   for drug,a in art.items():
    m=PopulationVI(a['calibration']['log_parameters'],len(a['people']));cp=path/f'{drug}_selected.pt';m.load_state_dict(torch.load(cp,weights_only=True));m.eval();models[drug]=m;checkpoints[mode][str(cp)]=sha(cp)
  for role,index in [('train',tr),('test',te)]:
   b=make_batch(ps,meta,index,z,ids);args=initial_observation_inputs(b,2);zz,ev,st,sy,sm,qt,di=args
   with torch.no_grad():
    if mode=='molecular_relation':
     out=m(*args);v=out['state']['hidden'];y=out['mean']
    elif mode=='poppk':
     v=torch.empty(len(index),5,dtype=torch.double);y=torch.empty_like(qt)
     for drug,model in models.items():
      loc=torch.tensor([j for j,idx in enumerate(b.indices) if ps[idx]['drug_id']==drug],dtype=torch.long)
      yy,d=map_forecast(model,ev[loc],st[loc],sy[loc],sm[loc],qt[loc],steps=cfg['map_steps']);v[loc]=d['parameters'];y[loc]=yy
    else:
     if mode=='dcm':
      context,_,_=m.prior(zz,di);past=floor_curve(context,ev,st)
     elif mode=='deepct':context=m.head(zz).clamp_min(1e-8).log();past=m.curve(zz,ev,st)
     else:context=m.native.cmptmodel.net[:2](torch.cat([zz,zz.new_zeros(len(zz),1)],1));past=m.curve(zz,ev,st)
     shift=torch.where(sm,sy-past,0.).sum(1)/(sm.sum(1)+.09);v=torch.cat([context,shift[:,None]],1)
     y=m(*args)['mean'] if role=='test' else None
   assert torch.isfinite(v).all();h[role]=v.numpy();pred=[]
   for j,idx in enumerate(b.indices):
    p=ps[idx];uid=json.dumps([str(p[k]) for k in ['source_id','drug_id','participant_id','profile_id']],ensure_ascii=False)
    rows.append(dict(model=mode,role=role,profile_uid=uid,**{k:p[k] for k in ['source_id','drug_id','participant_id','profile_id']},vector=json.dumps(v[j].tolist())))
    if role=='test':
     for q in torch.where(b.query_mask[j])[0].tolist():pred.append(dict(observation_id=str(p['query'].observation_id.iloc[q]),y_true_lnc=float(b.query_y[j,q]),y_pred_lnc=float(y[j,q])))
   if role=='test':
    raw=pd.read_parquet(path/'predictions.parquet');raw=raw[(raw.protocol=='first_k')&(raw.k==2)].set_index('observation_id');p=pd.DataFrame(pred).set_index('observation_id');assert raw.index.is_unique and p.index.is_unique and set(p.index)==set(raw.index)
    assert np.array_equal(raw.loc[p.index,'y_true_lnc'],p.y_true_lnc)
    delta=float(np.max(np.abs(raw.loc[p.index,'y_pred_lnc']-p.y_pred_lnc)));assert delta<1e-8,(mode,delta)
    checks.append(dict(model=mode,query_count=len(p),max_prediction_difference=delta))
  scaler=StandardScaler().fit(h['train']);matrices[mode]=scaler.transform(h['test']);np.savez(O/f'{mode}_features.npz',train=h['train'],test=h['test'],test_scaled=matrices[mode],mean=scaler.mean_,scale=scaler.scale_)
 df=pd.DataFrame(rows);assert not df.duplicated(['model','role','profile_uid']).any();df.to_parquet(O/'profile_features.parquet',index=False);pd.DataFrame(checks).to_csv(O/'prediction_parity.csv',index=False)
 coordinates=[];diagnostics=[]
 for mode,x in matrices.items():
  print('PROJECT',mode,flush=True);d=df[(df.model==mode)&(df.role=='test')].drop(columns='vector').reset_index(drop=True)
  kw=dict(n_components=2,perplexity=15,random_state=17,init='random',learning_rate=50,method='exact');kw['max_iter' if 'max_iter' in inspect.signature(TSNE).parameters else 'n_iter']=1500
  for name,proj in [('t-SNE',TSNE(**kw)),('UMAP',umap.UMAP(n_neighbors=15,min_dist=.1,n_components=2,random_state=17,n_jobs=1))]:
   xy=proj.fit_transform(x);t=d.copy();t['projection']=name;t['x']=xy[:,0];t['y']=xy[:,1];coordinates.append(t);diagnostics.append(dict(model=mode,projection=name,dimension=x.shape[1],profiles=len(x),trustworthiness5=float(trustworthiness(x,xy,n_neighbors=5))))
 c=pd.concat(coordinates,ignore_index=True);c.to_csv(O/'coordinates.csv',index=False);pd.DataFrame(diagnostics).to_csv(O/'projection_diagnostics.csv',index=False)
 plot(O)
 manifest=dict(status='completed',new_training=False,evaluation='Known-drug; pooled neural models; drug-specific PopPK; initial k=2; fit seed 1',train_profiles=len(tr),test_profiles=len(te),future_labels_used_for_features=False,features=FEATURES,checkpoints=checkpoints,prediction_parity=checks,scaling='Per-model StandardScaler fit to training profiles only',projection='Each model projected independently on the same held-out profiles; no shared axes or clustering superiority inference',tsne=kw,umap=dict(n_neighbors=15,min_dist=.1,random_state=17,n_jobs=1),source_sha256=sha(Path(__file__)),files={p.name:sha(p) for p in O.iterdir() if p.is_file()})
 (O/'manifest.json').write_text(json.dumps(manifest,indent=2));print('COMPLETED',O,flush=True)
def plot(out):
 c=pd.read_csv(out/'coordinates.csv');drugs=sorted(c.drug_id.unique());palette=dict(zip(drugs,plt.get_cmap('tab20')(np.linspace(0,1,len(drugs)))))
 for proj in ['t-SNE','UMAP']:
  fig,axs=plt.subplots(2,3,figsize=(11,7));handles=[]
  for ax,mode in zip(axs.flat,PATHS):
   d=c[(c.model==mode)&(c.projection==proj)]
   for drug in drugs:
    g=d[d.drug_id==drug];h=ax.scatter(g.x,g.y,c=[palette[drug]],s=26,edgecolors='white',linewidths=.35,alpha=.85,label=drug.replace('_',' ').capitalize())
    if mode=='molecular_relation':handles.append(h)
   ax.set_title(NAMES[mode],fontsize=13);ax.set_xticks([]);ax.set_yticks([])
  axs.flat[-1].axis('off');axs.flat[-1].legend(handles=handles,loc='center',frameon=False,fontsize=11,title='Drug')
  fig.suptitle(f'{proj}: 101 held-out profiles after two observations',fontsize=15);fig.tight_layout(rect=(0,0,1,.95));stem='Profile_comparison_'+proj.replace('-','');fig.savefig(out/(stem+'.pdf'));fig.savefig(out/(stem+'.png'),dpi=180);plt.close(fig)
if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='--plot':plot(Path(sys.argv[2]))
 else:main()
