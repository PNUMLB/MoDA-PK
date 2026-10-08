"""Profile-level observer states; no future labels enter the projection."""
from pathlib import Path
import sys,json,hashlib,inspect
import numpy as np,pandas as pd,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE,trustworthiness
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from mhap_pk.data.traceable import load_profiles,indices_for_split
from mhap_pk.data import representations,make_batch
from mhap_pk.evaluation.history_protocols import initial_observation_inputs
from mhap_pk.baselines.low_data import build
O=R/'results/summaries/profile_tsne_20261002';P=R/'results/runs/TRACEABLE/evaluation_20260925/known__s1__molecular_relation'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def project(x,perplexity,seed):
 kw=dict(n_components=2,perplexity=perplexity,random_state=seed,init='random',learning_rate=50,method='exact')
 kw['max_iter' if 'max_iter' in inspect.signature(TSNE).parameters else 'n_iter']=1500
 return TSNE(**kw).fit_transform(x)
def main():
 torch.set_num_threads(1);O.mkdir(parents=True,exist_ok=False);ps,meta,_=load_profiles();tr,va,te=indices_for_split(meta,'known');z,ids,sc=representations(meta,tr,'unimol',17)
 art=json.loads((P/'artifacts.json').read_text());assert art['train']==tr.tolist() and art['test']==te.tolist() and art['scaler']==sc
 pop=torch.tensor(json.loads((P/'population.json').read_text())['log_parameters'],dtype=torch.double);m=build('molecular',len(sc['fitted_drugs']),pop);m.load_state_dict(torch.load(P/'selected.pt',weights_only=True));m.eval();hidden={};metadata=[];prediction=[]
 for role,index in [('train',tr),('test',te)]:
  b=make_batch(ps,meta,index,z,ids);args=initial_observation_inputs(b,2)
  with torch.no_grad():
   h=m.condition(*args[:5],drug_indices=args[6])['hidden'];out=m(*args)
  assert torch.equal(h,out['state']['hidden']);hidden[role]=h.numpy()
  for i,idx in enumerate(b.indices):
   prof=ps[idx];uid=json.dumps([str(prof[k]) for k in ['source_id','drug_id','participant_id','profile_id']],ensure_ascii=False);q=b.query_mask[i];err=out['mean'][i,q]-b.query_y[i,q]
   metadata.append(dict(role=role,profile_uid=uid,**{k:prof[k] for k in ['source_id','drug_id','participant_id','profile_id']},k=2,fit_seed=1,cutoff=float(args[2][i,args[4][i]].max()),queries=int(q.sum()),future_log_rmse=float(err.square().mean().sqrt()),vector=json.dumps(h[i].tolist())))
   if role=='test':
    for j in torch.where(q)[0].tolist():prediction.append(dict(observation_id=str(prof['query'].observation_id.iloc[j]),y_true_lnc=float(b.query_y[i,j]),y_pred_lnc=float(out['mean'][i,j])))
 raw=pd.read_parquet(P/'predictions.parquet');raw=raw[(raw.protocol=='first_k')&(raw.k==2)].set_index('observation_id');check=pd.DataFrame(prediction).set_index('observation_id');assert raw.index.is_unique and check.index.is_unique;assert np.array_equal(raw.loc[check.index,['y_true_lnc','y_pred_lnc']],check[['y_true_lnc','y_pred_lnc']])
 df=pd.DataFrame(metadata);assert not df.profile_uid.duplicated().any();scaler=StandardScaler().fit(hidden['train']);x=scaler.transform(hidden['test']);xy=project(x,15,17);test=df[df.role=='test'].reset_index(drop=True);test['tsne_x']=xy[:,0];test['tsne_y']=xy[:,1]
 test.to_csv(O/'profile_coordinates.csv',index=False);df.to_parquet(O/'profile_hidden_states.parquet',index=False);np.savez(O/'projection_inputs.npz',test_scaled=x,train_hidden=hidden['train'],test_hidden=hidden['test'],mean=scaler.mean_,scale=scaler.scale_)
 drugs=sorted(test.drug_id.unique());sources=sorted(test.source_id.unique());drug_colors=dict(zip(drugs,plt.get_cmap('tab20')(np.linspace(0,1,len(drugs)))));source_colors=dict(zip(sources,plt.get_cmap('tab10')(np.linspace(0,1,len(sources)))))
 fig,axs=plt.subplots(1,3,figsize=(14,5.8))
 for ax,col,palette,title in zip(axs[:2],['drug_id','source_id'],[drug_colors,source_colors],['A. Drug identity','B. Data source']):
  for name,g in test.groupby(col):ax.scatter(g.tsne_x,g.tsne_y,s=27,color=palette[name],label=name.replace('_',' '),edgecolors='white',linewidths=.35,alpha=.85)
  ax.set_title(title);ax.legend(loc='upper center',bbox_to_anchor=(.5,-.06),ncol=2,fontsize=6.4,frameon=False)
 im=axs[2].scatter(xy[:,0],xy[:,1],c=test.future_log_rmse,cmap='viridis',s=27,edgecolors='white',linewidths=.35);axs[2].set_title('C. Future prediction error');fig.colorbar(im,ax=axs[2],label='Profile log-RMSE',shrink=.8)
 for ax in axs:ax.set_xticks([]);ax.set_yticks([])
 fig.suptitle(f'MoDE-PK (Ours): observer state after two observations\n{len(test)} held-out profiles | one shared projection; colors do not enter t-SNE',fontsize=12);fig.tight_layout(rect=(0,.22,1,.91));fig.savefig(O/'Profile_tSNE.pdf');fig.savefig(O/'Profile_tSNE.png',dpi=190);plt.close(fig)
 diagnostics=[];fig,axs=plt.subplots(3,3,figsize=(10,9));dist=cdist(x,x);np.fill_diagonal(dist,np.inf);nn=np.argsort(dist,axis=1)[:,:5];sens=[]
 for i,perp in enumerate([5,15,30]):
  for j,seed in enumerate([17,29,43]):
   v=xy if (perp,seed)==(15,17) else project(x,perp,seed);d=cdist(v,v);np.fill_diagonal(d,np.inf);n2=np.argsort(d,axis=1)[:,:5];overlap=float(np.mean([len(set(a)&set(b))/5 for a,b in zip(nn,n2)]));tw=float(trustworthiness(x,v,n_neighbors=5));diagnostics.append(dict(perplexity=perp,projection_seed=seed,neighbor_overlap5=overlap,trustworthiness5=tw))
   axs[i,j].scatter(v[:,0],v[:,1],c=[drug_colors[d] for d in test.drug_id],s=15);axs[i,j].set_title(f'p={perp}, seed={seed}; overlap={overlap:.2f}',fontsize=9);axs[i,j].set_xticks([]);axs[i,j].set_yticks([])
   for ix,(a,b) in enumerate(v):sens.append(dict(profile_uid=test.profile_uid.iloc[ix],perplexity=perp,projection_seed=seed,x=a,y=b))
 fig.suptitle('Profile-level projection sensitivity | same observer states',fontsize=13);fig.tight_layout(rect=(0,0,1,.96));fig.savefig(O/'Profile_tSNE_sensitivity.pdf');fig.savefig(O/'Profile_tSNE_sensitivity.png',dpi=150);plt.close(fig)
 pd.DataFrame(diagnostics).to_csv(O/'projection_diagnostics.csv',index=False);pd.DataFrame(sens).to_csv(O/'sensitivity_coordinates.csv',index=False)
 manifest=dict(status='completed',checkpoint=str(P/'selected.pt'),checkpoint_sha256=sha(P/'selected.pt'),original_predictions_sha256=sha(P/'predictions.parquet'),prediction_parity_rows=len(check),exact_prediction_parity=True,train_profiles=len(tr),test_profiles=len(te),test_participants=int(test[['source_id','participant_id']].drop_duplicates().shape[0]),drugs=len(drugs),sources=len(sources),dimension=32,k=2,fit_seed=1,scaler='Fit on training-profile hidden states only',projection='Descriptive t-SNE fitted jointly to test feature vectors; no predictive tuning',future_labels='Used only for point color and checkpoint parity, never feature or projection',primary=dict(perplexity=15,projection_seed=17,learning_rate=50,iterations=1500,method='exact'),source_sha256=sha(Path(__file__)),files={p.name:sha(p) for p in O.iterdir() if p.is_file()})
 (O/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({k:manifest[k] for k in ['train_profiles','test_profiles','test_participants','drugs','sources','prediction_parity_rows']},indent=2));print(pd.DataFrame(diagnostics).to_string(index=False))
if __name__=='__main__':main()
