"""PCA of actual learned representations; no fitted projection uses held-out drugs."""
from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'));sys.path.insert(0,str(R/'scripts'))
from complete_comparison_refresh import build
from mhap_pk.data import representations
from mhap_pk.data.traceable import load_profiles,indices_for_split
OLD=R/'results/runs/TRACEABLE/evaluation_20260925';NEW=R/'results/runs/COMPARISON_REFRESH/20260929/corrected120';O=R/'results/summaries/comparison_refresh_20260929';O.mkdir(exist_ok=True,parents=True);(O/'figures').mkdir(exist_ok=True)
N={'molecular_relation':'MoDE-PK: molecular context','dcm':'DCM: molecular context','deepct':'DeepCt: penultimate layer','unipk':'Uni-PK: dynamics hidden layer at C=0','poppk_vi_map':'PopPK: population parameters (not an embedding)','frozen_unimol':'Frozen Uni-Mol input'}
def label_points(ax,xy,names):
 ax.margins(.15);limits=np.array([ax.get_xlim(),ax.get_ylim()]);norm=(xy-limits[:,0])/(limits[:,1]-limits[:,0]);placed=[]
 for (x,y),point,name in zip(norm,xy,names):
  width=min(.38,.013*len(name));height=.034;best=None
  for dy in [0,.04,-.04,.08,-.08,.12,-.12,.16,-.16,.20,-.20,.24,-.24]:
   for dx in [.018,-width-.018,.08,-width-.08]:
    tx=float(np.clip(x+dx,.01,.99-width));ty=float(np.clip(y+dy,.02,.96));box=(tx,ty,tx+width,ty+height)
    overlap=sum(max(0,min(box[2],b[2])-max(box[0],b[0]))*max(0,min(box[3],b[3])-max(box[1],b[1])) for b in placed)
    score=overlap*10000+(tx-x)**2+(ty-y)**2
    if best is None or score<best[0]:best=(score,tx,ty,box)
  _,tx,ty,box=best;placed.append(box);ax.annotate(name,point,xytext=(tx,ty),textcoords='axes fraction',fontsize=6,ha='left',va='bottom',arrowprops=dict(arrowstyle='-',color='#888',lw=.4),annotation_clip=True)
ps,meta,_=load_profiles();torch.set_num_threads(1);drugs=sorted(meta.drug_id.unique());ix=[meta.index[meta.drug_id==d][0] for d in drugs];records=[];stats=[];manifest=[]
with PdfPages(O/'figures/representations_all_folds.pdf') as atlas:
 for fold in (['known'] if '--known-only' in sys.argv else ['known']+[f'loso_{i:02d}' for i in range(9)]):
  tr,va,te=indices_for_split(meta,fold);z,ids,sc=representations(meta,tr,'unimol',17);pop=torch.tensor(json.loads((OLD/'calibration'/fold/'population.json').read_text())['log_parameters'],dtype=torch.double)
  train=np.array([d in set(meta.iloc[tr].drug_id) for d in drugs]);test=np.array([d in set(meta.iloc[te].drug_id) for d in drugs]);roles=np.where(train,'train',np.where(test,'test','validation'))
  reps={'frozen_unimol':z[ix].numpy()}
  for mode in ['molecular_relation','dcm','deepct','unipk']:
   parent=(NEW if mode=='deepct' else OLD)/f'{fold}__s1__{mode}';cp=parent/'selected.pt';m=build('molecular' if mode=='molecular_relation' else mode,len(sc['fitted_drugs']),pop);m.load_state_dict(torch.load(cp,weights_only=True));m.eval()
   with torch.no_grad():
    x=z[ix]
    if mode in ['molecular_relation','dcm']:a=m.prior(x)[1]
    elif mode=='deepct':a=m.head.predict[:-1](x)
    else:a=m.native.cmptmodel.net[:2](torch.cat([x,torch.zeros(len(x),1,dtype=x.dtype)],1))
   reps[mode]=a.numpy();manifest.append(dict(fold=fold,model=mode,checkpoint=str(cp.relative_to(R)),sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),dimension=a.shape[1]))
  if fold=='known':
   a=[]
   for drug in drugs:
    cp=R/'results/runs/TRACEABLE_POPPK/evaluation_20260925/full__s1'/f'{drug}_selected.pt';sd=torch.load(cp,weights_only=True);manifest.append(dict(fold=fold,model='poppk_vi_map',drug_id=drug,checkpoint=str(cp.relative_to(R)),sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),dimension=8));a.append(torch.cat([sd['mu'],sd['log_omega'],sd['log_sigma'].reshape(1)]).numpy())
   reps['poppk_vi_map']=np.stack(a)
  fig,axs=plt.subplots(2,3,figsize=(15,9));axs=axs.ravel()
  for ax,(mode,a) in zip(axs,reps.items()):
   assert np.isfinite(a).all();scaler=StandardScaler().fit(a[train]);scaled=scaler.transform(a);pc=PCA(2).fit(scaled[train]);xy=pc.transform(scaled)
   for i,drug in enumerate(drugs):
    marker='o' if roles[i]=='train' else ('*' if roles[i]=='test' else '^');ax.scatter(*xy[i],marker=marker,s=65 if marker!='*' else 150,color=plt.get_cmap('tab20')(i/len(drugs)),edgecolors='black',linewidths=.4)
    records.append(dict(fold=fold,seed=1,model=mode,drug_id=drug,role=roles[i],pc1=xy[i,0],pc2=xy[i,1],dimension=a.shape[1],vector=json.dumps(a[i].tolist())))
   label_points(ax,xy,['R-lipoic acid' if d=='r_lipoic_acid' else d.replace('_',' ').capitalize() for d in drugs])
   ax.set_title(N[mode],fontsize=9);ax.set_xlabel(f'PC1 ({100*pc.explained_variance_ratio_[0]:.1f}%)');ax.set_ylabel(f'PC2 ({100*pc.explained_variance_ratio_[1]:.1f}%)');ax.grid(alpha=.15)
   dist=pdist(scaled);chem=pdist(reps['frozen_unimol']);rho=spearmanr(dist,chem).statistic
   stats.append(dict(fold=fold,model=mode,dimension=a.shape[1],training_drugs=int(train.sum()),pca_variance_2d=float(pc.explained_variance_ratio_.sum()),input_distance_spearman=float(rho),interpretation='descriptive drug pairs, dependent observations, not efficacy evidence'))
  for ax in axs[len(reps):]:ax.axis('off')
  fig.suptitle(f'{fold}, seed 1 | PCA/scaling fitted on training drugs only\nCircles: training drugs; stars: held-out drugs; triangles: validation drugs. Axes are model-specific.',fontsize=11);fig.tight_layout();atlas.savefig(fig)
  if fold=='known':fig.savefig(O/'figures/representations_known.pdf',bbox_inches='tight');fig.savefig(O/'figures/representations_known.png',dpi=160,bbox_inches='tight')
  plt.close(fig)
pd.DataFrame(records).to_parquet(O/'representations.parquet',index=False);pd.DataFrame(stats).to_csv(O/'representation_diagnostics.csv',index=False);(O/'representation_manifest.json').write_text(json.dumps(dict(code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),models=manifest,projection='train-drug standardized PCA per model and fold, seed1',joint_poppk='not applicable',population_is_parameter_space=True),indent=2))
print('Representations:',len(records),'model/drug/fold points')
