"""Matched retrained input controls: original-space geometry and drug-wise accuracy."""
from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from mhap_pk.data.traceable import load_profiles,indices_for_split
from mhap_pk.data import representations,make_batch
from mhap_pk.baselines.low_data import build
from mhap_pk.evaluation.history_protocols import initial_observation_inputs
O=R/'results/summaries/molecular_conditioning_20261002'
N={'no_molecule':'Without molecular features\nRelations OFF','molecular':'With Uni-Mol features\nRelations OFF','molecular_relation':'MoDE-PK (Ours)\nWith Uni-Mol + relations'}
def parent(f,s,m):return R/('results/runs/TRACEABLE/evaluation_20260925' if s<=5 else 'results/runs/TEN_SEEDS/20261001/panel')/f'{f}__s{s}__{m}'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def plot_results():
 c=pd.read_csv(O/'coordinates_LOCAL_ONLY.csv');e=pd.read_csv(O/'molecular_effect_summary.csv')
 drugs=sorted(c.drug_id.unique());palette=dict(zip(drugs,plt.get_cmap('tab20')(np.linspace(0,1,len(drugs)))))
 fig,axs=plt.subplots(1,3,figsize=(12,4.8))
 for ax,mode in zip(axs,N):
  t=c[c.model==mode]
  for drug,d in t.groupby('drug_id'):ax.scatter(d.x,d.y,color=palette[drug],s=24,label=drug.replace('_',' '),edgecolors='white',linewidths=.3)
  ax.set_title(N[mode],fontsize=11);ax.set_xlabel(f'Projection distortion = {t.stress.iloc[0]:.3f}',fontsize=9);ax.set_xticks([]);ax.set_yticks([])
 handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4,fontsize=8,frameon=False);fig.suptitle('Same held-out profiles and observer layer; separately retrained models',fontsize=12);fig.tight_layout(rect=(0,.18,1,.93));fig.savefig(O/'Molecular_conditioning.pdf');fig.savefig(O/'Molecular_conditioning.png',dpi=160);plt.close(fig)
 fig,axs=plt.subplots(1,2,figsize=(11,5))
 for ax,track in zip(axs,['Known-drug','Joint holdout']):
  t=e[e.track==track].set_index('drug_id').loc[drugs];ax.errorbar(t.delta_log_rmse,range(len(drugs)),xerr=t.sd,fmt='o',color='#287c8e',capsize=2);ax.axvline(0,color='gray',ls='--');ax.set_yticks(range(len(drugs)),[x.replace('_',' ') for x in drugs]);ax.set_title(track);ax.set_xlabel('Log-RMSE: molecular minus no-molecule\n< 0: molecular features improve forecasting');ax.grid(axis='x',alpha=.2)
 fig.suptitle('Molecular input effect with both neighborhood losses OFF | ten seeds');fig.tight_layout();fig.savefig(O/'Molecular_prediction_effect.pdf');plt.close(fig)
 # Matched-budget gain: same source learning-curve records, easier sign interpretation.
 low=pd.read_csv(R/'results/summaries/sample_size_effects_20261002/paired_by_seed.csv');low=low[low.k==2].copy();low['budget']=np.where(low.target_n==low.full_target_n,'Full',low.target_n.astype(str));low['gain']=-low.delta_rmse
 fig,axs=plt.subplots(1,2,figsize=(11,6));low_summary=[]
 shared_limit=float(low.groupby(['condition','drug_id','budget']).gain.mean().abs().max())
 for ax,mode,title in zip(axs,['pooled_molecular','pooled_molecular_relation'],['Data pooling; relations OFF','MoDE-PK pooling + relations']):
  t=low[low.condition==mode].groupby(['drug_id','budget']).gain.mean().unstack().reindex(index=drugs,columns=['2','4','8','16','Full']);v=t.to_numpy();lim=np.nanmax(np.abs(low.gain));lim=shared_limit;im=ax.imshow(np.ma.masked_invalid(v),cmap='RdBu',vmin=-lim,vmax=lim,aspect='auto')
  for i in range(len(drugs)):
   for j in range(5):
    value=v[i,j];ax.text(j,i,'N/A' if np.isnan(value) else f'{value:+.3f}',ha='center',va='center',fontsize=8,color='black' if np.isnan(value) or abs(value)<lim*.55 else 'white')
  ax.set_xticks(range(5),['2','4','8','16','Full']);ax.set_yticks(range(len(drugs)),[x.replace('_',' ') for x in drugs]);ax.set_xlabel('Target-drug training participants');ax.set_title(title);fig.colorbar(im,ax=ax,shrink=.65,label='Single-drug minus pooled log-RMSE')
 fig.suptitle('Does adding other drugs help? Positive = pooling improves accuracy\nSame target participants, validation/test, and k=2; ten-seed mean');fig.tight_layout(rect=(0,0,1,.93));fig.savefig(O/'Pooling_benefit.pdf');fig.savefig(O/'Pooling_benefit.png',dpi=160);plt.close(fig)

def main():
 torch.set_num_threads(1);O.mkdir(parents=True,exist_ok=False)
 ps,meta,_=load_profiles();tr,va,te=indices_for_split(meta,'known');z,ids,sc=representations(meta,tr,'unimol',17)
 geometry=[];features=[];coords=[];parity=[];provenance=[];scores=[];distances={}
 for mode in N:
  for seed in range(1,11):
   path=parent('known',seed,mode);a=json.loads((path/'artifacts.json').read_text());assert a['train']==tr.tolist() and a['test']==te.tolist() and a['scaler']==sc
   pop=torch.tensor(json.loads((path/'population.json').read_text())['log_parameters'],dtype=torch.double)
   m=build('molecular' if mode=='molecular_relation' else mode,len(sc['fitted_drugs']),pop);m.load_state_dict(torch.load(path/'selected.pt',weights_only=True));m.eval();hs={};check=[];rows=[]
   for role,index in [('train',tr),('test',te)]:
    b=make_batch(ps,meta,index,torch.zeros_like(z) if mode=='no_molecule' else z,ids);args=initial_observation_inputs(b,2)
    with torch.no_grad():out=m(*args)
    h=out['state']['hidden'].numpy();assert np.isfinite(h).all();hs[role]=h
    for i,idx in enumerate(b.indices):
     p=ps[idx];uid=json.dumps([str(p[k]) for k in ['source_id','drug_id','participant_id','profile_id']]);row=dict(model=mode,seed=seed,role=role,profile_uid=uid,drug_id=p['drug_id'],vector=json.dumps(h[i].tolist()));features.append(row)
     if role=='test':
      rows.append(row)
      for q in torch.where(b.query_mask[i])[0].tolist():check.append(dict(observation_id=str(p['query'].observation_id.iloc[q]),true=float(b.query_y[i,q]),pred=float(out['mean'][i,q])))
   saved=pd.read_parquet(path/'predictions.parquet');saved=saved[(saved.protocol=='first_k')&(saved.k==2)].set_index('observation_id');c=pd.DataFrame(check).set_index('observation_id');assert c.index.is_unique and set(c.index)==set(saved.index);assert np.array_equal(c.true,saved.loc[c.index,'y_true_lnc']);delta=float(np.max(np.abs(c.pred-saved.loc[c.index,'y_pred_lnc'])));assert delta<1e-8
   parity.append(dict(model=mode,seed=seed,max_difference=delta,queries=len(c)))
   x=StandardScaler().fit(hs['train']).transform(hs['test']);d=pdist(x);distances[(mode,seed)]=d
   labels=np.array([r['drug_id'] for r in rows]);same=(labels[:,None]==labels[None,:])[np.triu_indices(len(labels),1)]
   geometry.append(dict(model=mode,seed=seed,within_drug_mean_distance=float(d[same].mean()),between_drug_mean_distance=float(d[~same].mean()),between_within_ratio=float(d[~same].mean()/d[same].mean())))
   provenance.append(dict(model=mode,seed=seed,checkpoint=str((path/'selected.pt').relative_to(R)),checkpoint_sha256=sha(path/'selected.pt'),prediction_sha256=sha(path/'predictions.parquet')))
   if seed==1:
    xy=MDS(n_components=2,n_init=8,max_iter=1000,random_state=17,dissimilarity='euclidean').fit_transform(x);dd=pdist(xy);stress=float(np.sqrt(((d-dd)**2).sum()/(d*d).sum()))
    for r,(xx,yy) in zip(rows,xy):coords.append({k:v for k,v in r.items() if k!='vector'}|dict(x=xx,y=yy,stress=stress,distance_rho=float(spearmanr(d,dd).statistic)))
   print('EXTRACT',mode,seed,flush=True)
 # Every reported accuracy value is reaggregated from canonical query predictions.
 for track,folds in [('Known-drug',['known']),('Joint holdout',[f'loso_{i:02d}' for i in range(9)])]:
  for mode in N:
   for seed in range(1,11):
    frames=[]
    for fold in folds:
     path=parent(fold,seed,mode);d=pd.read_parquet(path/'predictions.parquet');d=d[(d.protocol=='first_k')&(d.k==2)].copy();d['fold']=fold;frames.append(d)
     provenance.append(dict(track=track,model=mode,seed=seed,fold=fold,prediction_path=str((path/'predictions.parquet').relative_to(R)),prediction_sha256=sha(path/'predictions.parquet')))
    d=pd.concat(frames);assert not d.duplicated(['fold','observation_id']).any()
    d['se']=(d.y_pred_lnc-d.y_true_lnc)**2;d['ae']=(d.y_pred_lnc-d.y_true_lnc).abs();d['twofold']=(d.ae<=np.log(2))*100.
    p=d.groupby(['source_id','drug_id','participant_id'])[['se','ae','twofold']].mean().reset_index()
    for (source,drug),g in p.groupby(['source_id','drug_id']):scores.append(dict(track=track,model=mode,seed=seed,source_id=source,drug_id=drug,log_rmse=np.sqrt(g.se.mean()),log_mae=g.ae.mean(),twofold=g.twofold.mean(),participants=len(g)))
 s=pd.DataFrame(scores);g=pd.DataFrame(geometry);c=pd.DataFrame(coords)
 for row in geometry:row['distance_rho_to_no_molecule']=float(spearmanr(distances[(row['model'],row['seed'])],distances[('no_molecule',row['seed'])]).statistic)
 pd.DataFrame(geometry).to_csv(O/'geometry_by_seed.csv',index=False);pd.DataFrame(features).to_parquet(O/'features_LOCAL_ONLY.parquet',index=False);c.to_csv(O/'coordinates_LOCAL_ONLY.csv',index=False);pd.DataFrame(parity).to_csv(O/'prediction_parity.csv',index=False);s.to_csv(O/'drug_metrics_by_seed.csv',index=False)
 summary=s.groupby(['track','model','drug_id']).agg(log_rmse=('log_rmse','mean'),log_rmse_sd=('log_rmse','std'),log_mae=('log_mae','mean'),twofold=('twofold','mean'),seeds=('seed','nunique')).reset_index();summary.to_csv(O/'drug_metrics_summary.csv',index=False)
 keys=['track','seed','source_id','drug_id'];a=s[s.model=='molecular'].set_index(keys);b=s[s.model=='no_molecule'].set_index(keys);assert a.index.equals(b.index)
 effects=a[['log_rmse','log_mae','twofold']]-b[['log_rmse','log_mae','twofold']];effects=effects.reset_index();effects.to_csv(O/'molecular_effect_by_seed.csv',index=False)
 e=effects.groupby(['track','drug_id']).agg(delta_log_rmse=('log_rmse','mean'),sd=('log_rmse','std'),delta_log_mae=('log_mae','mean'),delta_twofold=('twofold','mean')).reset_index();e.to_csv(O/'molecular_effect_summary.csv',index=False)
 plot_results()
 counts={track:int((t.delta_log_rmse<0).sum()) for track,t in e.groupby('track')}
 (O/'manifest.json').write_text(json.dumps(dict(status='completed',new_training=False,reused_retrained_controls=True,k=2,seeds=list(range(1,11)),geometry_track='Known-drug',geometry_seeds=10,plot_seed=1,models=N,improved_drugs_molecular_vs_none=counts,parity_max=max(x['max_difference'] for x in parity),inputs=provenance,source_sha256=sha(Path(__file__))),indent=2))
 (O/'REPORT_KO.md').write_text('# 분자 조건부 표현과 예측 검증\n\n동일 split·k=2·seed 1–10의 재학습 체크포인트를 사용했다. 새 학습 대신 30개 모델의 hidden state를 재추론하고 저장된 예측과 일치를 검증했다. 분자 입력의 효과는 molecular/no_molecule (둘 다 관계 손실 OFF)로 분리한다. Full은 별도 참고다.\n\n분자 입력으로 평균 drug log-RMSE가 낮아진 약물 수: '+json.dumps(counts,ensure_ascii=False)+' (각 11개). 약물별 수치는 molecular_effect_summary.csv 참조.\n\nFigure 4: 점은 같은 test profile이다. 각 모델의 32차원 observer state를 train 통계로 표준화 후 독립 MDS한다. 거리 왜곡(stress)은 0이면 거리의 완전 보존이며, 0.14는 14% 예측 오차가 아니다. 모델간 축은 비교할 수 없다. 색 구분은 입력에 drug 정보가 있음을 설명할 수 있으나 우수성을 증명하지 않는다. No-molecule도 투약·농도 history로 약물이 구분될 수 있다. 기하 분석은 원래 거리에서 하고 실제 효용은 별도 paired 예측 오차로 판단한다.\n\nFigure 6: 셀은 개별학습 RMSE−통합학습 RMSE. 양수는 다른 약물의 학습자료 추가가 유리, 음수는 불리. Full은 약물별로 인원이 다르며 N/A는 해당 크기의 축소 subset이 없다. 관계 OFF끼리 비교한 왼쪽은 pooling의 효과, 오른쪽은 pooling+관계 학습의 결합 효과다. 색 범위가 패널별로 다르므로 수치를 함께 읽는다. 기존 전체 learning curves는 보충자료에 보존한다.\n\n관측·투약·source의 교란과 독립적인 chemical transfer는 이 시각화만으로 증명되지 않는다. 이 분석은 기존 내부 split의 사후 진단이며 진행 중인 CV 결과와 섞지 않았다.\n')
 print('COMPLETE',counts,flush=True)
if __name__=='__main__':
 if '--plot-only' in sys.argv:plot_results()
 else:main()
