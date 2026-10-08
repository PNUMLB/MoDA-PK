"""Frozen-checkpoint support-only scale/rate calibration; exploratory validation selection."""
from pathlib import Path
import sys,json,hashlib,time,traceback
import numpy as np,pandas as pd,torch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from mhap_pk.data.traceable import load_profiles,indices_for_split
from mhap_pk.data import representations,make_batch,score
from mhap_pk.evaluation.history_protocols import initial_observation_inputs
from mhap_pk.evaluation.drug_metrics import balanced_metrics
from mhap_pk.relational_model import RelationalPK
from mhap_pk.relational import floor_curve
B=R/'results/runs/SCALE_SHAPE/20261002'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,default=str))
TIMES=[-.7,-.3,0.,.3,.7];LAM=[.1,1.,10.]
def cache(m,args,prior=False):
 z,ev,st,sy,sm,qt,ids=args
 mu,ctx,dev=m.prior(z,ids);results=[]
 for a in ([0.] if prior else TIMES):
  p=mu+mu.new_tensor([a,a,0,a,0]);pv=(p,ctx,dev)
  if prior:
   pred=floor_curve(p,ev,st);q=floor_curve(p,ev,qt)
   # Collapse repeated timestamps to one mean-log residual.
   valid=sm.clone();res=torch.zeros_like(sy)
   for row in range(len(z)):
    for t in torch.unique(st[row,sm[row]]):
     ix=torch.where(sm[row]&(st[row]==t))[0];valid[row,ix[1:]]=False
     res[row,ix[0]]=sy[row,ix].mean()-pred[row,ix[0]]
  else:
   o=m(*args,prior_values=pv);q=o['mean'];rs=[];ms=[]
   for tr in o['state']['trace']:
    mask=sm&(st==tr['time'][:,None]);obs=torch.where(mask,sy,0.).sum(1)/mask.sum(1).clamp_min(1)
    rs.append(obs-tr['log_prediction']);ms.append(tr['active'])
   res=torch.stack(rs,1);valid=torch.stack(ms,1)
  res=torch.where(valid,res,0.);results.append((a,q,res,valid))
 return results

def predict(c,lam,tau=0.):
 candidates=[];obj=[]
 for a,q,r,mask in c:
  if tau==0 and a!=0:continue
  n=mask.sum(1);b=r.sum(1)/(n+lam);b=b.clamp(-8.,8.)
  loss=torch.where(mask,(r-b[:,None]).square(),0.).sum(1)+lam*b.square()
  if tau:loss=loss+(a/tau)**2
  candidates.append((q+b[:,None],b,a));obj.append(loss)
 choice=torch.stack(obj,1).argmin(1);ix=torch.arange(len(choice))
 y=torch.stack([x[0] for x in candidates],1)[ix,choice]
 b=torch.stack([x[1] for x in candidates],1)[ix,choice]
 a=y.new_tensor([x[2] for x in candidates])[choice]
 return y,b,a

def run(fold,seed):
 torch.set_num_threads(1);out=B/f'{fold}__s{seed}';out.mkdir(exist_ok=False);start=time.time()
 save(out/'run_manifest.json',dict(status='running',fold=fold,seed=seed))
 ps,meta,_=load_profiles();tr,va,te=indices_for_split(meta,fold);z,ids,sc=representations(meta,tr,'unimol',17)
 base=R/('results/runs/TRACEABLE/evaluation_20260925' if seed<=5 else 'results/runs/TEN_SEEDS/20261001/panel');parent=base/f'{fold}__s{seed}__molecular_relation'
 art=json.loads((parent/'artifacts.json').read_text());assert art['train']==tr.tolist()
 m=RelationalPK().double();m.load_state_dict(torch.load(parent/'selected.pt',weights_only=True));m.eval()
 v=make_batch(ps,meta,va,z,ids);vb=initial_observation_inputs(v,2)
 with torch.no_grad():
  vc=cache(m,vb);vp=cache(m,vb,True)
  selected={};rows=[]
  for name,c,grid in [('prior_scale',vp,[(l,0.) for l in LAM]),('mode_scale',vc,[(l,0.) for l in LAM]),('mode_scale_shape',vc,[(l,t) for l in LAM for t in [.3,1.]])]:
   trial=[]
   for l,t in grid:
    y,_,_=predict(c,l,t);s=score(y,v,meta);trial.append((s,l,t));rows.append(dict(condition=name,lam=l,tau=t,validation_sb_hrmse=s))
   s,l,t=min(trial);selected[name]=dict(lam=l,tau=t,validation_sb_hrmse=s)
  save(out/'selection.json',selected);pd.DataFrame(rows).to_csv(out/'validation.csv',index=False)
  # Test batch is only built after all hyperparameters have been selected.
  batch=make_batch(ps,meta,te,z,ids);args=initial_observation_inputs(batch,2);tc=cache(m,args);tp=cache(m,args,True)
  zero=torch.zeros(len(te),dtype=torch.double)
  preds={'mode_original':(tc[2][1],zero,zero),'prior_only':(tp[0][1],zero,zero)}
  for name,config in selected.items():preds[name]=predict(tp if name=='prior_scale' else tc,config['lam'],config['tau'])
  output=[];diagnostics=[]
  for name,(y,b,a) in preds.items():
   assert torch.isfinite(y[batch.query_mask]).all()
   for row,idx in enumerate(te):
    p=ps[idx];key={k:p[k] for k in ['source_id','drug_id','participant_id','profile_id']}
    diagnostics.append(dict(condition=name,**key,log_scale=float(b[row]),log_rate_shift=float(a[row])))
    for j in torch.where(batch.query_mask[row])[0].tolist():output.append(dict(condition=name,fold=fold,seed=seed,k=2,protocol='first_k',**key,observation_id=str(p['query'].observation_id.iloc[j]),query_time=float(batch.query_t[row,j]),y_true_log=float(batch.query_y[row,j]),y_pred_log=float(y[row,j])))
 pd.DataFrame(output).to_parquet(out/'predictions.parquet',index=False);pd.DataFrame(diagnostics).to_csv(out/'diagnostics.csv',index=False)
 save(out/'artifacts.json',dict(train=tr.tolist(),validation=va.tolist(),test=te.tolist(),scaler=sc,parent=str(parent.relative_to(R)),checkpoint_sha256=sha(parent/'selected.pt')))
 save(out/'run_manifest.json',dict(status='completed',fold=fold,seed=seed,elapsed_seconds=time.time()-start,prediction_sha256=sha(out/'predictions.parquet'),script_sha256=sha(Path(__file__))))
 print(fold,seed,'completed',round(time.time()-start,1),flush=True)

def summary():
 d=pd.concat([pd.read_parquet(p) for p in B.glob('*/predictions.parquet')]);rows=[]
 d['track']=np.where(d.fold=='known','Known-drug','Joint drug-source')
 for key,g in d.groupby(['track','condition','seed','source_id','drug_id']):
  rows.append(dict(zip(['track','condition','seed','source_id','drug_id'],key))|balanced_metrics(g.y_true_log,g.y_pred_log,g.participant_id))
 f=pd.DataFrame(rows);f.to_csv(B/'drug_metrics_by_seed.csv',index=False)
 f.groupby(['track','condition','drug_id']).agg(log_rmse=('log_rmse','mean'),sd=('log_rmse','std'),bias=('signed_log_bias','mean'),log_mae=('log_mae','mean'),twofold=('within_twofold','mean'),seeds=('seed','nunique')).to_csv(B/'drug_summary.csv')
 s=f.groupby(['track','condition','seed','source_id']).log_rmse.mean().groupby(['track','condition','seed']).mean().reset_index(name='sb_hrmse');s.to_csv(B/'metrics_by_seed.csv',index=False);s.groupby(['track','condition']).sb_hrmse.agg(['mean','std','count']).to_csv(B/'summary.csv')
 d.to_parquet(B/'predictions.parquet',index=False)
if __name__=='__main__':
 if sys.argv[1]=='summary':summary()
 else:
  try:run(sys.argv[1],int(sys.argv[2]))
  except Exception:
   (B/f'{sys.argv[1]}__s{sys.argv[2]}'/'error.txt').write_text(traceback.format_exc());raise
