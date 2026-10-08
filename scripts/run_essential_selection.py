"""Essential E01: exact training replay, five seeds, paired validation selection only."""
from pathlib import Path
import sys,json,time,hashlib
import numpy as np,pandas as pd,torch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from mhap_pk.data.traceable import load_profiles,indices_for_split
from mhap_pk.data import representations,make_batch,score
from mhap_pk.baselines.low_data import build
from mhap_pk.relational import build_relations,auxiliary_loss
from mhap_pk.training.run_io import inputs
from mhap_pk.evaluation.quality_metrics import extended_metrics
B=R/'results/runs/ESSENTIAL_20261002'
def save(p,x):p.write_text(json.dumps(x,indent=2,default=str))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def records(y,b,ps,seed,epoch):
 rows=[]
 for row,idx in enumerate(b.indices):
  p=ps[idx]
  for j in torch.where(b.query_mask[row])[0].tolist():rows.append(dict(condition="molecular_relation",seed=seed,epoch=epoch,**{k:p[k] for k in ['source_id','drug_id','participant_id','profile_id']},observation_id=str(p['query'].observation_id.iloc[j]),query_time=float(b.query_t[row,j]),y_true_log=float(b.query_y[row,j]),y_pred_log=float(y[row,j])))
 return pd.DataFrame(rows)
def run(seed):
 torch.set_num_threads(1);out=B/f'E01__s{seed}';out.mkdir(exist_ok=False);(out/'candidates').mkdir();start=time.time();save(out/'manifest.json',dict(status='running',seed=seed))
 parent=R/f'results/runs/TRACEABLE/evaluation_20260925/known__s{seed}__molecular_relation';cfg=json.loads((parent/'config.resolved.json').read_text())
 ps,meta,_=load_profiles();tr,va,te=indices_for_split(meta,'known');z,ids,sc=representations(meta,tr,'unimol',17);oldart=json.loads((parent/'artifacts.json').read_text());assert oldart['train']==tr.tolist() and oldart['validation']==va.tolist() and oldart['test']==te.tolist()
 cal=json.loads((parent/'population.json').read_text());assert cal['fitting_indices']==tr.tolist();pop=torch.tensor(cal['log_parameters'],dtype=torch.double)
 torch.manual_seed(seed);m=build('molecular',len(sc['fitted_drugs']),pop);b=make_batch(ps,meta,tr,z,ids);v=make_batch(ps,meta,va,z,ids);rel=build_relations(ps,meta,tr,z,ids)
 rng=np.random.default_rng(1000+seed);opt=torch.optim.AdamW(m.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);traces=[];val=[];source_scores=[];chosen={};best={'V0':float('inf'),'V1':float('inf')};s5=meta[meta.drug_id=='fluorouracil'].source_id.unique();assert len(s5)==1;s5=s5[0]
 for epoch in range(cfg['epochs']+1):
  if epoch:
   m.train();opt.zero_grad(set_to_none=True);ks=torch.tensor(rng.integers(0,4,len(b.indices)));o=m(*b.inputs(ks));w=b.weights.sum(1);loss=((o['mean']-b.query_y).square()*b.weights).sum();_,_,dev=m.prior(b.z,b.ids);loss=loss+.001*(o['regularization']*w).sum()+.0001*(dev.square().mean(1)*w).sum();aux,pa,ca=auxiliary_loss(m.prior,rel,.1,.03);loss=loss+aux;assert torch.isfinite(loss);loss.backward();gn=torch.nn.utils.clip_grad_norm_(m.parameters(),5,error_if_nonfinite=True);opt.step();traces.append(dict(epoch=epoch,training_loss=float(loss.detach()),gradient_norm=float(gn)))
  if epoch%cfg['validate_every']==0:
   m.eval()
   with torch.no_grad():y=m(*inputs(v,2))['mean']
   d=records(y,v,ps,seed,epoch);d.to_parquet(out/'candidates'/f'validation_{epoch:03d}.parquet',index=False);ck=out/'candidates'/f'epoch_{epoch:03d}.pt';torch.save(m.state_dict(),ck)
   scores={'V0':extended_metrics(d)['sb_hrmse'],'V1':extended_metrics(d[d.source_id!=s5])['sb_hrmse']};assert abs(scores['V0']-score(y,v,meta))<1e-12
   for source,g in d.groupby('source_id'):source_scores.append(dict(epoch=epoch,source_id=source,queries=len(g),score=extended_metrics(g)['sb_hrmse']))
   for rule,s in scores.items():
    val.append(dict(epoch=epoch,rule=rule,validation_sb_hrmse=s))
    if s<best[rule]:best[rule]=s;chosen[rule]=dict(epoch=epoch,checkpoint=str(ck.relative_to(R)),checkpoint_sha256=sha(ck),validation_sb_hrmse=s)
 pd.DataFrame(traces).to_csv(out/'training.csv',index=False);pd.DataFrame(val).to_csv(out/'validation.csv',index=False);pd.DataFrame(source_scores).to_csv(out/'validation_sources.csv',index=False);save(out/'selection.json',chosen)
 # All checkpoint selection is finished before assessment construction.
 test=make_batch(ps,meta,te,z,ids);rows=[];metrics=[]
 for rule,c in chosen.items():
  m.load_state_dict(torch.load(R/c['checkpoint'],weights_only=True));m.eval()
  with torch.no_grad():d=records(m(*inputs(test,2))['mean'],test,ps,seed,c['epoch'])
  d['selection_rule']=rule;rows.append(d)
  for scope,g in [('all',d),('without_S5',d[d.source_id!=s5])]:metrics.append(dict(seed=seed,rule=rule,assessment=scope,selected_epoch=c['epoch'],**extended_metrics(g)))
 pd.concat(rows).to_parquet(out/'predictions.parquet',index=False);pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
 a=torch.load(R/chosen['V0']['checkpoint'],weights_only=True);old=torch.load(parent/'selected.pt',weights_only=True);delta=max(float((a[k]-old[k]).abs().max()) for k in old)
 ov=pd.read_csv(parent/'validation.csv');nv=pd.DataFrame(val).query("rule=='V0'");vdelta=float(abs(ov.validation_sb_hrmse.to_numpy()-nv.validation_sb_hrmse.to_numpy()).max());sel=json.loads((parent/'selection.json').read_text())
 save(out/'replay_verification.json',dict(checkpoint_max_difference=delta,validation_max_difference=vdelta,original_epoch=sel['epoch'],V0_epoch=chosen['V0']['epoch'],exact_replay=delta<1e-12 and vdelta<1e-12))
 save(out/'manifest.json',dict(status='completed',seed=seed,training_replay=True,reason='Original candidate epochs not retained; final selected checkpoint alone cannot be reselected',parent_checkpoint_sha256=sha(parent/'selected.pt'),config=cfg,source_sha256=sha(Path(__file__)),elapsed_seconds=time.time()-start))
 print('DONE',seed,chosen,'parity',delta,vdelta,flush=True)
if __name__=='__main__':run(int(sys.argv[1]))
