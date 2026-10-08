"""Aligned end-to-end head ablation; preserves the TRACEABLE training protocol."""
from pathlib import Path
import sys,json,time
import numpy as np,pandas as pd,torch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from mhap_pk.data import representations,make_batch,score,KS
from mhap_pk.data.traceable import load_profiles,indices_for_split
from mhap_pk.baselines.low_data import build
from mhap_pk.relational import build_relations,auxiliary_loss
from mhap_pk.training.run_io import inputs,save_json,digest
from mhap_pk.evaluation.history_protocols import common_origin,initial_observation_inputs
OLD=R/'results/runs/TRACEABLE/evaluation_20260925'
def model(mode,n,pop):
 m=build('molecular',n,pop)
 if mode=='base':
  m.prior_only=True
  for module in [m.gru,m.parameter_head,m.gain_head]:
   for p in module.parameters():p.requires_grad_(False)
 elif mode=='parameter':
  m.state_update=False
  for p in m.gain_head.parameters():p.requires_grad_(False)
 elif mode=='state':
  m.parameter_update=False
  for p in m.parameter_head.parameters():p.requires_grad_(False)
 elif mode!='full':raise ValueError(mode)
 return m

def run(root,fold,seed,mode):
 torch.set_num_threads(1);out=root/f'{fold}__s{seed}__{mode}';out.mkdir(exist_ok=False);start=time.time();save_json(out/'run_manifest.json',dict(status='running'))
 try:
  cfg=json.loads(((root/'plan.json') if (root/'plan.json').exists() else (OLD/'plan.json')).read_text());ps,meta,_=load_profiles();tr,va,te=indices_for_split(meta,fold);z,ids,sc=representations(meta,tr,'unimol',17);n=len(sc['fitted_drugs']);calpath=OLD/'calibration'/fold/'population.json';cal=json.loads(calpath.read_text());assert cal['fitting_indices']==tr.tolist();pop=torch.tensor(cal['log_parameters'],dtype=torch.double)
  save_json(out/'artifacts.json',dict(train=tr.tolist(),validation=va.tolist(),test=te.tolist(),scaler=sc));save_json(out/'config.resolved.json',cfg|dict(condition=mode,fold=fold,seed=seed,relation_weights=[.1,.03],prior_frozen=False));save_json(out/'population.json',cal|dict(parent_sha256=digest(calpath)))
  torch.manual_seed(seed);m=model(mode,n,pop);b=make_batch(ps,meta,tr,z,ids);v=make_batch(ps,meta,va,z,ids);relations=build_relations(ps,meta,tr,z,ids);torch.save(relations,out/'training_relations.pt');rng=np.random.default_rng(1000+seed);opt=torch.optim.AdamW(m.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);best=float('inf');logs=[];vals=[]
  for epoch in range(cfg['epochs']+1):
   if epoch:
    m.train();opt.zero_grad(set_to_none=True);ks=torch.tensor(rng.integers(0,4,len(b.indices)));o=m(*b.inputs(ks));w=b.weights.sum(1);loss=((o['mean']-b.query_y).square()*b.weights).sum();_,_,dev=m.prior(b.z,b.ids);loss=loss+.001*(o['regularization']*w).sum()+.0001*(dev.square().mean(1)*w).sum();aux,pk,chem=auxiliary_loss(m.prior,relations,.1,.03);loss=loss+aux;assert torch.isfinite(loss);loss.backward();gn=torch.nn.utils.clip_grad_norm_(m.parameters(),5,error_if_nonfinite=True);opt.step();logs.append(dict(epoch=epoch,loss=float(loss.detach()),gradient_norm=float(gn),pk_relation=float(pk),chemical_relation=float(chem)))
   if epoch%cfg['validate_every']==0:
    m.eval()
    with torch.no_grad():val=score(m(*inputs(v,2))['mean'],v,meta)
    vals.append(dict(epoch=epoch,validation_sb_hrmse=val))
    if val<best:best=val;chosen=epoch;state={k:x.detach().clone() for k,x in m.state_dict().items()}
    save_json(out/'progress.json',dict(epoch=epoch,seconds=time.time()-start))
  m.load_state_dict(state);m.eval();torch.save(state,out/'selected.pt');pd.DataFrame(logs).to_csv(out/'training.csv',index=False);pd.DataFrame(vals).to_csv(out/'validation.csv',index=False);save_json(out/'selection.json',dict(epoch=chosen,validation_sb_hrmse=best,checkpoint_sha256=digest(out/'selected.pt'),trainable_parameters=sum(p.numel() for p in m.parameters() if p.requires_grad)));save_json(out/'fitting_complete.json',dict(selected_before_test=True))
  replay=model(mode,n,pop);replay.load_state_dict(torch.load(out/'selected.pt',weights_only=True));replay.eval();test=make_batch(ps,meta,te,z,ids);rows=[];scores=[];maximum=0.
  for protocol,batch in [('first_k',test),('common_origin',common_origin(test))]:
   evaluation_pairs=list(enumerate(KS))
   if protocol=='first_k' and 3 in cfg.get('additional_inference_k',[]):evaluation_pairs.append((None,3))
   for ki,k in evaluation_pairs:
    args=initial_observation_inputs(batch,k) if ki is None else inputs(batch,ki)
    with torch.no_grad():y=m(*args)['mean'];yy=replay(*args)['mean']
    assert torch.isfinite(y[batch.query_mask]).all();maximum=max(maximum,float((y-yy).abs().max()));assert maximum<1e-10;scores.append(dict(fold=fold,seed=seed,condition=mode,protocol=protocol,k=k,sb_hrmse=score(y,batch,meta)))
    for i,idx in enumerate(batch.indices):
     p=ps[idx];support_end=float(args[2][i,args[4][i]].max()) if k else None;cutoff=float(test.support_t[i,3][test.support_mask[i,3]].max()) if protocol=='common_origin' else support_end
     for j in torch.where(batch.query_mask[i])[0].tolist():rows.append({key:p[key] for key in ['source_id','drug_id','participant_id','profile_id']}|dict(dataset_id='source_traceable_20260925',split_id='source_traceable_20260925',fold=fold,track='Known-drug' if fold=='known' else 'Joint drug–source',seed=seed,condition=mode,protocol=protocol,k=k,cutoff_time=cutoff,support_end_time=support_end,observation_id=str(p['query'].observation_id.iloc[j]),query_time=float(batch.query_t[i,j]),y_true_log=float(batch.query_y[i,j]),y_pred_log=float(y[i,j]),fallback_flag=False))
  frame=pd.DataFrame(rows);frame.to_parquet(out/'predictions.parquet',index=False);pd.DataFrame(scores).to_csv(out/'metrics.csv',index=False)
  if mode=='full':
   old=pd.read_parquet(OLD/f'{fold}__s{seed}__molecular_relation/predictions.parquet');key=['protocol','k','observation_id'];a=frame[frame.k.isin(KS)].set_index(key).y_pred_log.sort_index();b=old.set_index(key).y_pred_lnc.sort_index();assert a.index.equals(b.index);maximum=max(maximum,float((a-b).abs().max()));assert maximum<1e-10
  save_json(out/'run_manifest.json',dict(status='completed',fold=fold,seed=seed,condition=mode,elapsed_seconds=time.time()-start,prediction_sha256=digest(out/'predictions.parquet'),max_replay_difference=maximum,full_reference_parity=(mode=='full')))
 except Exception as e:save_json(out/'run_manifest.json',dict(status='failed',error=repr(e)));raise
if __name__=='__main__':run(Path(sys.argv[1]),sys.argv[2],int(sys.argv[3]),sys.argv[4])
