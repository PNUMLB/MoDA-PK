"""One auditable matched low-data fit. Existing files are never overwritten."""
from pathlib import Path
import sys,json,time,argparse
import numpy as np,pandas as pd,torch
R=Path(__file__).resolve().parents[3]
from mhap_pk.data import representations,make_batch,score,KS
from mhap_pk.data.traceable import load_profiles,indices_for_split
from mhap_pk.data.frozen_fingerprints import representations as fingerprints
from mhap_pk.data.low_data import subset
from mhap_pk.baselines.low_data import build,HEADS
from mhap_pk.training.population import fit_population
from mhap_pk.relational import build_relations,auxiliary_loss
from mhap_pk.training.run_io import inputs,save_json,digest
from mhap_pk.evaluation.history_protocols import common_origin,initial_observation_inputs
B=R/'results/runs/TRACEABLE/evaluation_20260925'

def run(fold,seed,mode,output_root=None,model_builder=None,target_drug=None,partition_spec=None,training_augmentation=None):
 B=Path(output_root) if output_root is not None else R/'results/runs/TRACEABLE/evaluation_20260925'
 builder=model_builder if model_builder is not None else build
 case=fold
 base_mode={'molecular_relation':'molecular','offset_relation':'molecular_offset','ecfp':'molecular','chemical_only':'molecular','pk_only':'molecular'}.get(mode,mode)
 torch.set_num_threads(1);start=time.time();out=B/f'{case}__s{seed}__{mode}';out.mkdir(exist_ok=False);save_json(out/'run_manifest.json',dict(status='running',case=case,seed=seed,condition=mode))
 try:
  cfg=json.loads((B/'plan.json').read_text());ps,meta,_=load_profiles();tr,va,te=indices_for_split(meta,fold)
  if partition_spec is not None:
   from mhap_pk.data.fitting_partitions import validate_partition
   if fold!='known' or target_drug is not None:raise ValueError('Explicit partitions require Known and no target filter')
   tr,va,te=validate_partition(meta,(tr,va,te),partition_spec)
   cfg=cfg|dict(partition_spec=partition_spec,fit_scope=partition_spec['fit_scope'])
  if target_drug is not None:
   if fold!='known':raise ValueError('Single-drug fitting requires the fixed Known-drug partition')
   tr,va,te=tuple(ix[meta.iloc[ix].drug_id.to_numpy()==target_drug] for ix in (tr,va,te))
   if not all(len(ix) for ix in (tr,va,te)):raise ValueError('Each single-drug partition must be nonempty')
   groups=[set(map(tuple,meta.iloc[ix][['source_id','participant_id']].to_numpy())) for ix in (tr,va,te)]
   assert not (groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
   cfg=cfg|dict(target_drug=target_drug,fit_scope='single_drug',partition_rule='Filter existing Known-drug roles without resplitting')
  z,ids,sc=fingerprints(meta,tr) if mode=='ecfp' else representations(meta,tr,'unimol',17);n=len(sc['fitted_drugs']);counts=meta.iloc[tr].groupby('drug_id').participant_id.nunique();count=torch.tensor([counts[d] for d in sc['fitted_drugs']],dtype=torch.double)
  if fold!='known':assert set(meta.iloc[tr].drug_id).isdisjoint(meta.iloc[te].drug_id)
  art=dict(train=tr.tolist(),validation=va.tolist(),test=te.tolist(),scaler=sc,drug_participants=counts.to_dict());save_json(out/'artifacts.json',art);save_json(out/'config.resolved.json',cfg|dict(fold=fold,seed=seed,condition=mode))
  calpath=B/'calibration'/fold/'population.json';cal=json.loads(calpath.read_text());assert cal['fitting_indices']==tr.tolist();pop=torch.tensor(cal['log_parameters'],dtype=torch.double);save_json(out/'population.json',cal|dict(parent_sha256=digest(calpath)))
  torch.manual_seed(seed);m=builder(base_mode,n,pop);vectors=torch.zeros_like(z) if mode in ['no_molecule','drug_id'] else z;b=make_batch(ps,meta,tr,vectors,ids);v=make_batch(ps,meta,va,vectors,ids)
  synthetic_batch=training_augmentation(ps,meta,tr,b,pop,seed,out) if training_augmentation is not None else None
  relations=build_relations(ps,meta,tr,z,ids) if mode in ['molecular_relation','offset_relation','chemical_only','pk_only'] else None
  if relations is not None:torch.save(relations,out/'training_relations.pt')
  rng=np.random.default_rng(1000+seed);opt=torch.optim.AdamW(m.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);initial={k:x.clone() for k,x in m.state_dict().items()};best=float('inf');logs=[];validation=[];solver_failures=[]
  for epoch in range(cfg['epochs']+1):
   if epoch:
    m.train();opt.zero_grad(set_to_none=True);ks=torch.tensor(rng.integers(0,4,len(b.indices)));o=m(*b.inputs(ks));w=b.weights.sum(1);l=((o['mean']-b.query_y).square()*b.weights).sum()
    if synthetic_batch is not None:
     synthetic_output=m(*synthetic_batch.inputs(ks));l=.5*l+.5*((synthetic_output['mean']-synthetic_batch.query_y).square()*synthetic_batch.weights).sum()
    if base_mode in HEADS:
     _,_,dev=m.prior(b.z,b.ids);l=l+.001*(o['regularization']*w).sum()+.0001*(dev.square().mean(1)*w).sum()
     if base_mode=='molecular_offset':l=l+.01*((2*m.prior.offset.weight.tanh()).square()/count[:,None]).mean()
    pk_aux=chem_aux=0.
    if relations is not None:
     aux,pk_aux,chem_aux=auxiliary_loss(m.prior,relations,0. if mode=='chemical_only' else .1,0. if mode=='pk_only' else .03);l=l+aux
    if not torch.isfinite(l):raise FloatingPointError('training loss')
    l.backward();gn=torch.nn.utils.clip_grad_norm_(m.parameters(),5,error_if_nonfinite=True);opt.step();logs.append(dict(epoch=epoch,training_loss=float(l.detach()),gradient_norm=float(gn),pk_relation=float(pk_aux),chemical_relation=float(chem_aux)))
   if epoch%cfg['validate_every']==0:
    m.eval()
    with torch.no_grad():val=score(m(*inputs(v,2))['mean'],v,meta)
    validation.append(dict(epoch=epoch,validation_sb_hrmse=val))
    if val<best:best=val;chosen=epoch;state={k:x.clone() for k,x in m.state_dict().items()};torch.save(state,out/'selected.pt')
    if epoch==120 and cfg['epochs']>120:
     torch.save(state,out/'selected_at120.pt')
     save_json(out/'selection_at120.json',dict(epoch=chosen,validation_sb_hrmse=best,checkpoint_sha256=digest(out/'selected_at120.pt')))
    pd.DataFrame(logs).to_csv(out/'training.csv',index=False);pd.DataFrame(validation).to_csv(out/'validation.csv',index=False);save_json(out/'progress.json',dict(epoch=epoch,elapsed_seconds=time.time()-start))
  distance=float(sum((x-initial[k]).square().sum() for k,x in m.state_dict().items() if x.is_floating_point()));assert distance>0
  selection=dict(epoch=chosen,validation_sb_hrmse=best,parameters=sum(p.numel() for p in m.parameters()),checkpoint_sha256=digest(out/'selected.pt'),weight_change_squared=distance)
  save_json(out/'selection.json',selection);save_json(out/'fitting_complete.json',dict(status='selected_before_test',elapsed_seconds=time.time()-start));m.load_state_dict(state);m.eval();test=make_batch(ps,meta,te,vectors,ids)
  rows=[];metrics=[];latency=[];maximum=0.
  replay=builder(base_mode,n,pop);replay.load_state_dict(torch.load(out/'selected.pt',weights_only=True));replay.eval()
  for protocol,batch in [('first_k',test),('common_origin',common_origin(test))]:
   evaluation_pairs=list(enumerate(KS))
   if protocol=='first_k' and 3 in cfg.get('additional_inference_k',[]):evaluation_pairs.append((None,3))
   for ki,k in evaluation_pairs:
    t=time.perf_counter()
    args=initial_observation_inputs(batch,k) if ki is None else inputs(batch,ki)
    with torch.no_grad():y=m(*args)['mean'];yy=replay(*args)['mean']
    maximum=max(maximum,float((y[batch.query_mask]-yy[batch.query_mask]).abs().max()));assert maximum<1e-10
    latency.append(dict(protocol=protocol,k=k,seconds=time.perf_counter()-t,profiles=len(te)));assert torch.isfinite(y[batch.query_mask]).all() and torch.isfinite(y[batch.query_mask].exp()).all()
    metrics.append(dict(condition=mode,seed=seed,fold=fold,protocol=protocol,k=k,sb_hrmse=score(y,batch,meta)))
    for i,idx in enumerate(batch.indices):
     p=ps[idx];support_end=float(args[2][i,args[4][i]].max()) if k else None;cutoff=float(test.support_t[i,3][test.support_mask[i,3]].max()) if protocol=='common_origin' else support_end
     base=dict(case=fold,fold=fold,seed=seed,condition=mode,k=k,protocol=protocol,cutoff_time=cutoff,**{key:p[key] for key in ['source_id','drug_id','participant_id','profile_id']})
     for j in torch.where(batch.query_mask[i])[0].tolist():rows.append(base|dict(observation_id=str(p['query'].observation_id.iloc[j]),query_time=float(batch.query_t[i,j]),support_end_time=support_end,y_true_lnc=float(batch.query_y[i,j]),y_pred_lnc=float(y[i,j]),fallback_flag=False))
  pd.DataFrame(rows).to_parquet(out/'predictions.parquet',index=False);pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False);pd.DataFrame(latency).to_csv(out/'latency.csv',index=False)
  save_json(out/'verification.json',dict(max_prediction_difference=maximum,k_count=4,protocols=['first_k','common_origin']))
  save_json(out/'run_manifest.json',dict(status='completed',case=case,seed=seed,condition=mode,prediction_sha256=digest(out/'predictions.parquet'),elapsed_seconds=time.time()-start));print('DONE',case,seed,mode,time.time()-start,flush=True)
 except Exception as e:save_json(out/'run_manifest.json',dict(status='failed',case=case,seed=seed,condition=mode,error=repr(e),elapsed_seconds=time.time()-start));raise
