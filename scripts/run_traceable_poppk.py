"""Drug-specific VI population estimation + EBE MAP under fixed Known splits."""
from pathlib import Path
import sys,json,time,argparse
import numpy as np,pandas as pd,torch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from mhap_pk.data import representations,make_batch,score,KS
from mhap_pk.data.traceable import load_profiles,indices_for_split
from mhap_pk.evaluation.history_protocols import common_origin,initial_observation_inputs
from mhap_pk.data.low_data import subset
from mhap_pk.baselines.population_vi import PopulationVI,map_forecast
from mhap_pk.training.population import fit_population
from mhap_pk.training.run_io import inputs,save_json,digest
B=R/'results/runs/TRACEABLE_POPPK/evaluation_20260925'
def run(case,seed,output_root=None):
 B=Path(output_root) if output_root is not None else R/'results/runs/TRACEABLE_POPPK/evaluation_20260925'
 torch.set_num_threads(1);start=time.time();out=B/f'{case}__s{seed}';out.mkdir(exist_ok=False);save_json(out/'run_manifest.json',dict(status='running',case=case,seed=seed,condition='poppk_vi_map'))
 try:
  cfg=json.loads((B/'plan.json').read_text());scenario=next(c for c in cfg['cases'] if c['case']==case);ps,meta,_=load_profiles();original,va,te=indices_for_split(meta,'known');tr=original if case=='full' else subset(meta,original,scenario['drugs'],scenario['cap'],seed)[0]
  drugs=sorted(meta.drug_id.unique()) if case=='full' else scenario['drugs'];z,ids,_=representations(meta,tr,'none',17);records={};logs=[];validation=[];selection={};models={};artifacts={}
  save_json(out/'config.resolved.json',cfg|dict(case=case,seed=seed,scenario=scenario))
  for drug in drugs:
   train=np.array([i for i in tr if meta.iloc[i].drug_id==drug]);valid=np.array([i for i in va if meta.iloc[i].drug_id==drug]);test=np.array([i for i in te if meta.iloc[i].drug_id==drug]);b=make_batch(ps,meta,train,z,ids);v=make_batch(ps,meta,valid,z,ids)
   pop,cal=fit_population(b);people=sorted(set((meta.iloc[i].source_id,meta.iloc[i].participant_id) for i in train));person=torch.tensor([people.index((meta.iloc[i].source_id,meta.iloc[i].participant_id)) for i in train]);artifacts[drug]=dict(train=train.tolist(),validation=valid.tolist(),test=test.tolist(),people=people,calibration=cal)
   times=torch.cat([b.support_t[:,3],b.query_t],1);labels=torch.cat([b.support_y[:,3],b.query_y],1);mask=torch.cat([b.support_mask[:,3],b.query_mask],1)
   torch.manual_seed(seed);m=PopulationVI(pop,len(people));opt=torch.optim.Adam(m.parameters(),lr=cfg['population_learning_rate']);best=float('inf');generator=torch.Generator().manual_seed(17000+seed)
   for epoch in range(cfg['epochs']+1):
    if epoch:
     eps=torch.randn((2,len(people),2),generator=generator,dtype=torch.double);eps=torch.cat([eps,-eps]);opt.zero_grad();loss=m.objective(b.events,times,labels,mask,person,eps)
     if not torch.isfinite(loss):raise FloatingPointError('ELBO')
     loss.backward();gn=torch.nn.utils.clip_grad_norm_(m.parameters(),10,error_if_nonfinite=True);opt.step();m.constrain();logs.append(dict(drug=drug,epoch=epoch,negative_elbo=float(loss.detach()),gradient_norm=float(gn)))
    if epoch%cfg['validate_every']==0:
     args=inputs(v,2);pred,diag=map_forecast(m,*args[1:6],steps=cfg['map_steps']);val=score(pred,v,meta);validation.append(dict(drug=drug,epoch=epoch,validation_sb_hrmse=val))
     if epoch>0 and val<best:best=val;chosen=epoch;state={key:x.clone() for key,x in m.state_dict().items()}
     save_json(out/'progress.json',dict(drug=drug,epoch=epoch,elapsed_seconds=time.time()-start))
   m.load_state_dict(state);models[drug]=m;torch.save(state,out/f'{drug}_selected.pt');selection[drug]=dict(epoch=chosen,validation_sb_hrmse=best,mu=m.mu.detach().tolist(),omega=m.log_omega.detach().exp().tolist(),sigma=float(m.log_sigma.detach().exp()),participants=len(people),checkpoint_sha256=digest(out/f'{drug}_selected.pt'))
  save_json(out/'artifacts.json',artifacts);save_json(out/'selection.json',selection);pd.DataFrame(logs).to_csv(out/'training.csv',index=False);pd.DataFrame(validation).to_csv(out/'validation.csv',index=False);save_json(out/'fitting_complete.json',dict(status='all_drugs_selected_before_test',elapsed_seconds=time.time()-start))
  rows=[];diagnostics=[];latency=[];verification=[]
  for drug,m in models.items():
   art=artifacts[drug];test=make_batch(ps,meta,np.array(art['test']),z,ids);replay=PopulationVI(art['calibration']['log_parameters'],len(art['people']));replay.load_state_dict(torch.load(out/f'{drug}_selected.pt',weights_only=True))
   first_test=test
   for protocol,test in [('first_k',first_test),('common_origin',common_origin(first_test))]:
    evaluation_pairs=list(enumerate(KS))
    if protocol=='first_k' and 3 in cfg.get('additional_inference_k',[]):evaluation_pairs.append((None,3))
    for ki,k in evaluation_pairs:
     args=initial_observation_inputs(test,k) if ki is None else inputs(test,ki);t=time.perf_counter();y,diag=map_forecast(m,*args[1:6],steps=cfg['map_steps']);latency.append(dict(drug=drug,k=k,seconds=time.perf_counter()-t,profiles=len(test.indices)))
     yy,_=map_forecast(replay,*args[1:6],steps=cfg['map_steps']);diff=float((y[test.query_mask]-yy[test.query_mask]).abs().max());assert diff<1e-10;verification.append(dict(drug=drug,k=k,max_prediction_difference=diff))
     assert torch.isfinite(y[test.query_mask]).all() and torch.isfinite(y[test.query_mask].exp()).all()
     for i,idx in enumerate(test.indices):
      p=ps[idx];identity={key:p[key] for key in ['source_id','drug_id','participant_id','profile_id']};base=dict(dataset_id=cfg['dataset_id'],split_id=cfg['split_id'],fold='known',protocol=protocol,case=case,seed=seed,condition='poppk_vi_map',k=k,is_target=True,fit_participants_for_drug=len(art['people']),**identity)
      diagnostics.append(base|dict(eta=diag['eta'][i].tolist(),support_objective=float(diag['objective'][i]),fallback_flag=bool(diag['fallback'][i])))
      for j in torch.where(test.query_mask[i])[0].tolist():rows.append(base|dict(observation_id=str(p['query'].observation_id.iloc[j]),query_time=float(test.query_t[i,j]),support_end_time=float(args[2][i,args[4][i]].max()) if k else None,y_true_lnc=float(test.query_y[i,j]),y_pred_lnc=float(y[i,j]),fallback_flag=bool(diag['fallback'][i])))
  pd.DataFrame(rows).to_parquet(out/'predictions.parquet',index=False);pd.DataFrame(diagnostics).to_parquet(out/'diagnostics.parquet',index=False);pd.DataFrame(latency).to_csv(out/'latency.csv',index=False);pd.DataFrame(verification).to_csv(out/'verification.csv',index=False)
  save_json(out/'run_manifest.json',dict(status='completed',case=case,seed=seed,condition='poppk_vi_map',prediction_sha256=digest(out/'predictions.parquet'),elapsed_seconds=time.time()-start,drug_fits=len(models)));print('DONE',case,seed,time.time()-start,flush=True)
 except Exception as e:save_json(out/'run_manifest.json',dict(status='failed',case=case,seed=seed,error=repr(e),elapsed_seconds=time.time()-start));raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args();run(a.case,a.seed)
