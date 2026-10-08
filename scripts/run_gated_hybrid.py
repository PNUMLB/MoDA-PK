"""Version 7: fitting-only cross-fitting, frozen expert gates, exact test ODEs."""
from pathlib import Path
import argparse,json,os,sys,time,traceback,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor,as_completed
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
import torch
from pkforecast.data import load_profiles,indices_for_split,representations,make_batch,score,sha256,KS
from pkforecast.relational import floor_curve,build_relations,auxiliary_loss
from pkforecast.relational_model import RelationalPK
from pkforecast.relational_information import CalibratedInformationPK
from pkforecast.iterated_information import CONFIGS,predict as info_predict
from pkforecast.soft_hybrid import FrozenExpertSoftHybrid
from pkforecast.gated_study import crossfit_splits,predict_cached,profile_loss,loss_grid,interpolate_loss,fit_gate,FEATURE_SETS
from run_relational import dump

PLAN=ROOT/'experiments/forecast_v7_gated/PLAN.json'
RUN=ROOT/'results/forecast_v7_gated'
V3=ROOT/'results/forecast_v3/jobs'
VARIANTS={
    'unimol':dict(representation='unimol',relations=True,mapping=17,seq='relational',info='information_relational'),
    'no_relations':dict(representation='unimol',relations=False,mapping=17,seq='without_relations',info='information'),
    'none':dict(representation='none',relations=True,mapping=17,seq='relational_none',info=None),
    'id':dict(representation='id',relations=True,mapping=17,seq='relational_id',info=None),
    **{f'random{m}':dict(representation='random',relations=True,mapping=m,seq='relational_random',info=None) for m in [17,29,43]}}


def freeze():
    if PLAN.exists():raise FileExistsError('Frozen v7 plan already exists')
    profiles,meta,_=load_profiles()
    folds=['known']+[f'loso_{i:02d}' for i in range(10)]
    episodes={}
    for fold in folds:
        tr,va,te=indices_for_split(meta,fold);fitting=np.sort(np.r_[tr,va])
        episodes[fold]=crossfit_splits(meta,fitting)
        for ep in episodes[fold]:
            assert set(ep['fitting_indices']+ep['heldout_indices']).isdisjoint(te)
    manifest=PLAN.parent/'CROSSFIT_MANIFEST_LOCAL_ONLY.json';dump(manifest,episodes)
    jobs=[dict(fold=f,variant=v,seed=s,id=f'{f}__{v}__s{s}') for f in folds for v in VARIANTS for s in [1,2,3]]
    prior=json.loads((ROOT/'experiments/forecast_v6_routing/PLAN.json').read_text())
    hashes=dict(prior['hashes'])
    paths=[Path(__file__),ROOT/'src/pkforecast/soft_hybrid.py',ROOT/'src/pkforecast/gated_study.py',
        ROOT/'tests/test_soft_hybrid.py',ROOT/'tests/test_gated_study.py',manifest,PLAN.parent/'PROTOCOL_KO.md',PLAN.parent/'PREFLIGHT.json',
        ROOT/'experiments/forecast_v6_routing/PLAN.json',ROOT/'results/forecast_v6_routing/COMPLETE.json']
    for v in VARIANTS.values():
        for f in folds:
            for s in [1,2,3]:
                for c in [v['seq'],v['info']]:
                    if c is None:continue
                    folder=V3/f'{f}__{c}__s{s}__m{v["mapping"]}'
                    receipt=json.loads((folder/'COMPLETE.json').read_text())
                    assert sha256(folder/'checkpoint.pt')==receipt['checkpoint_sha256']
                    paths.extend([folder/'checkpoint.pt',folder/'COMPLETE.json',folder/'predictions_LOCAL_ONLY.csv.gz'])
    for p in paths:hashes[str(p.relative_to(ROOT))]=sha256(p)
    for p,h in hashes.items():assert sha256(ROOT/p)==h,p
    dump(PLAN,dict(version='forecast_v7_gated',jobs=jobs,variants=VARIANTS,hashes=hashes,
        crossfit_epochs=120,final_new_expert_max_epochs=240,validate_every=10,
        learning_rate=.003,weight_decay=.0001,gradient_clip=5.,personal_penalty=.001,chemical_penalty=.0001,
        population_steps=320,population_initial_volumes=[10.,100.,1000.],
        gate_steps=400,gate_learning_rate=.03,gate_l2=.01,gate_l2_sensitivity=[.001,.1],
        gate_feature_sets=FEATURE_SETS,dynamics_grid_nodes=65,information_config=CONFIGS[2],
        primary_k=2,support_k=KS,primary_candidates=['output_full','dynamics_full'],
        gate_training='2 participant and 3 source-group OOF folds inside outer refit, fixed 120 epochs per OOF expert',
        gate_loss='equal known/unseen OOF domains, each normalized by its OOF sequential hierarchical MSE; equal history levels',
        exact_deployment=True,shared_prior=False,joint_expert_training=False,
        repeated_development_data=True,new_external_validation=False,
        original_v3_final_expert_reuse=True,missing_information_controls_train_and_select_like_v3=True))
    print(f'Frozen {len(jobs)} outer jobs, {len(folds)*5} OOF fitting subsets',flush=True)


def calibrate(item):
    fold,ep=item;torch.set_num_threads(1)
    folder=RUN/'calibration';path=folder/f'{fold}__{ep["name"]}.json'
    if path.exists():
        value=json.loads(path.read_text());assert value['fitting_indices']==ep['fitting_indices']
        return path.name,'already_complete'
    profiles,meta,_=load_profiles();indices=np.array(ep['fitting_indices'])
    z,ids,_=representations(meta,indices,'unimol',17)
    batch=make_batch(profiles,meta,indices,z,ids)
    best=float('inf');parameters=None;runs=[]
    for volume in [10.,100.,1000.]:
        initial=torch.tensor([0.,0.,np.log(volume),0.,np.log(volume)],dtype=torch.float64)
        theta=initial.clone().requires_grad_(True);optimizer=torch.optim.Adam([theta],lr=.05)
        for epoch in range(321):
            optimizer.zero_grad(set_to_none=True)
            pred=floor_curve(theta[None].expand(len(indices),-1),batch.events,batch.query_t)
            mse=((pred-batch.query_y).square()*batch.weights).sum()
            loss=mse+.0001*(theta-initial).square().mean()
            if not torch.isfinite(loss):raise FloatingPointError('Invalid OOF population initialization')
            if float(mse.detach())<best:best=float(mse.detach());parameters=theta.detach().tolist()
            if epoch==320:break
            loss.backward();torch.nn.utils.clip_grad_norm_([theta],5.);optimizer.step()
            with torch.no_grad():theta.clamp_(-12.,16.)
        runs.append(dict(initial_volume=volume,final_mse=float(mse.detach())))
    dump(path,dict(fitting_indices=ep['fitting_indices'],heldout_indices=ep['heldout_indices'],
        log_parameters=parameters,best_fitting_mse=best,runs=runs,plan_sha256=sha256(PLAN)))
    return path.name,f'complete mse={best:.4f}'


def setup(profiles,meta,fitting,variant,seed,kind,base):
    torch.manual_seed(seed);v=VARIANTS[variant]
    z,ids,scaler=representations(meta,fitting,v['representation'],v['mapping'])
    count=len(scaler['fitted_drugs']) if v['representation']=='id' else 0
    model=(RelationalPK if kind=='seq' else CalibratedInformationPK)(id_count=count).double()
    with torch.no_grad():model.prior.base.copy_(torch.tensor(base,dtype=torch.float64))
    chemistry,_,_=representations(meta,fitting,'unimol',17)
    rel=build_relations(profiles,meta,fitting,chemistry,ids)
    first=[next(int(i) for i in fitting if profiles[i]['drug_id']==d) for d in rel['drugs']]
    rel['z']=z[first];model.relations=rel
    return model,make_batch(profiles,meta,fitting,z,ids),z,ids,scaler


def step(model,batch,optimizer,rng,relations):
    model.train();optimizer.zero_grad(set_to_none=True)
    ki=torch.tensor(rng.integers(0,4,len(batch.indices)))
    out=model(*batch.inputs(ki),variance=False)
    mse=((out['mean']-batch.query_y).square()*batch.weights).sum()
    pw=batch.weights.sum(1)
    personal=(out.get('regularization',torch.zeros_like(pw))*pw).sum()
    _,_,chem=model.prior(batch.z,batch.ids)
    aux,_,_=auxiliary_loss(model.prior,model.relations,.1 if relations else 0.,.03 if relations else 0.)
    loss=mse+.001*personal+.0001*(chem.square().mean(1)*pw).sum()+aux
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite expert training')
    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step()
    return float(loss.detach())


def fit_fixed(profiles,meta,fitting,job,kind,base,epochs,path):
    model,batch,z,ids,scaler=setup(profiles,meta,fitting,job['variant'],job['seed'],kind,base)
    if path.exists():
        c=torch.load(path,weights_only=False);assert c['fitting_indices']==list(map(int,fitting)) and c['epochs']==epochs
        assert c['scaler']==scaler;model.load_state_dict(c['model'])
    else:
        opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.0001);rng=np.random.default_rng(1000+job['seed'])
        loss=None
        for epoch in range(epochs):loss=step(model,batch,opt,rng,VARIANTS[job['variant']]['relations'])
        path.parent.mkdir(parents=True,exist_ok=True)
        torch.save(dict(model=model.state_dict(),scaler=scaler,fitting_indices=list(map(int,fitting)),epochs=epochs,
            last_loss=loss,plan_sha256=sha256(PLAN)),path)
    return model.eval(),z,ids,scaler


def final_experts(profiles,meta,job,folder):
    tr,va,te=indices_for_split(meta,job['fold']);fitting=np.sort(np.r_[tr,va]);v=VARIANTS[job['variant']]
    z,ids,scaler=representations(meta,fitting,v['representation'],v['mapping'])
    count=len(scaler['fitted_drugs']) if v['representation']=='id' else 0
    models=[];paths=[]
    for kind,cls in [('seq',RelationalPK),('info',CalibratedInformationPK)]:
        if v[kind] is not None:
            p=V3/f'{job["fold"]}__{v[kind]}__s{job["seed"]}__m{v["mapping"]}'/'checkpoint.pt'
            c=torch.load(p,weights_only=False);assert c['scaler']==scaler
            model=cls(id_count=count).double();model.load_state_dict(c['model']);model.eval()
        else:
            selected=folder/'NEW_INFORMATION_SELECTION.json'
            if selected.exists():chosen=json.loads(selected.read_text())['selected_epoch']
            else:
                base=json.loads((ROOT/f'experiments/forecast_v3/calibration/{job["fold"]}__train.json').read_text())['log_parameters']
                model,batch,tz,tids,_=setup(profiles,meta,tr,job['variant'],job['seed'],kind,base)
                vb=make_batch(profiles,meta,va,tz,tids)
                opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.0001);rng=np.random.default_rng(1000+job['seed'])
                best=float('inf');chosen=0;history=[]
                for epoch in range(241):
                    if epoch:step(model,batch,opt,rng,v['relations'])
                    if epoch%10==0:
                        model.eval()
                        with torch.no_grad():value=score(model(*vb.inputs(torch.full((len(va),),2)),variance=False)['mean'],vb,meta)
                        history.append(dict(epoch=epoch,validation=value))
                        if value<best:best=value;chosen=epoch
                dump(selected,dict(selected_epoch=chosen,validation=best,history=history,train=tr.tolist(),validation_indices=va.tolist()))
            base=json.loads((ROOT/f'experiments/forecast_v3/calibration/{job["fold"]}__refit.json').read_text())['log_parameters']
            p=folder/'information_refit.pt'
            model,_,_,_=fit_fixed(profiles,meta,fitting,job,kind,base,chosen,p)
        models.append(model);paths.append(str(p.relative_to(ROOT)))
    return *models,z,ids,scaler,paths


def cache_one(seq,info,batch,registry,variant,ki):
    model=FrozenExpertSoftHybrid(seq,info,registry,CONFIGS[2]).eval()
    inputs=batch.inputs(torch.full((len(batch.indices),),ki))
    with torch.no_grad():state=model.condition(*inputs[:5],ids=inputs[6])
    features=state['gate_features'].clone()
    if VARIANTS[variant]['representation']=='id':features[:,0]=(batch.ids<0).double().log1p()
    if VARIANTS[variant]['representation']=='none':features[:,0]=0.
    cache={k:state[k].detach() for k in ['sequential_parameters','sequential_state','information_parameters','information_state','time','include_start']}
    cache.update(features=features,events=batch.events,query_t=batch.query_t,query_y=batch.query_y,
        query_mask=batch.query_mask,profile_index=torch.tensor(batch.indices),k=torch.full((len(batch.indices),),KS[ki]))
    with torch.no_grad():
        # Endpoints are actual original experts, including sequential state updates.
        cache['seq_prediction']=predict_cached(cache,features.new_zeros(len(features)),'dynamics')
        cache['info_prediction']=predict_cached(cache,features.new_ones(len(features)),'dynamics')
    return cache


def combine(caches):
    out={}
    for key in caches[0]:
        values=[c[key] for c in caches]
        if key in ['query_t','query_y','query_mask','seq_prediction','info_prediction']:
            width=max(v.shape[1] for v in values)
            values=[torch.cat([v,(v[:,-1:].expand(-1,width-v.shape[1]) if key=='query_t' else v.new_zeros((len(v),width-v.shape[1])))],1) for v in values]
        elif key=='events':
            width=max(v.shape[1] for v in values)
            values=[torch.cat([v,v.new_zeros((len(v),width-v.shape[1],4))],1) for v in values]
        out[key]=torch.cat(values)
    return out


def collect_oof(profiles,meta,job,folder):
    path=folder/'oof_cache_LOCAL_ONLY.pt'
    if path.exists():return torch.load(path,weights_only=False)
    eps=json.loads((PLAN.parent/'CROSSFIT_MANIFEST_LOCAL_ONLY.json').read_text())[job['fold']]
    tr,va,_=indices_for_split(meta,job['fold']);fitting=np.sort(np.r_[tr,va])
    z,ids,_=representations(meta,fitting,'unimol',17)
    full=make_batch(profiles,meta,fitting,z,ids)
    pw=dict(zip(fitting,full.weights.sum(1).tolist()))
    caches=[];audit=[]
    for ep in eps:
        fit=np.array(ep['fitting_indices']);held=np.array(ep['heldout_indices'])
        basepath=RUN/'calibration'/f'{job["fold"]}__{ep["name"]}.json'
        base=json.loads(basepath.read_text())['log_parameters']
        models=[];checkpoints=[]
        for kind in ['seq','info']:
            p=folder/'oof'/ep['name']/f'{kind}.pt'
            model,z,ids,scaler=fit_fixed(profiles,meta,fit,job,kind,base,120,p)
            models.append(model);checkpoints.append(dict(path=str(p.relative_to(ROOT)),sha256=sha256(p)))
        registry=z[[next(i for i in fit if profiles[i]['drug_id']==d) for d in scaler['fitted_drugs']]]
        batch=make_batch(profiles,meta,held,z,ids)
        for ki in range(4):
            c=cache_one(*models,batch,registry,job['variant'],ki)
            c['weights']=torch.tensor([pw[i]/8 for i in held],dtype=torch.float64)
            c['domain']=torch.full((len(held),),int(ep['kind']=='unseen'))
            c['grid']=loss_grid(c)
            caches.append(c)
        audit.append(dict(**ep,checkpoint_hashes=checkpoints,calibration_sha256=sha256(basepath),
            scaler_fitted_drugs=scaler['fitted_drugs'],relation_query_ids=models[0].relations['contributing_query_ids']))
        dump(folder/'PROGRESS.json',dict(phase='oof',episode=ep['name'],finished_episodes=len(audit)))
    cache=combine(caches)
    assert abs(float(cache['weights'].sum())-1)<1e-10
    torch.save(cache,path);dump(folder/'OOF_AUDIT_LOCAL_ONLY.json',audit)
    return cache


def train_gates(cache,job,folder):
    w=cache['weights'];features=cache['features'];domain=cache['domain']
    seq_loss=profile_loss(cache['seq_prediction'],cache)
    d=cache['info_prediction']-cache['seq_prediction'];e=cache['seq_prediction']-cache['query_y']
    n=cache['query_mask'].sum(1);mask=cache['query_mask']
    a=torch.where(mask,d.square(),0.).sum(1)/n;b=2*torch.where(mask,d*e,0.).sum(1)/n
    norms=torch.tensor([float((seq_loss[domain==i]*w[domain==i]).sum()/w[domain==i].sum()) for i in [0,1]],dtype=torch.float64).clamp_min(.01)
    loss_weights=w/norms[domain];gates={};logs={};audits=[]
    specs=[(mode,fs,.01) for mode in ['output','dynamics'] for fs in FEATURE_SETS]
    if job['variant']=='unimol':specs += [(mode,'full',r) for mode in ['output','dynamics'] for r in [.001,.1]]
    for mode,fs,reg in specs:
        name=f'{mode}_{fs}'+(f'_l2{reg:g}' if reg!=.01 else '')
        objective=(lambda alpha:((seq_loss+b*alpha+a*alpha.square())*loss_weights).sum()) if mode=='output' else (lambda alpha:(interpolate_loss(cache['grid'],alpha)*loss_weights).sum())
        gate,history=fit_gate(features,w,objective,fs,reg)
        with torch.no_grad():
            alpha=gate(features);exact=profile_loss(predict_cached(cache,alpha,mode),cache)
            approx=seq_loss+b*alpha+a*alpha.square() if mode=='output' else interpolate_loss(cache['grid'],alpha)
            gap=float(((approx-exact).abs()*w).sum())
            weighted=float((exact*w).sum())
            audits.append(dict(gate=name,exact_oof_mse=weighted,weighted_absolute_surrogate_gap=gap,
                relative_surrogate_gap=gap/max(weighted,1e-12),mean_alpha=float((alpha*w).sum()),
                min_alpha=float(alpha.min()),max_alpha=float(alpha.max())))
        gates[name]=gate;logs[name]=history
    torch.save({name:g.state_dict() for name,g in gates.items()},folder/'gates.pt')
    dump(folder/'GATE_TRAINING.json',dict(domain_sequential_mse=norms.tolist(),histories=logs,audit=audits,
        feature_sets=FEATURE_SETS,training_rows=len(features),gate_hyperparameter_selection='fixed before outer predictions'))
    return gates


def run_job(job):
    torch.set_num_threads(1);began=time.time();digest=sha256(PLAN)
    folder=RUN/'jobs'/job['id'];folder.mkdir(parents=True,exist_ok=True)
    if (folder/'COMPLETE.json').exists():
        r=json.loads((folder/'COMPLETE.json').read_text());assert r['plan_sha256']==digest
        for p,h in r['outputs'].items():assert sha256(folder/p)==h,p
        return job['id'],'already_complete'
    if (folder/'FAILED.json').exists():return job['id'],'FAILED retained'
    try:
        profiles,meta,_=load_profiles()
        cache=collect_oof(profiles,meta,job,folder)
        gates=train_gates(cache,job,folder)
        seq,info,z,ids,scaler,paths=final_experts(profiles,meta,job,folder)
        tr,va,test=indices_for_split(meta,job['fold']);fitting=np.sort(np.r_[tr,va])
        registry=z[[next(i for i in fitting if profiles[i]['drug_id']==d) for d in scaler['fitted_drugs']]]
        batch=make_batch(profiles,meta,test,z,ids)
        rows=[];diagnostics=[];test_cache=[];timings={};ep_error=0.
        for ki,k in enumerate(KS):
            c=cache_one(seq,info,batch,registry,job['variant'],ki);test_cache.append(c)
            with torch.no_grad():
                alphas={name:g(c['features']) for name,g in gates.items()}
                alphas.update(output_fixed=c['features'].new_full((len(test),),.5),dynamics_fixed=c['features'].new_full((len(test),),.5))
                if VARIANTS[job['variant']]['representation']!='none':
                    alphas['hard']=(batch.ids<0).double()
                predictions={'sequential':c['seq_prediction'],'information_sd1.5':c['info_prediction']}
                inputs=batch.inputs(torch.full((len(test),),ki))
                for config in [CONFIGS[0],CONFIGS[1],CONFIGS[3]]:
                    label='information_legacy' if config['kind']=='legacy' else f'information_sd{config["prior_sd"]:g}'
                    predictions[label]=info_predict(info,inputs,config)['mean']
                ref=seq(*inputs,variance=False)['mean'];ep_error=max(ep_error,float((ref-c['seq_prediction']).abs().max()))
                ref=info_predict(info,inputs,CONFIGS[2])['mean'];ep_error=max(ep_error,float((ref-c['info_prediction']).abs().max()))
                for name,alpha in alphas.items():
                    start=time.perf_counter();fusion='dynamics' if name.startswith('dynamics') else 'output'
                    predictions[name]=predict_cached(c,alpha,fusion)
                    timings[name]=timings.get(name,0.)+time.perf_counter()-start
            for name,pred in predictions.items():
                if not torch.isfinite(pred).all():raise FloatingPointError(f'Nonfinite prediction {name}')
                for row,index in enumerate(test):
                    p=profiles[index]
                    for col,target in enumerate(p['query'].itertuples()):
                        rows.append(dict(condition=name,variant=job['variant'],seed=job['seed'],mapping_seed=VARIANTS[job['variant']]['mapping'],
                            fold=job['fold'],k=k,observation_id=target.observation_id,source_id=p['source_id'],drug_id=p['drug_id'],
                            participant_id=p['participant_id'],profile_id=p['profile_id'],time_h=target.time,
                            y_true_lnc=target.target_log_dv,y_pred_lnc=float(pred[row,col])))
            for name,alpha in alphas.items():
                for row,index in enumerate(test):
                    diagnostics.append(dict(gate=name,k=k,profile_index=int(index),alpha=float(alpha[row]),
                        **{f'feature_{j}':float(c['features'][row,j]) for j in range(9)}))
        assert ep_error<1e-9
        pd.DataFrame(rows).to_csv(folder/'predictions_LOCAL_ONLY.csv.gz',index=False,float_format='%.17g')
        pd.DataFrame(diagnostics).to_csv(folder/'gate_diagnostics_LOCAL_ONLY.csv.gz',index=False,float_format='%.17g')
        torch.save(combine(test_cache),folder/'test_cache_LOCAL_ONLY.pt')
        dump(folder/'DEPLOYMENT.json',dict(expert_paths=paths,expert_hashes={p:sha256(ROOT/p) for p in paths},
            fitting_indices=fitting.tolist(),test_indices=test.tolist(),scaler=scaler,information_config=CONFIGS[2],
            gate_feature_representation=VARIANTS[job['variant']]['representation'],exact_ode_at_inference=True))
        names=['predictions_LOCAL_ONLY.csv.gz','gate_diagnostics_LOCAL_ONLY.csv.gz','gates.pt','GATE_TRAINING.json',
            'OOF_AUDIT_LOCAL_ONLY.json','oof_cache_LOCAL_ONLY.pt','test_cache_LOCAL_ONLY.pt','DEPLOYMENT.json']
        if (folder/'information_refit.pt').exists():names+=['information_refit.pt','NEW_INFORMATION_SELECTION.json']
        dump(folder/'COMPLETE.json',dict(**job,plan_sha256=digest,failures=0,prediction_rows=len(rows),
            methods=sorted(set(r['condition'] for r in rows)),oof_expert_fits=10,new_final_expert_fits=int(VARIANTS[job['variant']]['info'] is None),
            gate_fits=len(gates),elapsed_seconds=time.time()-began,endpoint_max_error=ep_error,
            stored_expert_parameters=sum(p.numel() for m in [seq,info] for p in m.parameters()),
            gate_parameters=10,forecast_only_seconds=timings,outputs={p:sha256(folder/p) for p in names}))
        return job['id'],f'complete {len(rows)} rows {time.time()-began:.1f}s'
    except Exception:
        dump(folder/'FAILED.json',dict(job=job,traceback=traceback.format_exc(),elapsed_seconds=time.time()-began))
        return job['id'],'FAILED '+traceback.format_exc().splitlines()[-1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['plan','calibrate','run'])
    parser.add_argument('--workers',type=int,default=24);parser.add_argument('--filter',default='');args=parser.parse_args()
    if args.action=='plan':freeze();return
    plan=json.loads(PLAN.read_text())
    for p,h in plan['hashes'].items():assert sha256(ROOT/p)==h,p
    if args.action=='calibrate':
        manifest=json.loads((PLAN.parent/'CROSSFIT_MANIFEST_LOCAL_ONLY.json').read_text())
        items=[(fold,ep) for fold,eps in manifest.items() for ep in eps];function=calibrate
    else:items=[j for j in plan['jobs'] if args.filter in j['id']];function=run_job
    failures=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn')) as pool:
        futures=[pool.submit(function,item) for item in items]
        for i,future in enumerate(as_completed(futures),1):
            name,status=future.result();print(f'[{i}/{len(items)}] {name}: {status}',flush=True)
            if status.startswith('FAILED'):failures.append(name)
            dump(RUN/f'PROGRESS_{args.action}.json',dict(finished=i,requested=len(items),failed=failures,latest=name,status=status))
    if failures:raise RuntimeError(f'{len(failures)} failed jobs retained')


if __name__=='__main__':main()
