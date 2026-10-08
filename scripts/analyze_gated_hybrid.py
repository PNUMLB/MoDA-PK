"""Validate every v7 job and generate paired, hierarchy-correct result tables."""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from pkforecast.data import load_profiles,indices_for_split,sha256,KS
from run_gated_hybrid import PLAN,RUN,VARIANTS
from run_relational import dump
from analyze_forecast import summarize,paired_bootstrap
from analyze_relational import LABELS as V3_LABELS

MAIN=['sequential','information_legacy','information_sd0.6','information_sd1.5','information_sd3',
      'hard','output_fixed','dynamics_fixed','output_full','dynamics_full']
LABELS=dict(sequential='Sequential observer',information_legacy='Information: one update',
    **{'information_sd0.6':'Iterated information (SD 0.6)','information_sd1.5':'Iterated information (SD 1.5)',
       'information_sd3':'Iterated information (SD 3)'},
    hard='Hard routing (SD 1.5)',output_fixed='Fixed output mixture',dynamics_fixed='Fixed dynamics hybrid',
    output_full='Learned output gate',dynamics_full='Learned dynamics gate',
    v6_routed='Earlier selected hard routing',v6_routed_fixed_sd3='Earlier SD3 hard routing',
    v5_iterated_selected='Earlier selected information',v3_history_gru='History GRU',
    v3_history_gru_relational='History GRU + relations',v3_deepct_adapted='DeepCt (adapted)',v3_unipk_adapted='Uni-PK (adapted)',
    v3_population_scale_map='Population + scale MAP')
LABELS.update({'v3_'+c:label.replace(' (ours)','') for c,label in V3_LABELS.items()})
LABELS['v3_relational']='Original sequential + relations'
LABELS['v3_information_relational']='Original information + relations'


def condition_name(method,variant):
    if variant=='unimol':return method
    return method+'__'+('random' if variant.startswith('random') else variant)


def main():
    plan=json.loads(PLAN.read_text());digest=sha256(PLAN)
    for p,h in plan['hashes'].items():assert sha256(ROOT/p)==h,p
    profiles,meta,_=load_profiles()
    manifest=json.loads((PLAN.parent/'CROSSFIT_MANIFEST_LOCAL_ONLY.json').read_text())
    frames=[];receipts=[];inputs={};gates=[];training=[];audits=[]
    for number,job in enumerate(plan['jobs'],1):
        folder=RUN/'jobs'/job['id'];receipt=json.loads((folder/'COMPLETE.json').read_text())
        assert receipt['plan_sha256']==digest and receipt['failures']==0
        assert not (folder/'FAILED.json').exists(),job['id']
        for p,h in receipt['outputs'].items():assert sha256(folder/p)==h,(job['id'],p)
        inputs[str((folder/'COMPLETE.json').relative_to(ROOT))]=sha256(folder/'COMPLETE.json')
        tr,va,te=indices_for_split(meta,job['fold']);fit=np.sort(np.r_[tr,va])
        deployment=json.loads((folder/'DEPLOYMENT.json').read_text())
        assert deployment['fitting_indices']==fit.tolist() and deployment['test_indices']==te.tolist()
        for p,h in deployment['expert_hashes'].items():assert sha256(ROOT/p)==h,p
        inner=json.loads((folder/'OOF_AUDIT_LOCAL_ONLY.json').read_text())
        assert len(inner)==5
        for a,e in zip(inner,manifest[job['fold']]):
            for key in ['name','kind','fitting_indices','heldout_indices']:assert a[key]==e[key]
            assert set(a['fitting_indices']+a['heldout_indices']).isdisjoint(te)
            allowed={str(q.observation_id) for i in a['fitting_indices'] for q in profiles[i]['query'].itertuples()}
            assert set(map(str,a['relation_query_ids']))==allowed
            assert set(a['scaler_fitted_drugs'])==set(meta.iloc[a['fitting_indices']].drug_id)
            for c in a['checkpoint_hashes']:assert sha256(ROOT/c['path'])==c['sha256']
            cp=RUN/'calibration'/f'{job["fold"]}__{a["name"]}.json'
            assert sha256(cp)==a['calibration_sha256']
            audits.append(dict(job=job['id'],episode=a['name'],kind=a['kind'],fit_profiles=len(a['fitting_indices']),
                held_profiles=len(a['heldout_indices']),fit_drugs=len(a['scaler_fitted_drugs']),outer_excluded=True))
        expected=pd.concat([profiles[i]['query'].assign(source_id=profiles[i]['source_id'],drug_id=profiles[i]['drug_id'],
            participant_id=profiles[i]['participant_id'],profile_id=profiles[i]['profile_id']) for i in te]).set_index('observation_id')
        frame=pd.read_csv(folder/'predictions_LOCAL_ONLY.csv.gz',float_precision='round_trip')
        assert len(frame)==receipt['prediction_rows'] and set(frame.k)==set(KS)
        assert sorted(frame.condition.unique())==receipt['methods']
        assert set(frame.variant)=={job['variant']} and set(frame.seed)=={job['seed']}
        for (condition,k),g in frame.groupby(['condition','k']):
            assert g.observation_id.is_unique and set(g.observation_id)==set(expected.index)
            true=expected.loc[g.observation_id]
            for field in ['source_id','drug_id','participant_id','profile_id']:
                assert np.array_equal(g[field].astype(str),true[field].astype(str))
            np.testing.assert_allclose(g.y_true_lnc,true.target_log_dv,atol=1e-12,rtol=0)
            np.testing.assert_allclose(g.time_h,true.time,atol=1e-12,rtol=0)
            assert np.isfinite(g.y_pred_lnc).all()
        frame.condition=[condition_name(c,job['variant']) for c in frame.condition]
        frames.append(frame.drop(columns='variant'))
        d=pd.read_csv(folder/'gate_diagnostics_LOCAL_ONLY.csv.gz')
        for key in ['fold','seed','variant']:d[key]=job[key]
        d['mapping_seed']=VARIANTS[job['variant']]['mapping'];gates.append(d)
        gt=json.loads((folder/'GATE_TRAINING.json').read_text())
        for a in gt['audit']:training.append(dict(**job,**a))
        receipts.append(receipt)
        if number%30==0:print(f'Validated {number}/{len(plan["jobs"])} jobs',flush=True)
    # Existing adapted baselines and earlier development controls remain explicit.
    inherited=ROOT/'results/forecast_v3/all_predictions_LOCAL_ONLY.csv.gz'
    v3=pd.read_csv(inherited)
    v3.condition='v3_'+v3.condition
    frames.append(v3[[c for c in frames[0].columns if c in v3.columns]])
    inputs[str(inherited.relative_to(ROOT))]=sha256(inherited)
    for fold in ['known']+[f'loso_{i:02d}' for i in range(10)]:
        for seed in [1,2,3]:
            name=f'{fold}__information_relational__s{seed}__m17'
            for version,selected,prefix in [('forecast_v6_routing',['routed','routed_fixed_sd3'],'v6_'),('forecast_v5_iterated',['iterated_selected'],'v5_')]:
                path=ROOT/f'results/{version}/jobs/{name}/predictions_LOCAL_ONLY.csv.gz'
                f=pd.read_csv(path);f=f[f.condition.isin(selected)].copy();f.condition=prefix+f.condition
                frames.append(f[frames[0].columns]);inputs[str(path.relative_to(ROOT))]=sha256(path)
    p=pd.concat(frames,ignore_index=True);p['task']=np.where(p.fold=='known','known','loso')
    assert not p.duplicated(['condition','seed','mapping_seed','k','observation_id','task']).any()
    summary,seed,source,drug,p=summarize(p)
    tables=RUN/'tables';tables.mkdir(exist_ok=True)
    for name,f in dict(all_conditions=summary,primary_k2=summary[summary.k==2],seed_metrics=seed,
        source_metrics=source,drug_metrics=drug,oof_audit=pd.DataFrame(audits),gate_training=pd.DataFrame(training)).items():
        f.to_csv(tables/f'{name}.csv',index=False)
    pd.DataFrame([{k:r[k] for k in ['fold','variant','seed','elapsed_seconds','stored_expert_parameters','gate_parameters','oof_expert_fits','new_final_expert_fits','gate_fits']} for r in receipts]).to_csv(tables/'compute_by_job.csv',index=False)
    pd.DataFrame(receipts).groupby('variant').agg(jobs=('id','size'),median_job_seconds=('elapsed_seconds','median'),
        sum_job_seconds=('elapsed_seconds','sum'),minimum_expert_parameters=('stored_expert_parameters','min'),
        maximum_expert_parameters=('stored_expert_parameters','max')).to_csv(tables/'compute_summary.csv')
    point=summary[summary.k==2].pivot(index='condition',columns='task',values='sb_hrmse')
    for v in VARIANTS:
        orig=VARIANTS[v]['seq'];current=condition_name('sequential',v)
        assert np.max(np.abs(point.loc[current]-point.loc['v3_'+orig]))<1e-10
        if VARIANTS[v]['info'] is not None:
            assert np.max(np.abs(point.loc[condition_name('information_legacy',v)]-point.loc['v3_'+VARIANTS[v]['info']]))<1e-10
    comparisons=[];rng=np.random.default_rng(26091707)
    primary=['output_full','dynamics_full']
    comparators=MAIN+['v6_routed','v6_routed_fixed_sd3','v3_history_gru','v3_history_gru_relational','v3_deepct_adapted','v3_unipk_adapted']
    comparators += [f'{mode}_{feature}' for mode in ['output','dynamics'] for feature in ['no_distance','distance_only','constant']]
    comparators += [f'{c}__{v}' for c in primary for v in ['no_relations','none','id','random']]
    for task in ['known','loso']:
        wide=source[(source.task==task)&(source.k==2)].groupby(['source_id','condition']).sb_hrmse.mean().unstack()
        draws=rng.integers(0,len(wide),size=(10000,len(wide)))
        for c in primary:
            for other in comparators:
                if c==other:continue
                d=(wide[c]-wide[other]).to_numpy();lo,hi=np.quantile(d[draws].mean(1),[.025,.975])
                comparisons.append(dict(task=task,candidate=c,comparator=other,difference=d.mean(),ci_low=lo,ci_high=hi,
                    improved_sources=int((d < -1e-10).sum()),tied_sources=int((abs(d)<=1e-10).sum()),worse_sources=int((d>1e-10).sum()),
                    draws=10000,scope='source blocks conditional on fixed fits; exploratory, not selection-adjusted'))
    pd.DataFrame(comparisons).to_csv(tables/'source_block_comparisons.csv',index=False)
    participant=[]
    for c in primary:
        chosen=[c,'sequential','information_sd1.5','hard','output_fixed','dynamics_fixed','v6_routed']
        sub=p[p.condition.isin(chosen)].copy();sub.condition=sub.condition.replace({c:'v4_unimol'})
        b=paired_bootstrap(sub,draws=5000);b.candidate=c;participant.append(b)
    pd.concat(participant).to_csv(tables/'participant_block_comparisons.csv',index=False)
    gate=pd.concat(gates,ignore_index=True)
    gate=gate.merge(meta.reset_index(names='profile_index')[['profile_index','drug_id','source_id','participant_id','profile_id']],on='profile_index',validate='many_to_one')
    gate['task']=np.where(gate.fold=='known','known','loso')
    gate.to_csv(RUN/'gate_diagnostics_LOCAL_ONLY.csv.gz',index=False)
    gate.groupby(['variant','task','gate','k']).agg(mean_alpha=('alpha','mean'),median_alpha=('alpha','median'),
        sd_alpha=('alpha','std'),low_saturation=('alpha',lambda x:np.mean(x<.05)),high_saturation=('alpha',lambda x:np.mean(x>.95)),
        profile_runs=('alpha','size')).to_csv(tables/'gate_summary.csv')
    gate.groupby(['variant','task','gate','k','source_id','drug_id']).alpha.agg(['mean','std','min','max']).to_csv(tables/'gate_by_drug.csv')
    p.to_csv(RUN/'all_predictions_LOCAL_ONLY.csv.gz',index=False,float_format='%.17g')
    dump(RUN/'ANALYSIS_COMPLETE.json',dict(passed=True,completed_jobs=len(receipts),expected_jobs=len(plan['jobs']),
        new_prediction_rows=sum(r['prediction_rows'] for r in receipts),aligned_prediction_rows=len(p),
        oof_expert_fits=sum(r['oof_expert_fits'] for r in receipts),additional_information_selection_refit_pipelines=sum(r['new_final_expert_fits'] for r in receipts),
        gate_fits=sum(r['gate_fits'] for r in receipts),endpoint_max_error=max(r['endpoint_max_error'] for r in receipts),
        plan_sha256=digest,analysis_sha256=sha256(Path(__file__)),inputs=inputs,
        max_relative_surrogate_gap=max(r['relative_surrogate_gap'] for r in training),
        all_prediction_rows_and_metadata_checked=True,all_oof_exclusions_checked=True))
    print(point.loc[MAIN+['v6_routed','v3_deepct_adapted','v3_unipk_adapted']].round(5).to_string())


if __name__=='__main__':main()
