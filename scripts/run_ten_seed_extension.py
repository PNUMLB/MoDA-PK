"""Immutable seed 6..10 extension; preserves all existing seed 1..5 evidence."""
from pathlib import Path
import sys,os,json,time,hashlib,tarfile,subprocess,shutil,itertools
from concurrent.futures import ThreadPoolExecutor,as_completed
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'));sys.path.insert(0,str(R/'scripts'))
B=R/'results/runs/TEN_SEEDS/20261001';O=R/'results/summaries/ten_seeds_20261001';OLD=R/'results/runs/TRACEABLE/evaluation_20260925'
ENV=os.environ|dict(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2));tmp.replace(p)
def initialize():
 from mhap_pk.data.traceable import load_profiles,indices_for_split
 ps,meta,_=load_profiles();base=json.loads((OLD/'plan.json').read_text());seeds=list(range(6,11));drugs=sorted(meta.drug_id.unique());cfg=base|dict(experiment='TEN_SEEDS_20261001',seeds=seeds,report_seeds=list(range(1,11)),additional_inference_k=[3],training_k=[0,1,2,4],initial_observation_evaluation_k=list(range(5)),common_origin_evaluation_k=[0,1,2,4],selection='Same original validation k2 checkpoint selection and hyperparameters; all 10 seeds retained regardless of score',extension='Seed 1..5 reused from immutable parents; 6..10 newly trained')
 assert sha(R/'data/splits/source_traceable_20260925/partitions.json')==base['split_sha256']
 jobs=[]
 for fold in base['folds']:
  for seed in seeds:
   for mode in base['modes']:jobs.append(dict(family='panel',fold=fold,seed=seed,mode=mode,drug=None))
   for mode in ['base','parameter','state']:jobs.append(dict(family='components',fold=fold,seed=seed,mode=mode,drug=None))
 for seed in seeds:
  for drug in drugs:jobs.append(dict(family='single',fold='known',seed=seed,mode='molecular',drug=drug))
  jobs.append(dict(family='poppk',fold='known',seed=seed,mode='poppk_vi_map',drug=None))
 plan=cfg|dict(jobs=jobs,expected_worker_jobs=len(jobs),expected_new_model_fits=860,workers=40,slow_workers=24,fast_workers=16,drug_ids=drugs,scope='Current source-traceable primary panel and retrained components/representations/relations/offset controls; drug-specific MHAP and PopPK. Separate historical fixed-prior/simulation studies are not relabelled as ten-seed evidence.')
 B.mkdir(parents=True,exist_ok=True);O.mkdir(parents=True,exist_ok=True)
 if (B/'registered_plan.json').exists():assert json.loads((B/'registered_plan.json').read_text())==plan
 else:
  save(B/'registered_plan.json',plan)
  paths=list((R/'src/mhap_pk').rglob('*.py'))+[Path(__file__),R/'scripts/run_aligned_components.py',R/'scripts/run_traceable_poppk.py',R/'scripts/complete_comparison_refresh.py']
  with tarfile.open(B/'source_snapshot.tar.gz','w:gz') as t:
   for p in paths:t.add(p,arcname=str(p.relative_to(R)))
  save(B/'input_hashes.json',{str(p.relative_to(R)):sha(p) for p in paths+[R/'data/splits/source_traceable_20260925/partitions.json',R/'data/processed/full16_v1/primary_full16.csv',R/'data/processed/full16_v1/dose_events.csv',R/'data/processed/full16_v1/embeddings/unimol.npz']})
  (B/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
 for name in ['panel','components','parity']:
  root=B/name;root.mkdir(exist_ok=True);file=root/'plan.json'
  if not file.exists():save(file,cfg)
  if not (root/'calibration').exists():shutil.copytree(OLD/'calibration',root/'calibration')
 for drug in drugs:
  root=B/'single'/drug;root.mkdir(parents=True,exist_ok=True)
  if not (root/'plan.json').exists():save(root/'plan.json',cfg|dict(target_drug=drug,fit_scope='single_drug',modes=['molecular'],relation_scope='no eligible distinct-drug pairs'))
  parent=R/'results/runs/SINGLE_DRUG/20261001'/drug/'calibration'
  if not (root/'calibration').exists():shutil.copytree(parent,root/'calibration')
 root=B/'poppk';root.mkdir(exist_ok=True);popcfg=json.loads((R/'results/runs/TRACEABLE_POPPK/evaluation_20260925/plan.json').read_text())|dict(seeds=seeds,additional_inference_k=[3],experiment='TEN_SEEDS_POPPK_20261001')
 if not (root/'plan.json').exists():save(root/'plan.json',popcfg)
 return plan

def outpath(j):
 if j['family']=='single':return B/'single'/j['drug']/f"known__s{j['seed']}__molecular"
 if j['family']=='poppk':return B/'poppk'/f"full__s{j['seed']}"
 return B/j['family']/f"{j['fold']}__s{j['seed']}__{j['mode']}"

def worker(family,fold,seed,mode,drug):
 if family in ['panel','single','parity']:
  from mhap_pk.training.traceable_panel import run
  from complete_comparison_refresh import build
  root=B/'single'/drug if family=='single' else B/family
  run(fold,seed,mode,output_root=root,model_builder=build,target_drug=drug if family=='single' else None)
 elif family=='components':
  from run_aligned_components import run
  run(B/'components',fold,seed,mode)
 elif family=='poppk':
  from run_traceable_poppk import run
  run('full',seed,output_root=B/'poppk')
 else:raise ValueError(family)

def launch(j):
 out=outpath(j);m=out/'run_manifest.json'
 if m.exists():return j|dict(status=json.loads(m.read_text())['status'],reused=True)
 if out.exists():return j|dict(status='incomplete',reason='existing directory has no manifest; preserved')
 log=B/'logs'/(j['family']+'__'+out.name+'.log');log.parent.mkdir(exist_ok=True)
 with log.open('w') as f:
  result=subprocess.run([sys.executable,__file__,'--worker',j['family'],j['fold'],str(j['seed']),j['mode'],j['drug'] or '-'],env=ENV,stdout=f,stderr=subprocess.STDOUT)
 return j|dict(status='completed' if result.returncode==0 else 'failed',exit_code=result.returncode,manifest=str(m.relative_to(R)))

def parity():
 import pandas as pd,numpy as np
 j=dict(family='parity',fold='known',seed=1,mode='molecular_relation',drug=None);result=launch(j)
 if result['status']!='completed':raise RuntimeError(result)
 out=outpath(j);p=pd.read_parquet(out/'predictions.parquet');old=pd.read_parquet(OLD/'known__s1__molecular_relation/predictions.parquet');keys=['protocol','k','observation_id'];a=p[p.k!=3].set_index(keys).sort_index();b=old.set_index(keys).sort_index();assert a.index.equals(b.index)
 err=float(np.max(abs(a.y_pred_lnc-b.y_pred_lnc)));assert err<1e-10
 ref=pd.read_parquet(R/'results/runs/HISTORY_COUNT/20261001/known__s1__molecular_relation/predictions.parquet').set_index('observation_id').sort_index();new=p[p.k==3].set_index('observation_id').sort_index();err3=float(np.max(abs(new.y_pred_lnc-ref.y_pred_log)));assert err3<1e-10
 save(B/'parity_check.json',dict(status='passed',max_original_prediction_difference=err,max_k3_difference=err3,weight_hash_new=sha(out/'selected.pt'),scope='Full training replay seed1; code additions only change output-root handling and optional extra inference'))

def main():
 plan=initialize();save(B/'status.json',dict(status='parity_check',pid=os.getpid(),started=time.time(),expected_worker_jobs=len(plan['jobs'])));parity();started=time.time();save(B/'status.json',dict(status='running',pid=os.getpid(),started=started,expected_worker_jobs=len(plan['jobs'])))
 slow=[j for j in plan['jobs'] if j['mode'] in ['unipk','deepct']];fast=[j for j in plan['jobs'] if j['mode'] not in ['unipk','deepct']];rows=[]
 with ThreadPoolExecutor(max_workers=plan['slow_workers']) as sp,ThreadPoolExecutor(max_workers=plan['fast_workers']) as fp:
  futures=[sp.submit(launch,j) for j in slow]+[fp.submit(launch,j) for j in fast]
  for future in as_completed(futures):
   row=future.result();rows.append(row);save(B/'progress.json',rows);completed=sum(x['status']=='completed' for x in rows);save(B/'status.json',dict(status='running',pid=os.getpid(),started=started,finished_jobs=len(rows),completed_jobs=completed,failed_jobs=len(rows)-completed,expected_worker_jobs=len(plan['jobs']),elapsed_seconds=time.time()-started));print(len(rows),'/',len(plan['jobs']),row,flush=True)
 status='completed' if all(x['status']=='completed' for x in rows) else 'completed_with_failures';save(B/'status.json',dict(status=status,jobs=rows,expected_worker_jobs=len(plan['jobs']),elapsed_seconds=time.time()-started));print(status,flush=True)
 if status=='completed':
  script=R/'scripts/summarize_ten_seed_extension.py'
  if script.exists():
   with (B/'aggregation.log').open('w') as log:rc=subprocess.call([sys.executable,str(script)],stdout=log,stderr=subprocess.STDOUT,env=ENV)
   save(B/'aggregation_status.json',dict(status='completed' if rc==0 else 'failed',exit_code=rc))
if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='--worker':
  if sys.argv[2]=='single':
   from contextlib import redirect_stdout,redirect_stderr
   import traceback
   log=B/'single'/sys.argv[6]/f'known__s{sys.argv[4]}__molecular.worker.log'
   with log.open('x') as stream,redirect_stdout(stream),redirect_stderr(stream):
    try:worker(sys.argv[2],sys.argv[3],int(sys.argv[4]),sys.argv[5],sys.argv[6])
    except Exception:traceback.print_exc();raise
  else:worker(sys.argv[2],sys.argv[3],int(sys.argv[4]),sys.argv[5],None if sys.argv[6]=='-' else sys.argv[6])
 else:main()
