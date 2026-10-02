"""Registered CV/ratio validation using existing trainers via explicit partition providers.
No change to model mathematics, losses, epoch budgets or checkpoint selection.
"""
from pathlib import Path
import sys,os,json,time,hashlib,subprocess,tarfile,fcntl
from concurrent.futures import ThreadPoolExecutor,as_completed
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'src'),str(R/'scripts')]
B=R/'results/runs/RESAMPLING/20261002';S=R/'data/splits/participant_resampling_20261002'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(p)
def provider(meta,fold,partition):
 from mhap_pk.data.resampling import partition_check
 reg=json.loads((S/'partitions.json').read_text());assert fold=='known'
 assert meta[['source_id','drug_id','participant_id','profile_id']].astype(str).values.tolist()==reg['profile_keys']
 return partition_check(meta,reg['partitions'][partition])
def register():
 import pandas as pd
 from mhap_pk.data.traceable import load_profiles
 from mhap_pk.data.resampling import make_partitions
 ps,meta,_=load_profiles();parts=make_partitions(meta)
 reg=dict(split_id=S.name,dataset_id='source_traceable_20260925',split_seed=20261002,grouping='source_id + participant_id; coadministration and occasions indivisible; unresolved cross-source identities are not invented',profile_keys=meta[['source_id','drug_id','participant_id','profile_id']].astype(str).values.tolist(),partitions=parts,protocol='5-fold OOF; inner validation approximately 10% of total per linked-drug stratum. Rounded minimum one per role; fixed 712/811 validation and nested tests.',legacy_split_sha256=sha(R/'data/splits/source_traceable_20260925/partitions.json'))
 if (S/'partitions.json').exists():assert json.loads((S/'partitions.json').read_text())==reg
 else:write(S/'partitions.json',reg)
 counts=[]
 for part,spec in parts.items():
  for role,ix in spec.items():
   g=meta.iloc[ix]
   for drug,x in [('ALL',g)]+list(g.groupby('drug_id')):counts.append(dict(partition=part,role=role,drug_id=drug,profiles=len(x),participants=len(x[['source_id','participant_id']].drop_duplicates())))
 pd.DataFrame(counts).to_csv(S/'actual_counts.csv',index=False)
 cfg=json.loads((R/'configs/experiments/source_traceable_20260925.json').read_text())|dict(experiment='RESAMPLING_20261002',split_id=S.name,split_sha256=sha(S/'partitions.json'),seeds=list(range(1,11)),additional_inference_k=[3],folds=['known'],primary_k=2,internal_validation_only=True)
 jobs=[];calibrations=[]
 for part in parts:
  for scope,drug in [('pooled',None)]+[('single',d) for d in sorted(meta.drug_id.unique())]:
   root=B/part/scope/(drug or 'all');write(root/'plan.json',cfg|dict(partition=part,fit_scope=scope,target_drug=drug));calibrations.append(dict(partition=part,scope=scope,drug=drug,root=str(root.relative_to(R))))
   for seed in range(1,11):
    for mode in ['molecular_relation','dcm','unipk','deepct']:jobs.append(dict(kind='neural',partition=part,scope=scope,drug=drug,seed=seed,mode=mode,label=mode,root=str(root.relative_to(R))))
  for ablation in ['without_updates','parameter_only','state_only']:
   root=B/part/'ablation'/ablation;write(root/'plan.json',cfg|dict(partition=part,fit_scope='pooled_ablation',ablation=ablation))
   for seed in range(1,11):jobs.append(dict(kind='neural',partition=part,scope='ablation',drug=None,seed=seed,mode='molecular_relation',label=ablation,root=str(root.relative_to(R))))
  root=B/part/'poppk';old=json.loads((R/'results/runs/TRACEABLE_POPPK/evaluation_20260925/full__s1/config.resolved.json').read_text());write(root/'plan.json',old|dict(split_id=S.name,split_sha256=sha(S/'partitions.json'),partition=part,seeds=list(range(1,11)),additional_inference_k=[3],cases=[dict(case='full',drugs=[],cap=None)]))
  for seed in range(1,11):jobs.append(dict(kind='poppk',partition=part,scope='drug_specific',drug=None,seed=seed,mode='poppk_vi_map',label='poppk_vi_map',root=str(root.relative_to(R))))
 plan=dict(status='registered',jobs=jobs,calibrations=calibrations,job_count=len(jobs),fit_count=sum(11 if j['kind']=='poppk' else 1 for j in jobs),workers=32,split_sha256=sha(S/'partitions.json'),test_use='Evaluation only; all hyperparameters fixed. Reused cohort, internal validation, not independent external validation.',selection='Existing per-model validation k=2; neural epoch 0..120 every10, PopPK positive epoch only. No method chosen using new test results.',aggregation='CV: concatenate once-per-participant OOF predictions before drug metrics, separately per seed. Ratios: distinct test populations, not paired full-test comparisons. Also report matched nested 10% test subset.',joint='Existing Joint drug-source experiment unchanged',seed_interpretation='Ten optimization seeds on one registered split allocation; five CV folds are not fifty independent repetitions',dataset_files={str(p.relative_to(R)):sha(p) for p in [R/'data/processed/full16_v1/primary_full16.csv',R/'data/processed/full16_v1/dose_events.csv']})
 if (B/'registered_plan.json').exists():assert json.loads((B/'registered_plan.json').read_text())==plan
 else:
  write(B/'registered_plan.json',plan)
  files=list((R/'src/mhap_pk').rglob('*.py'))+[Path(__file__),R/'scripts/run_traceable_poppk.py',R/'scripts/run_aligned_components.py',R/'scripts/complete_comparison_refresh.py',R/'scripts/summarize_resampling_validation.py']
  with tarfile.open(B/'source_snapshot.tar.gz','w:gz') as t:
   for p in files:t.add(p,arcname=str(p.relative_to(R)))
  write(B/'source_manifest.json',{str(p.relative_to(R)):sha(p) for p in files});(B/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
 print('REGISTERED',len(jobs),'jobs',plan['fit_count'],'fits',flush=True);return plan

def calibration(index):
 import torch
 torch.set_num_threads(1)
 from mhap_pk.data.traceable import load_profiles
 from mhap_pk.data import representations,make_batch
 from mhap_pk.training.population import fit_population
 plan=json.loads((B/'registered_plan.json').read_text());j=plan['calibrations'][index];root=R/j['root'];path=root/'calibration/known/population.json'
 ps,meta,_=load_profiles();tr=provider(meta,'known',j['partition'])[0]
 if j['drug']:tr=tr[meta.iloc[tr].drug_id.to_numpy()==j['drug']]
 if path.exists():assert json.loads(path.read_text())['fitting_indices']==tr.tolist();return
 z,ids,_=representations(meta,tr,'unimol',17);_,cal=fit_population(make_batch(ps,meta,tr,z,ids));write(path,cal|dict(split_sha256=plan['split_sha256'],partition=j['partition'],training_only=True))
 if j['scope']=='pooled':
  for a in ['without_updates','parameter_only','state_only']:write(B/j['partition']/'ablation'/a/'calibration/known/population.json',cal|dict(split_sha256=plan['split_sha256'],partition=j['partition'],training_only=True))
 print('CALIBRATED',index,flush=True)

def output(j):return R/j['root']/(f"full__s{j['seed']}" if j['kind']=='poppk' else f"known__s{j['seed']}__{j['mode']}")
def worker(index):
 plan=json.loads((B/'registered_plan.json').read_text());j=plan['jobs'][index];root=R/j['root'];splitter=lambda meta,fold:provider(meta,fold,j['partition'])
 if j['kind']=='poppk':
  import run_traceable_poppk as trainer
  trainer.indices_for_split=splitter
  trainer.run('full',j['seed'],output_root=root)
 else:
  import mhap_pk.training.traceable_panel as trainer
  from complete_comparison_refresh import build
  trainer.indices_for_split=splitter
  def builder(mode,n,pop):
   if j['scope']=='ablation':
    from run_aligned_components import model as ablation_model
    return ablation_model({'without_updates':'base','parameter_only':'parameter','state_only':'state'}[j['label']],n,pop)
   return build(mode,n,pop)
  trainer.run('known',j['seed'],j['mode'],output_root=root,model_builder=builder,target_drug=j['drug'])
 path=output(j)/'run_manifest.json';m=json.loads(path.read_text());m.update(evaluation_partition=j['partition'],split_id=S.name,split_sha256=plan['split_sha256'],fit_scope=j['scope'],model_id=j['label'],source_snapshot_sha256=sha(B/'source_snapshot.tar.gz'));write(path,m)

def launch(kind,index):
 log=B/'logs'/f'{kind}_{index:04d}.log';log.parent.mkdir(exist_ok=True)
 with log.open('w') as h:rc=subprocess.call([sys.executable,__file__,'--'+kind,str(index)],cwd=R,stdout=h,stderr=subprocess.STDOUT,env=os.environ|dict(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
 return dict(index=index,status='completed' if rc==0 else 'failed',exit_code=rc)
def main():
 B.mkdir(parents=True,exist_ok=True)
 lock=(B/'driver.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 plan=register();write(B/'status.json',dict(status='running',phase='calibration',jobs=plan['job_count'],fits=plan['fit_count'],pid=os.getpid(),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
 with ThreadPoolExecutor(max_workers=16) as pool:
  for f in as_completed([pool.submit(launch,'calibrate',i) for i in range(len(plan['calibrations']))]):
   row=f.result()
   if row['status']!='completed':write(B/'status.json',dict(status='failed',phase='calibration',failure=row));return
 jobs=list(enumerate(plan['jobs']))
 # A first-seed coverage pass, then remaining seeds; alternate pooled and individual jobs.
 jobs.sort(key=lambda it:(it[1]['seed']!=1,it[1]['partition'],it[1]['scope'] not in ['pooled','ablation'],it[1]['mode']!='unipk',it[0]))
 rows=[];write(B/'status.json',dict(status='running',phase='training',jobs=len(jobs),fits=plan['fit_count'],pid=os.getpid()))
 def one(item):
  i,j=item;p=output(j)/'run_manifest.json'
  if p.exists():
   prev=json.loads(p.read_text())
   if prev.get('status')=='completed':return dict(index=i,status='completed',reused=True)
   return dict(index=i,status='failed',reason='Existing incomplete run preserved; manual diagnosis required')
  return launch('worker',i)
 with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
  for f in as_completed([pool.submit(one,it) for it in jobs]):
   rows.append(f.result());write(B/'progress.json',rows);print(len(rows),'/',len(jobs),rows[-1],flush=True)
 ok=all(x['status']=='completed' for x in rows);write(B/'status.json',dict(status='completed' if ok else 'completed_with_failures',phase='training',jobs=len(jobs),results=rows,finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
 if ok:
  r=launch('summarize',0);write(B/'summary_status.json',r)
if __name__=='__main__':
 if '--register' in sys.argv:register()
 elif '--calibrate' in sys.argv:calibration(int(sys.argv[2]))
 elif '--worker' in sys.argv:worker(int(sys.argv[2]))
 elif '--summarize' in sys.argv:
  from summarize_resampling_validation import main as summarize
  summarize()
 else:main()
