"""Migrate every retained primary-model analysis to the adopted hybrid model."""
from pathlib import Path
import sys,os,json,shutil,time,hashlib,subprocess,tarfile
from concurrent.futures import ThreadPoolExecutor,as_completed
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'src'),str(R/'scripts')]
B=R/'results/runs/HYBRID_PRIMARY/20261002'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2));tmp.replace(p)
def register():
 if (B/'registered_plan.json').exists():return json.loads((B/'registered_plan.json').read_text())
 base=R/'results/runs/TRACEABLE/evaluation_20260925';cfg=json.loads((base/'plan.json').read_text())|dict(seeds=list(range(1,11)),additional_inference_k=[3],model='MoDE-PK molecular+identity',relations='OFF unless explicitly tested',selection='Fixed adopted hybrid recipe; validation checkpoint selection. Model identity revised after prior internal test inspection; not independent confirmation.')
 jobs=[]
 def add(root,parent,extra,j):
  if not (root/'plan.json').exists():
   root.mkdir(parents=True,exist_ok=True);parentcfg=json.loads((parent/'plan.json').read_text());save(root/'plan.json',parentcfg|cfg|extra)
   shutil.copytree(parent/'calibration',root/'calibration')
   if (parent/'partition.json').exists():shutil.copy2(parent/'partition.json',root/'partition.json')
  jobs.append(j|dict(root=str(root.relative_to(R)),parent=str(parent.relative_to(R))))
 variants=['without_updates','parameter_only','state_only','without_molecule','without_identity','chemical_only','pk_only','both_relations','ecfp','chemberta_mlm','chemberta_mtr','memoryless','budget240']
 for variant in variants:
  for fold in cfg['folds']:
   for seed in range(1,11):add(B/'panel'/variant,base,dict(variant=variant,epochs=240 if variant=='budget240' else 120),dict(kind='panel',variant=variant,fold=fold,seed=seed))
 parts=json.loads((R/'data/splits/participant_resampling_20261002/partitions.json').read_text())['partitions']
 for part in parts:
  parent=R/'results/runs/RESAMPLING/20261002'/part/'pooled/all'
  for seed in range(1,11):add(B/'resampling'/part,parent,dict(partition=part,split_id='participant_resampling_20261002',split_sha256=sha(R/'data/splits/participant_resampling_20261002/partitions.json')),dict(kind='resampling',variant='full',partition=part,fold='known',seed=seed))
 old=json.loads((R/'results/runs/TRAINING_SCOPE/20261001/registered_plan.json').read_text())['jobs']
 for j in old:
  if j['kind']!='low_data':continue
  parent=R/j['root'];root=B/'low_data'/j['drug']/f"n{j['cap']}"/f"s{j['seed']}"/j['scope']
  add(root,parent,dict(fit_scope='matched_'+j['scope'],target_drug=j['drug'],target_training_participants=j['cap']),dict(kind='low_data',variant='full',drug=j['drug'],cap=j['cap'],scope=j['scope'],fold='known',seed=j['seed']))
 from mhap_pk.data.traceable import load_profiles
 _,meta,_=load_profiles()
 for drug in sorted(meta.drug_id.unique()):
  parent=R/'results/runs/SINGLE_DRUG/20261001'/drug
  for seed in range(1,11):add(B/'single'/drug,parent,dict(target_drug=drug,fit_scope='single_drug'),dict(kind='single',variant='full',drug=drug,fold='known',seed=seed))
 plan=dict(status='registered',utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),new_fits=len(jobs),jobs=jobs,workers=32,primary='reuse100 MOLECULAR_IDENTITY fits; Uni-Mol+ID, both updates, no relation losses',adoption='User-directed revision after internal test results; report Known trade-off and model-selection history once. No independent external evaluation.',legacy='All old model analyses archived; no old exploratory encoder/synthetic/fixed-prior scores labeled as new Ours.',main='same six profiles Known and Joint, all comparator curves; new ablation, embedding, history, horizon, failures, pooling, resampling, budget.')
 save(B/'registered_plan.json',plan)
 files=list((R/'src/mhap_pk').rglob('*.py'))+[Path(__file__)]
 save(B/'code_manifest.json',{str(p.relative_to(R)):sha(p) for p in files})
 with tarfile.open(B/'source_snapshot.tar.gz','w:gz') as t:
  for p in files:t.add(p,arcname=str(p.relative_to(R)))
 (B/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
 print('REGISTERED',len(jobs),flush=True);return plan

def worker(i):
 from types import FunctionType
 import torch
 from mhap_pk.training.traceable_panel import run as original
 from mhap_pk.models.priors.hybrid_controls import build_control
 j=json.loads((B/'registered_plan.json').read_text())['jobs'][i];root=R/j['root'];bindings=dict(original.__globals__)
 if j['variant'] in ['ecfp','chemberta_mlm','chemberta_mtr']:
  from mhap_pk.data.molecular_encoders import representations
  bindings['representations']=lambda meta,tr,name,seed:representations(meta,tr,j['variant'])
 if j['kind']=='resampling':
  from run_resampling_validation import provider
  bindings['indices_for_split']=lambda meta,fold:provider(meta,fold,j['partition'])
 run=FunctionType(original.__code__,bindings,original.__name__,original.__defaults__,original.__closure__)
 mode={'chemical_only':'chemical_only','pk_only':'pk_only','both_relations':'molecular_relation'}.get(j['variant'],'molecular')
 spec=json.loads((root/'partition.json').read_text()) if j['kind']=='low_data' else None
 run(j['fold'],j['seed'],mode,output_root=root,model_builder=lambda mode,n,pop:build_control(j['variant'],n,pop),target_drug=j['drug'] if j['kind']=='single' else None,partition_spec=spec)
 out=root/f"{j['fold']}__s{j['seed']}__{mode}";m=json.loads((out/'run_manifest.json').read_text());save(out/'run_manifest.json',m|dict(primary_family='hybrid',variant=j['variant'],job=j,source_snapshot_sha256=sha(B/'source_snapshot.tar.gz')))
def main():
 if len(sys.argv)>1 and sys.argv[1]=='--worker':return worker(int(sys.argv[2]))
 plan=register();done=[];start=time.time();save(B/'status.json',dict(status='running',completed=0,total=len(plan['jobs'])))
 def launch(pair):
  i,j=pair;mode={'chemical_only':'chemical_only','pk_only':'pk_only','both_relations':'molecular_relation'}.get(j['variant'],'molecular');out=R/j['root']/f"{j['fold']}__s{j['seed']}__{mode}"
  if out.exists():return dict(index=i,status=json.loads((out/'run_manifest.json').read_text())['status'],reused=True)
  log=B/'logs'/f'{i:04d}.log';log.parent.mkdir(exist_ok=True)
  with log.open('w') as f:r=subprocess.run([sys.executable,__file__,'--worker',str(i)],stdout=f,stderr=subprocess.STDOUT,env=os.environ|dict(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
  return dict(index=i,status='completed' if r.returncode==0 else 'failed',returncode=r.returncode)
 with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
  for f in as_completed([pool.submit(launch,j) for j in enumerate(plan['jobs'])]):
   done.append(f.result());save(B/'status.json',dict(status='running',completed=sum(x['status']=='completed' for x in done),failed=sum(x['status']=='failed' for x in done),total=len(plan['jobs']),elapsed=time.time()-start,results=done));print(len(done),done[-1],flush=True)
 save(B/'status.json',dict(status='completed' if all(x['status']=='completed' for x in done) else 'failed',completed=sum(x['status']=='completed' for x in done),total=len(plan['jobs']),elapsed=time.time()-start,results=done))
if __name__=='__main__':main()
