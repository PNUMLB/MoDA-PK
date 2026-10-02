"""Single-drug comparators and matched target-participant learning curves."""
from pathlib import Path
import sys,os,json,time,hashlib,subprocess,tarfile,shutil
from concurrent.futures import ThreadPoolExecutor,as_completed
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'));sys.path.insert(0,str(R/'scripts'))
B=R/'results/runs/TRAINING_SCOPE/20261001'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(x,indent=2));temp.replace(p)
def register():
 from mhap_pk.data.traceable import load_profiles,indices_for_split
 from mhap_pk.data.fitting_partitions import matched_partitions,validate_partition
 _,meta,_=load_profiles();roles=indices_for_split(meta,'known')
 cfg=json.loads((R/'configs/experiments/source_traceable_20260925.json').read_text())|dict(seeds=list(range(1,11)),additional_inference_k=[3],epochs=120)
 assert sha(R/'data/splits/source_traceable_20260925/partitions.json')==cfg['split_sha256']
 jobs=[]
 for drug in sorted(meta.drug_id.unique()):
  old=R/'results/runs/SINGLE_DRUG/20261001'/drug
  root=B/'single'/drug;root.mkdir(parents=True,exist_ok=True)
  write(root/'plan.json',cfg|dict(experiment='SINGLE_DRUG_COMPARATORS',target_drug=drug))
  (root/'calibration/known').mkdir(parents=True,exist_ok=True)
  dest=root/'calibration/known/population.json'
  if not dest.exists():shutil.copy2(old/'calibration/known/population.json',dest)
  for seed in range(1,11):
   for mode in ['unipk','deepct','dcm']:jobs.append(dict(kind='single',drug=drug,seed=seed,mode=mode,root=str(root.relative_to(R))))
  nt=meta.iloc[roles[0]].query('drug_id==@drug').participant_id.nunique()
  for cap in [n for n in [2,4,8,16] if n<nt]+[int(nt)]:
   for seed in range(1,11):
    for pooled in [False,True]:
     # Full target-only MHAP fits already exist with this exact recipe.
     if not pooled and cap==nt:continue
     scope='pooled' if pooled else 'target'
     root=B/'low_data'/drug/f'n{cap}'/f's{seed}'/scope
     spec=matched_partitions(meta,roles,drug,cap,seed,pooled);validate_partition(meta,roles,spec)
     write(root/'partition.json',spec)
     write(root/'plan.json',cfg|dict(experiment='MATCHED_LOW_DATA',target_drug=drug,target_training_participants=cap,full_target_training_participants=int(nt),validation='Fixed target-drug validation only, epochs 0..120 every 10',seed_interpretation='Coupled nested target subset sampling and optimization; not independent cohorts'))
     modes=['molecular','molecular_relation'] if pooled else ['molecular']
     if pooled and cap==2:modes+=['no_molecule','drug_id']
     jobs.append(dict(kind='low_data',drug=drug,seed=seed,cap=cap,scope=scope,modes=modes,root=str(root.relative_to(R))))
 plan=dict(status='registered',created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),jobs=jobs,new_fits=sum(1 if j['kind']=='single' else len(j['modes']) for j in jobs),workers=40,primary_k=2,all_k=[0,1,2,3,4],split_sha256=cfg['split_sha256'],design='Single-drug benchmarks and paired donor effects; same target subset and target validation/test, omitted linked target people removed from all donor data; fixed hyperparameters; no test-based selection',controls='Low-data main contrast relation-off target vs relation-off pooled; full relation model separate; n=2 no-molecule and Drug-ID donor controls; full-target-only existing fits reused',calibration='Re-estimate on exact reduced training partition; normalization and relations fit only there',model_builder='complete_comparison_refresh.build, amount_over_volume DeepCt',epoch_zero='Eligible under unchanged validation recipe; selected epoch recorded')
 if (B/'registered_plan.json').exists():
  prev=json.loads((B/'registered_plan.json').read_text());assert prev['jobs']==jobs;return prev
 write(B/'registered_plan.json',plan)
 with tarfile.open(B/'source_snapshot.tar.gz','w:gz') as tar:
  for p in list((R/'src/mhap_pk').rglob('*.py'))+[Path(__file__),R/'scripts/complete_comparison_refresh.py']:tar.add(p,arcname=str(p.relative_to(R)))
 (B/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
 write(B/'code_manifest.json',{str(p.relative_to(R)):sha(p) for p in list((R/'src/mhap_pk').rglob('*.py'))+[Path(__file__),R/'scripts/complete_comparison_refresh.py']})
 return plan

def worker(index):
 import torch
 torch.set_num_threads(1)
 from mhap_pk.training.traceable_panel import run
 from complete_comparison_refresh import build
 job=json.loads((B/'registered_plan.json').read_text())['jobs'][index];root=R/job['root']
 if job['kind']=='single':run('known',job['seed'],job['mode'],output_root=root,model_builder=build,target_drug=job['drug']);return
 from mhap_pk.data.traceable import load_profiles,indices_for_split
 from mhap_pk.data.fitting_partitions import validate_partition
 from mhap_pk.data import representations,make_batch
 from mhap_pk.training.population import fit_population
 spec=json.loads((root/'partition.json').read_text());ps,meta,_=load_profiles();tr,_,_=validate_partition(meta,indices_for_split(meta,'known'),spec)
 calpath=root/'calibration/known/population.json'
 if not calpath.exists():
  z,ids,_=representations(meta,tr,'unimol',17);b=make_batch(ps,meta,tr,z,ids)
  _,cal=fit_population(b,steps=120);write(calpath,cal|dict(training_only=True,partition_sha256=sha(root/'partition.json')))
 for mode in job['modes']:
  out=root/f"known__s{job['seed']}__{mode}"
  if out.exists():
   assert json.loads((out/'run_manifest.json').read_text())['status']=='completed';continue
  run('known',job['seed'],mode,output_root=root,model_builder=build,partition_spec=spec)

def main():
 plan=register();jobs=list(enumerate(plan['jobs']));write(B/'status.json',dict(status='running',jobs=len(jobs),new_fits=plan['new_fits'],started=time.time()))
 # Start slow single-drug Uni-PK fits early, then alternate the remaining tasks.
 jobs.sort(key=lambda x:(x[1]['kind']!='single' or x[1].get('mode')!='unipk',x[0]))
 def launch(item):
  i,j=item;root=R/j['root'];modes=[j['mode']] if j['kind']=='single' else j['modes']
  outputs=[root/f"known__s{j['seed']}__{m}"/'run_manifest.json' for m in modes]
  if all(p.exists() and json.loads(p.read_text())['status']=='completed' for p in outputs):return dict(index=i,status='completed',reused=True)
  log=B/'logs'/f'{i:04d}.log';log.parent.mkdir(exist_ok=True)
  with log.open('w') as f:rc=subprocess.call([sys.executable,__file__,'--worker',str(i)],stdout=f,stderr=subprocess.STDOUT,env=os.environ|dict(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
  return dict(index=i,status='completed' if rc==0 else 'failed',exit_code=rc)
 rows=[]
 with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
  for f in as_completed([pool.submit(launch,j) for j in jobs]):
   rows.append(f.result());write(B/'progress.json',rows);print(len(rows),'/',len(jobs),rows[-1],flush=True)
 write(B/'status.json',dict(status='completed' if all(x['status']=='completed' for x in rows) else 'completed_with_failures',jobs=len(jobs),new_fits=plan['new_fits'],finished=time.time(),results=rows))
if __name__=='__main__':
 if '--worker' in sys.argv:worker(int(sys.argv[2]))
 else:main()
