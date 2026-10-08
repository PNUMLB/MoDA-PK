"""Encoder and training-only synthetic augmentation: same prior, loss and split."""
from pathlib import Path
import sys,os,json,shutil,subprocess,time
from concurrent.futures import ThreadPoolExecutor,as_completed
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'src'),str(R/'scripts')];B=R/'results/runs/COMPLETION_SUITE/20261002'
def write(p,x):p.write_text(json.dumps(x,indent=2))
def register():
 cfg=json.loads((R/'results/runs/TRACEABLE/evaluation_20260925/plan.json').read_text());jobs=[]
 for variant in ['lstm','last_mlp','deep_sets','attention','synthetic']:
  root=B/variant;root.mkdir(exist_ok=True);write(root/'plan.json',cfg|dict(variant=variant,seeds=list(range(1,11)),additional_inference_k=[3],same_pk_dynamics=True,tuning='No test-driven hyperparameter search; same 120epoch selection; encoder hidden32, attention2heads 1layer dropout0; both relations ON'))
  shutil.copytree(R/'results/runs/TRACEABLE/evaluation_20260925/calibration',root/'calibration',dirs_exist_ok=True)
  jobs.extend(dict(variant=variant,fold=f,seed=s) for f in cfg['folds'] for s in range(1,11))
 write(B/'controls_plan.json',dict(jobs=jobs,workers=24,seeds=list(range(1,11)),baseline='Existing full GRU 120epoch; exact parity test required before reuse',scope='Encoder controls preserve dose dynamics, parameter/state heads, molecular representation and relation objectives. Synthetic uses train-only generator.'))
 return jobs

def worker(j):
 import torch
 from mhap_pk.training.traceable_panel import run
 from mhap_pk.models.adapters.encoder_controls import EncoderControlPK
 from mhap_pk.baselines.low_data import build
 from mhap_pk.training.synthetic_augmentation import generate
 def builder(mode,n,pop):
  if j['variant']=='synthetic':return build(mode,n,pop)
  m=EncoderControlPK(j['variant']).double()
  with torch.no_grad():m.prior.base.copy_(pop)
  return m
 run(j['fold'],j['seed'],'molecular_relation',output_root=B/j['variant'],model_builder=builder,training_augmentation=generate if j['variant']=='synthetic' else None)

def launch():
 jobs=register();logs=B/'logs';start=time.time();done=[]
 def go(i,j):
  out=B/j['variant']/f"{j['fold']}__s{j['seed']}__molecular_relation"
  if (out/'run_manifest.json').exists():
   if json.loads((out/'run_manifest.json').read_text())['status']=='completed':return j|dict(status='completed',reused=True)
   raise RuntimeError('Incomplete existing '+str(out))
  with (logs/f'control_{i:03d}.log').open('w') as f:r=subprocess.run([sys.executable,__file__,str(i)],cwd=R,stdout=f,stderr=subprocess.STDOUT,env=os.environ|dict(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
  return j|dict(status='completed' if not r.returncode else 'failed',returncode=r.returncode)
 with ThreadPoolExecutor(max_workers=24) as ex:
  fs=[ex.submit(go,i,j) for i,j in enumerate(jobs)]
  for f in as_completed(fs):
   done.append(f.result());write(B/'controls_status.json',dict(status='running',completed=sum(x['status']=='completed' for x in done),failed=sum(x['status']=='failed' for x in done),total=len(jobs),elapsed=time.time()-start,jobs=done));print(len(done),done[-1],flush=True)
 write(B/'controls_status.json',dict(status='completed' if all(x['status']=='completed' for x in done) else 'failed',jobs=done,elapsed=time.time()-start))
if __name__=='__main__':
 if len(sys.argv)>1:worker(json.loads((B/'controls_plan.json').read_text())['jobs'][int(sys.argv[1])])
 else:launch()
