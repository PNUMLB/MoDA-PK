"""Registered additional experiments; checkpoints and test results never select scope."""
from pathlib import Path
import sys,json,shutil,subprocess,os,time,argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'src'),str(R/'scripts')]
B=R/'results/runs/COMPLETION_SUITE/20261002'
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2))
def register():
 B.mkdir(parents=True,exist_ok=True);c=json.loads((R/'results/runs/TRACEABLE/evaluation_20260925/plan.json').read_text());root=B/'budget240';root.mkdir(exist_ok=True);write(root/'plan.json',c|dict(epochs=240,seeds=list(range(2,11)),purpose='Equal doubled budget sensitivity; seed1 previously completed; no assessment-driven stopping',additional_inference_k=[3]));shutil.copytree(R/'results/runs/TRACEABLE/evaluation_20260925/calibration',root/'calibration',dirs_exist_ok=True)
 jobs=[dict(kind='budget',fold=f,seed=s,mode=m) for f in c['folds'] for s in range(2,11) for m in ['unipk','deepct','dcm','molecular_relation']]
 write(B/'registered_plan.json',dict(status='registered',budget_jobs=jobs,budget_seed1_reused='results/runs/COMPARISON_REFRESH/20260929/budget240',seeds=list(range(1,11)),other_tasks=['verify_resampling','verify_molecular_encoders','history_encoder_controls','synthetic_augmentation','synthetic_positive_control','failure_stage_diagnostics'],selection='Validation only, earliest best checkpoint; all new results separate from primary tables',workers=24));return jobs

def worker(j):
 from mhap_pk.training.traceable_panel import run
 from complete_comparison_refresh import build
 run(j['fold'],j['seed'],j['mode'],output_root=B/'budget240',model_builder=build)
def launch():
 jobs=register();logs=B/'logs';logs.mkdir(exist_ok=True);start=time.time();done=[]
 def go(i,j):
  out=B/'budget240'/f"{j['fold']}__s{j['seed']}__{j['mode']}"
  if (out/'run_manifest.json').exists():
   status=json.loads((out/'run_manifest.json').read_text())['status']
   if status=='completed':return j|dict(status='completed',reused=True)
   raise RuntimeError(f'Existing incomplete run requires explicit recovery: {out}')
  with (logs/f'budget_{i:03d}.log').open('w') as f:r=subprocess.run([sys.executable,__file__,'--job',str(i)],cwd=R,stdout=f,stderr=subprocess.STDOUT,env=os.environ|dict(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
  return j|dict(status='completed' if r.returncode==0 else 'failed',returncode=r.returncode)
 with ThreadPoolExecutor(max_workers=24) as pool:
  futs=[pool.submit(go,i,j) for i,j in enumerate(jobs)]
  for f in as_completed(futs):
   done.append(f.result());write(B/'budget_status.json',dict(status='running',completed=sum(x['status']=='completed' for x in done),failed=sum(x['status']=='failed' for x in done),total=len(jobs),elapsed=time.time()-start,jobs=done));print(len(done),len(jobs),done[-1],flush=True)
 write(B/'budget_status.json',dict(status='completed' if all(x['status']=='completed' for x in done) else 'failed',jobs=done,elapsed=time.time()-start))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--job',type=int);x=a.parse_args()
 if x.job is None:launch()
 else:worker(json.loads((B/'registered_plan.json').read_text())['budget_jobs'][x.job])
