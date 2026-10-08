"""Exploratory bias-isolation control; preserve the original perturbations and noise."""
from pathlib import Path
import sys,json,subprocess,os,time
from concurrent.futures import ThreadPoolExecutor,as_completed
import torch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'));B=R/'results/runs/COMPLETION_SUITE/20261002';V=B/'residual_centered';V.mkdir(exist_ok=True)
def worker(j):
 from mhap_pk.training.traceable_panel import run
 from mhap_pk.training.centered_augmentation import generate
 def augmentation(ps,meta,tr,b,pop,seed,out):
  source=B/'synthetic'/f"{j['fold']}__s{seed}__molecular_relation"
  return generate(ps,meta,tr,b,pop,seed,out,source)
 run(j['fold'],j['seed'],'molecular_relation',output_root=V,training_augmentation=augmentation)
def main():
 import shutil
 cfg=json.loads((B/'synthetic/plan.json').read_text());cfg.update(variant='residual_centered',augmentation='Observed training concentration plus original PK perturbation delta plus original noise; exact same generated perturbations as synthetic condition',purpose='Isolate generator bias identified from training-only teacher diagnostics; exploratory, no primary model replacement')
 (V/'plan.json').write_text(json.dumps(cfg,indent=2));shutil.copytree(B/'synthetic/calibration',V/'calibration',dirs_exist_ok=True)
 jobs=[dict(variant='residual_centered',fold=f,seed=s) for f in cfg['folds'] for s in range(1,11)];(B/'residual_augmentation_plan.json').write_text(json.dumps(dict(jobs=jobs,registered_at=time.time(),workers=12,scope=cfg['purpose']),indent=2));start=time.time()
 def go(j):
  out=V/f"{j['fold']}__s{j['seed']}__molecular_relation"
  if out.exists():
   assert json.loads((out/'run_manifest.json').read_text())['status']=='completed';return j|dict(status='completed',reused=True)
  with (V/f"{j['fold']}__s{j['seed']}.log").open('w') as f:r=subprocess.run([sys.executable,__file__,j['fold'],str(j['seed'])],cwd=R,stdout=f,stderr=subprocess.STDOUT,env=os.environ|dict(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
  return j|dict(status='completed' if r.returncode==0 else 'failed',returncode=r.returncode)
 rows=[]
 with ThreadPoolExecutor(max_workers=12) as pool:
  for f in as_completed([pool.submit(go,j) for j in jobs]):
   rows.append(f.result());(B/'residual_augmentation_status.json').write_text(json.dumps(dict(status='completed' if len(rows)==100 and all(x['status']=='completed' for x in rows) else 'running',jobs=rows,elapsed=time.time()-start),indent=2));print(len(rows),rows[-1],flush=True)
 if any(x['status']!='completed' for x in rows):raise RuntimeError('A centered augmentation fit failed')
if __name__=='__main__':
 if len(sys.argv)>1:worker(dict(fold=sys.argv[1],seed=int(sys.argv[2])))
 else:main()
