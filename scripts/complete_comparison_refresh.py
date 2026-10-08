"""Registered correction and budget sensitivity. Existing runs remain immutable."""
from pathlib import Path
import sys,json,os,subprocess,hashlib,shutil,tarfile,time,argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
B=R/'results/runs/COMPARISON_REFRESH/20260929';OLD=R/'results/runs/TRACEABLE/evaluation_20260925'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def build(mode,n,pop):
 from mhap_pk.baselines.low_data import build as original
 import torch
 if mode!='deepct':return original(mode,n,pop)
 from mhap_pk.literature_adapters import DeepCtAdapter
 m=DeepCtAdapter(concentration_convention='amount_over_volume').double()
 with torch.no_grad():m.prior.base.copy_(pop);m.initialize(pop)
 return m

def worker(phase,fold,seed,mode):
 from mhap_pk.training.traceable_panel import run
 run(fold,seed,mode,output_root=B/phase,model_builder=build)

def main():
 cfg=json.loads((OLD/'plan.json').read_text());B.mkdir(parents=True,exist_ok=True)
 configs={'corrected120':cfg|dict(epochs=120,modes=['deepct'],reason='Correct oral amount-to-concentration conversion; all seeds/folds, no other mechanism change'),
 'budget240':cfg|dict(epochs=240,seeds=[1],modes=['molecular_relation','dcm','deepct','unipk'],reason='Predeclared seed1 all folds/all main neural methods equal doubled epoch ceiling; validation selection, diagnostic not final model reselection')}
 jobs=[]
 for phase,c in configs.items():
  p=B/phase;p.mkdir(exist_ok=True);f=p/'plan.json'
  if f.exists():assert json.loads(f.read_text())==c
  else:f.write_text(json.dumps(c,indent=2));shutil.copytree(OLD/'calibration',p/'calibration')
  jobs += [dict(phase=phase,fold=f,seed=s,mode=m) for f in c['folds'] for s in c['seeds'] for m in c['modes']]
 if not (B/'registered_plan.json').exists():
  (B/'registered_plan.json').write_text(json.dumps(dict(registered_at=time.time(),jobs=jobs,selection='validation only; primary corrected120 retains original budget; budget240 is sensitivity',dataset=cfg['dataset_id'],split_sha256=cfg['split_sha256']),indent=2))
  with tarfile.open(B/'source_snapshot.tar.gz','w:gz') as t:
   for p in list((R/'src/mhap_pk').rglob('*.py'))+[Path(__file__)]:t.add(p,arcname=str(p.relative_to(R)))
  (B/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
 def launch(j):
  p=B/j['phase']/f"{j['fold']}__s{j['seed']}__{j['mode']}"
  if p.exists():return j|dict(status=json.loads((p/'run_manifest.json').read_text())['status'],reused=True)
  log=B/(j['phase']+'__'+p.name+'.log')
  with log.open('w') as h:rc=subprocess.call([sys.executable,__file__,'--worker',j['phase'],j['fold'],str(j['seed']),j['mode']],stdout=h,stderr=subprocess.STDOUT,env=os.environ|dict(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
  return j|dict(status='completed' if rc==0 else 'failed',exit_code=rc)
 rows=[]
 # Interleave the slow dynamics with faster independent jobs.
 jobs.sort(key=lambda j:(j['mode']!='unipk',j['fold'],j['seed']))
 with ThreadPoolExecutor(max_workers=24) as pool:
  for future in as_completed([pool.submit(launch,j) for j in jobs]):
   rows.append(future.result());(B/'progress.json').write_text(json.dumps(rows,indent=2));print(len(rows),'/',len(jobs),rows[-1],flush=True)
 (B/'status.json').write_text(json.dumps(dict(status='completed' if all(j['status']=='completed' for j in rows) else 'completed_with_failures',jobs=rows),indent=2))
if __name__=='__main__':
 if '--worker' in sys.argv:worker(sys.argv[2],sys.argv[3],int(sys.argv[4]),sys.argv[5])
 else:main()
