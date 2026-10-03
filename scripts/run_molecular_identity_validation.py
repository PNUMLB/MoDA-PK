"""Fixed-design molecular/identity concatenation; held-out labels never select settings."""
from pathlib import Path
import sys,os,json,time,hashlib,shutil,subprocess,tarfile
from concurrent.futures import ThreadPoolExecutor,as_completed
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'src'),str(R/'scripts')]
B=R/'results/runs/MOLECULAR_IDENTITY/20261002'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def register():
 if (B/'registered_plan.json').exists():return json.loads((B/'registered_plan.json').read_text())
 cfg=json.loads((R/'results/runs/TRACEABLE/evaluation_20260925/plan.json').read_text())|dict(seeds=list(range(1,11)),additional_inference_k=[3],experiment='MOLECULAR_IDENTITY',modes=['molecular'],input='concatenate train-standardized frozen Uni-Mol512 and learned ID512; Linear1024to32+SiLU; same PK/observer',unknown='mean training-ID embedding, not a separately fitted UNK',relations='OFF',selection='Fixed architecture; checkpoint chosen by original validation SB-hRMSE at k2; no test hyperparameter search',scope='Exploratory reused cohort; parameter counts differ and are reported')
 save(B/'plan.json',cfg);shutil.copytree(R/'results/runs/TRACEABLE/evaluation_20260925/calibration',B/'calibration')
 plan=dict(status='registered',utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),jobs=[dict(fold=f,seed=s) for f in cfg['folds'] for s in cfg['seeds']],workers=20,controls='Existing relation-OFF Uni-Mol and Drug-ID, verify raw prediction hash and fitting partition before reuse',primary_k=2,primary_comparison='Hybrid versus each single input, Known and Joint separately; report all drug effects; no automatic replacement of primary model')
 save(B/'registered_plan.json',plan)
 files=list((R/'src/mhap_pk').rglob('*.py'))+[Path(__file__)]
 save(B/'code_manifest.json',{str(p.relative_to(R)):sha(p) for p in files})
 with tarfile.open(B/'source_snapshot.tar.gz','w:gz') as t:
  for p in files:t.add(p,arcname=str(p.relative_to(R)))
 (B/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
 return plan

def main():
 if len(sys.argv)>1:
  from mhap_pk.training.traceable_panel import run
  from mhap_pk.models.priors.molecular_identity import build_joint_input
  j=json.loads((B/'registered_plan.json').read_text())['jobs'][int(sys.argv[1])]
  run(j['fold'],j['seed'],'molecular',output_root=B,model_builder=build_joint_input);return
 plan=register();done=[];start=time.time();save(B/'status.json',dict(status='running',completed=0,total=len(plan['jobs'])))
 def launch(pair):
  i,j=pair;out=B/f"{j['fold']}__s{j['seed']}__molecular"
  if out.exists():
   m=json.loads((out/'run_manifest.json').read_text());return j|dict(status=m['status'],reused=True)
  log=B/'logs'/f'{i:03d}.log';log.parent.mkdir(exist_ok=True)
  with log.open('w') as f:r=subprocess.run([sys.executable,__file__,str(i)],stdout=f,stderr=subprocess.STDOUT,env=os.environ|dict(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
  return j|dict(status='completed' if r.returncode==0 else 'failed',returncode=r.returncode)
 with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
  for f in as_completed([pool.submit(launch,j) for j in enumerate(plan['jobs'])]):
   done.append(f.result());save(B/'status.json',dict(status='running',completed=sum(x['status']=='completed' for x in done),total=len(plan['jobs']),failed=sum(x['status']=='failed' for x in done),elapsed_seconds=time.time()-start,jobs=done));print(len(done),done[-1],flush=True)
 save(B/'status.json',dict(status='completed' if all(x['status']=='completed' for x in done) else 'failed',completed=sum(x['status']=='completed' for x in done),total=len(plan['jobs']),elapsed_seconds=time.time()-start,jobs=done))
if __name__=='__main__':main()
