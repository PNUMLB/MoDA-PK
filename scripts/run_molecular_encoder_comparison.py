"""Pinned frozen encoder comparison under the unchanged traceable PK recipe."""
from pathlib import Path
import sys,os,json,time,hashlib,shutil,tarfile,subprocess
from concurrent.futures import ThreadPoolExecutor,as_completed
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
B=R/'results/runs/MOLECULAR_ENCODER/20261002'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2));q.replace(p)
def oldrun(fold,seed,mode):return R/('results/runs/TRACEABLE/evaluation_20260925' if seed<=5 else 'results/runs/TEN_SEEDS/20261001/panel')/f'{fold}__s{seed}__{mode}'
def register():
 if (B/'registered_plan.json').exists():return json.loads((B/'registered_plan.json').read_text())
 from mhap_pk.data.traceable import load_profiles
 _,meta,_=load_profiles();cfg=json.loads((R/'configs/experiments/source_traceable_20260925.json').read_text())|dict(experiment='MOLECULAR_ENCODER',seeds=list(range(1,11)),additional_inference_k=[3],relations='OFF',encoder_frozen=True,chemberta_pooling='Mean final hidden state excluding special and padding tokens',dimension='384 native ChemBERTa normalized train-only then128 zeros; unchanged512->32 projection',selection='Same120epoch recipe; validation checkpoint only; no test-driven encoder selection or tuning')
 jobs=[]
 for fold in cfg['folds']:
  for encoder in ['chemberta_mlm','chemberta_mtr']:
   root=B/'panel'/encoder/fold;(root/'calibration'/fold).mkdir(parents=True,exist_ok=True);shutil.copy2(oldrun(fold,1,'molecular')/'population.json',root/'calibration'/fold/'population.json');write(root/'plan.json',cfg|dict(representation=encoder,cache_sha256=sha(R/'data/features/molecular_comparison_20261002'/f'{encoder}.npz')))
   for seed in range(1,11):jobs.append(dict(kind='train',scope='panel',root=str(root.relative_to(R)),encoder=encoder,fold=fold,seed=seed))
  for seed in range(1,11):
   for mode in ['ecfp','no_molecule','drug_id']:
    parent=oldrun(fold,seed,mode);assert json.loads((parent/'run_manifest.json').read_text())['status']=='completed'
    jobs.append(dict(kind='inference',scope='panel',root=str((B/'control_inference'/f'{fold}__s{seed}__{mode}').relative_to(R)),encoder=mode,fold=fold,seed=seed,parent=str(parent.relative_to(R)),parent_checkpoint_sha256=sha(parent/'selected.pt')))
 for drug in sorted(meta.drug_id.unique()):
  for seed in range(1,11):
   parent=R/f'results/runs/TRAINING_SCOPE/20261001/low_data/{drug}/n2/s{seed}/pooled'
   for encoder in ['chemberta_mlm','chemberta_mtr','ecfp']:
    root=B/'low_data'/drug/f's{seed}'/encoder;(root/'calibration/known').mkdir(parents=True,exist_ok=True)
    shutil.copy2(parent/'calibration/known/population.json',root/'calibration/known/population.json');shutil.copy2(parent/'partition.json',root/'partition.json');write(root/'plan.json',cfg|dict(representation=encoder,target_drug=drug,target_n=2,partition_sha256=sha(root/'partition.json'),cache_sha256=sha(R/'data/features/molecular_comparison_20261002'/f'{encoder}.npz') if encoder.startswith('chemberta') else sha(R/'data/processed/full16_v1/embeddings/morgan.npz')))
    jobs.append(dict(kind='train',scope='low_data',root=str(root.relative_to(R)),encoder=encoder,fold='known',seed=seed,drug=drug))
 plan=dict(status='registered',utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),jobs=jobs,new_fits=sum(j['kind']=='train' for j in jobs),inference_only=sum(j['kind']=='inference' for j in jobs),workers=32,primary_k=2,k=list(range(5)),seeds=list(range(1,11)),panel=['no_molecule','drug_id','ecfp','unimol','chemberta_mlm','chemberta_mtr'],low_data='Matched target n=2 plus donors; same target validation/test and linkage exclusions',selection='Checkpoint validation only. Fixed encoders and mean pooling before fitting. No final encoder replacement based on test score.',scope='Internal reused cohort, not independent external validation; Joint means unseen PK drug-source, not guaranteed unseen encoder-pretraining structure',reuse='Existing Uni-Mol/shared and low-data no-molecule/Drug-ID fits reused; ECFP/no-molecule/Drug-ID full checkpoints re-inferred including k3',stage2='Relation loss extension deferred until encoder effects and validation choice reviewed; no test-based encoder choice')
 write(B/'registered_plan.json',plan);files=list((R/'src/mhap_pk').rglob('*.py'))+[Path(__file__),R/'scripts/cache_chemberta_comparison.py']
 write(B/'code_manifest.json',{str(p.relative_to(R)):sha(p) for p in files})
 with tarfile.open(B/'source_snapshot.tar.gz','w:gz') as t:
  for p in files:t.add(p,arcname=str(p.relative_to(R)))
 inputs=json.loads((R/'results/runs/HALF_HEAD/20261001/input_manifest.json').read_text())
 for item in inputs['files']:assert sha(R/item['path'])==item['sha256']
 for p in (R/'data/features/molecular_comparison_20261002').glob('*'):inputs['files'].append(dict(path=str(p.relative_to(R)),sha256=sha(p)))
 write(B/'input_manifest.json',inputs);(B/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True));return plan

def infer(j):
 import numpy as np,pandas as pd,torch
 from mhap_pk.data.traceable import load_profiles,indices_for_split
 from mhap_pk.data import make_batch
 from mhap_pk.data.molecular_encoders import representations
 from mhap_pk.baselines.low_data import build
 from mhap_pk.evaluation.history_protocols import initial_observation_inputs
 root=R/j['root'];root.mkdir(parents=True,exist_ok=False);parent=R/j['parent'];assert sha(parent/'selected.pt')==j['parent_checkpoint_sha256']
 ps,meta,_=load_profiles();tr,va,te=indices_for_split(meta,j['fold']);z,ids,sc=representations(meta,tr,'ecfp' if j['encoder']=='ecfp' else 'unimol')
 art=json.loads((parent/'artifacts.json').read_text());assert art['train']==tr.tolist() and art['test']==te.tolist() and art['scaler']==sc
 vectors=torch.zeros_like(z) if j['encoder'] in ['no_molecule','drug_id'] else z
 pop=torch.tensor(json.loads((parent/'population.json').read_text())['log_parameters'],dtype=torch.double);m=build('molecular' if j['encoder']=='ecfp' else j['encoder'],len(sc['fitted_drugs']),pop);m.load_state_dict(torch.load(parent/'selected.pt',weights_only=True));m.eval();batch=make_batch(ps,meta,te,vectors,ids);rows=[]
 for k in range(5):
  args=initial_observation_inputs(batch,k)
  with torch.no_grad():pred=m(*args)['mean']
  assert torch.isfinite(pred[batch.query_mask]).all() and torch.isfinite(pred[batch.query_mask].exp()).all()
  for i,idx in enumerate(batch.indices):
   p=ps[idx];end=float(args[2][i,args[4][i]].max()) if k else None
   for q in torch.where(batch.query_mask[i])[0].tolist():rows.append(dict(seed=j['seed'],fold=j['fold'],protocol='first_k',k=k,condition=j['encoder'],**{c:p[c] for c in ['source_id','drug_id','participant_id','profile_id']},observation_id=str(p['query'].observation_id.iloc[q]),query_time=float(batch.query_t[i,q]),support_end_time=end,cutoff_time=end,y_true_log=float(batch.query_y[i,q]),y_pred_log=float(pred[i,q]),fallback_flag=False))
 p=pd.DataFrame(rows);old=pd.read_parquet(parent/'predictions.parquet').rename(columns={'y_true_lnc':'y_true_log','y_pred_lnc':'y_pred_log'});old=old[old.protocol=='first_k'];ix=['seed','k','observation_id'];a=p.set_index(ix);b=old.set_index(ix);assert np.array_equal(a.loc[b.index,['y_true_log','y_pred_log']],b[['y_true_log','y_pred_log']])
 p.to_parquet(root/'predictions.parquet',index=False);write(root/'config.resolved.json',j|dict(inference_only=True,k=list(range(5))));write(root/'run_manifest.json',dict(status='completed',prediction_sha256=sha(root/'predictions.parquet'),parent=j['parent'],parent_checkpoint_sha256=j['parent_checkpoint_sha256'],exact_parent_predictions=True,new_k3=j['seed']<=5));print('INFER',j['fold'],j['seed'],j['encoder'],flush=True)
def worker(index):
 import torch
 torch.set_num_threads(1);j=json.loads((B/'registered_plan.json').read_text())['jobs'][index]
 if j['kind']=='inference':return infer(j)
 from mhap_pk.training.feature_panel import run
 root=R/j['root'];spec=json.loads((root/'partition.json').read_text()) if j['scope']=='low_data' else None
 run(j['fold'],j['seed'],root,j['encoder'],spec)
def main():
 plan=register();write(B/'status.json',dict(status='running',new_fits=plan['new_fits'],inference_only=plan['inference_only']))
 def launch(pair):
  i,j=pair;out=R/j['root'] if j['kind']=='inference' else R/j['root']/f"{j['fold']}__s{j['seed']}__molecular"
  if out.exists():return dict(index=i,status=json.loads((out/'run_manifest.json').read_text())['status'],reused=True)
  log=B/'logs'/f'{i:03d}.log';log.parent.mkdir(exist_ok=True)
  with log.open('w') as f:rc=subprocess.call([sys.executable,__file__,'--worker',str(i)],stdout=f,stderr=subprocess.STDOUT,env=os.environ|dict(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
  return dict(index=i,status='completed' if rc==0 else 'failed',exit_code=rc)
 rows=[]
 with ThreadPoolExecutor(max_workers=32) as pool:
  for f in as_completed([pool.submit(launch,j) for j in enumerate(plan['jobs'])]):rows.append(f.result());write(B/'progress.json',rows);print(len(rows),len(plan['jobs']),rows[-1],flush=True)
 write(B/'status.json',dict(status='completed' if all(x['status']=='completed' for x in rows) else 'completed_with_failures',new_fits=plan['new_fits'],inference_only=plan['inference_only'],jobs=rows))
if __name__=='__main__':
 if '--worker' in sys.argv:worker(int(sys.argv[2]))
 elif '--register' in sys.argv:register()
 else:main()
