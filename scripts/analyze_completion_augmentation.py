"""Training-only teacher fit diagnostics; no assessment labels or tuning."""
from pathlib import Path
import sys,json
import torch,pandas as pd
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from mhap_pk.data.traceable import load_profiles,indices_for_split
from mhap_pk.data import make_batch
from mhap_pk.relational import floor_curve
from mhap_pk.evaluation.drug_metrics import balanced_metrics
B=R/'results/runs/COMPLETION_SUITE/20261002/synthetic';O=R/'results/summaries/completion_suite_20261002';torch.set_num_threads(1);ps,meta,_=load_profiles();rows=[];checks=[]
for fold in json.loads((B/'plan.json').read_text())['folds']:
 p=B/f'{fold}__s1__molecular_relation';tr,_,_=indices_for_split(meta,fold);info=json.loads((p/'synthetic_generator.json').read_text());assert info['fitting_indices']==tr.tolist();a=torch.load(p/'synthetic_training_LOCAL_ONLY.pt',weights_only=True)['teacher_parameters']
 for seed in range(1,11):
  q=B/f'{fold}__s{seed}__molecular_relation';other=torch.load(q/'synthetic_training_LOCAL_ONLY.pt',weights_only=True)['teacher_parameters'];assert torch.equal(a,other)
  checks.append(dict(fold=fold,seed=seed,same_teacher_across_seeds=True))
 b=make_batch(ps,meta,tr,torch.zeros(len(meta),512,dtype=torch.double),torch.zeros(len(meta),dtype=torch.long))
 with torch.no_grad():pred=floor_curve(a,b.events,b.query_t)
 for drug in sorted(meta.iloc[tr].drug_id.unique()):
  indices=[i for i,idx in enumerate(tr) if meta.iloc[idx].drug_id==drug];ys=[];preds=[];participants=[]
  for i in indices:
   mask=b.query_mask[i];ys.extend(b.query_y[i,mask].tolist());preds.extend(pred[i,mask].tolist());participants.extend([str(meta.iloc[tr[i]].source_id)+'::'+str(meta.iloc[tr[i]].participant_id)]*int(mask.sum()))
  rows.append(dict(fold=fold,drug_id=drug,**info['statistics'][drug])|balanced_metrics(ys,preds,participants))
pd.DataFrame(rows).to_csv(O/'augmentation_teacher_fit.csv',index=False);pd.DataFrame(checks).to_csv(O/'augmentation_teacher_checks.csv',index=False)
print(pd.DataFrame(rows).query("fold=='known'")[['drug_id','log_rmse','signed_log_bias','noise_sd','parameter_sd']].to_string(index=False))
