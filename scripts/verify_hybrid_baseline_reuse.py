"""Verify raw lineage of unchanged single-drug, resampling and budget baselines."""
from pathlib import Path
import json,hashlib,sys
import pandas as pd,numpy as np
R=Path(__file__).resolve().parents[1];S=R/'results/summaries/hybrid_primary_20261002';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();rows=[]
inputs=[('primary','ten_seeds_20261001/canonical_predictions.parquet','condition','run_id',['dcm','deepct','unipk','poppk_vi_map']),('single','training_scope_extension_20261001/single_drug_predictions.parquet','condition','run_id',['dcm','deepct','unipk','poppk_vi_map']),('resampling','resampling_20261002/canonical_predictions.parquet','model','run_path',['dcm','deepct','unipk','poppk_vi_map']),('budget','completion_suite_20261002/canonical_predictions_LOCAL_ONLY.parquet','model','run_id',['dcm_240','unipk_240','deepct_240'])]
if '--resume-budget' in sys.argv:
 rows=pd.read_csv(S/'baseline_reuse_audit_in_progress.csv').to_dict('records');assert set(x['family'] for x in rows)=={'primary','single','resampling'}
 for fam,file,_,_,_ in inputs[:3]:
  h=sha(R/'results/summaries'/file);assert all(x['canonical_parent_sha256']==h for x in rows if x['family']==fam)
 inputs=inputs[-1:]
for family,file,col,pathcol,models in inputs:
 filters=[(col,'in',models),('protocol','==','first_k')]+([('training_scope','==','pooled')] if family=='resampling' else [])
 src=R/'results/summaries'/file;p=pd.read_parquet(src,filters=filters);parent_hash=sha(src)
 for rid,g in p.groupby(pathcol):
  folder=R/rid if '/' in rid else R/('results/runs/TRACEABLE_POPPK/evaluation_20260925' if rid.startswith('full__s') else 'results/runs/TRACEABLE/evaluation_20260925')/rid;raw=folder/'predictions.parquet';m=json.loads((folder/'run_manifest.json').read_text());assert m['status']=='completed';assert sha(raw)==m['prediction_sha256']
  a=pd.read_parquet(raw).rename(columns={'y_true_lnc':'y_true_log','y_pred_lnc':'y_pred_log'});a=a[a.protocol=='first_k'];extra_hash=None
  if family=='budget' and int(g.seed.iloc[0])==1 and 3 not in set(a.k):
   extra=R/'results/runs/COMPLETION_SUITE/20261002/budget_seed1_k3_verified'/f'{m["case"]}__{m["condition"]}.parquet';em=json.loads((extra.parent/'manifest.json').read_text());extra_hash=sha(extra);assert extra_hash==em['files'][extra.name];a=pd.concat([a,pd.read_parquet(extra)],ignore_index=True)
  keys=['seed','k','observation_id'];a=a.set_index(keys);gg=g.set_index(keys);assert a.index.is_unique and gg.index.is_unique
  assert len(gg.index.difference(a.index))==0,(family,rid,'missing raw query')
  for c in ['query_time','y_true_log','y_pred_log']:assert np.array_equal(a.loc[gg.index,c],gg[c]),(family,rid,c)
  checkpoint={}
  if (folder/'selected.pt').exists():
   sel=json.loads((folder/'selection.json').read_text());assert sha(folder/'selected.pt')==sel['checkpoint_sha256'];checkpoint['selected.pt']=sel['checkpoint_sha256']
  else:
   for cp in folder.glob('*_selected.pt'):
    checkpoint[cp.name]=sha(cp)
    if (folder/'selection.json').exists():
     selections=json.loads((folder/'selection.json').read_text());assert selections[cp.name.replace('_selected.pt','')]['checkpoint_sha256']==sha(cp)
  if not checkpoint and 'parent' in m:
   parent=Path(m['parent']);cp=parent/(m['drug']+'_selected.pt') if m.get('drug') else parent/'selected.pt';assert sha(cp)==m['checkpoint_sha256'];checkpoint[str(cp.relative_to(R))]=sha(cp)
  assert checkpoint,(family,rid,'no checkpoint')
  rows.append(dict(family=family,model=g[col].iloc[0],run_id=rid,queries=len(g),raw_prediction_sha256=sha(raw),checkpoint_hashes=json.dumps(checkpoint),canonical_parent_sha256=parent_hash,additional_k3_sha256=extra_hash))
 pd.DataFrame(rows).to_csv(S/'baseline_reuse_audit_in_progress.csv',index=False)
 print(family,'verified',len(p),flush=True)
pd.DataFrame(rows).to_csv(S/'baseline_reuse_audit.csv',index=False);(S/'baseline_reuse_manifest.json').write_text(json.dumps(dict(status='completed',verified_runs=len(rows),all_raw_predictions_exact=True),indent=2));print('Complete',len(rows))
