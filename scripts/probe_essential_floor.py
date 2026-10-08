from pathlib import Path
import sys,json
import pandas as pd,numpy as np,torch
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'src'),str(R/'scripts')]
from mhap_pk.data.traceable import load_profiles,indices_for_split
from mhap_pk.data import representations,make_batch
from mhap_pk.training.run_io import inputs
from complete_comparison_refresh import build
S=R/'results/summaries/essential_20261002';torch.set_num_threads(1)
t=pd.read_csv(S/'E02_mavoglurant_ties.csv');t=t[(t.panel=='pooled')&(t.track=='Joint drug–source')&(t.seed==1)].iloc[0];ps,meta,_=load_profiles();rows=[]
for c in ['deepct','unipk']:
 p=R/t[c+'_run'];cfg=json.loads((p/'config.resolved.json').read_text());tr,va,te=indices_for_split(meta,cfg['fold']);z,ids,sc=representations(meta,tr,'unimol',17);te=te[meta.iloc[te].drug_id.to_numpy()=='mavoglurant'];b=make_batch(ps,meta,te,z,ids);pop=json.loads((p/'population.json').read_text())['log_parameters'];m=build(c,len(sc['fitted_drugs']),torch.tensor(pop));m.load_state_dict(torch.load(p/'selected.pt',weights_only=True));m.eval();args=inputs(b,2)
 with torch.no_grad():raw=m.curve(args[0],args[1],torch.cat([args[2],args[5]],1));y=m(*args)['mean']
 ns=args[2].shape[1];base=raw[:,ns:][b.query_mask].numpy();support=raw[:,:ns][args[4]].numpy()
 stored=pd.read_parquet(p/'predictions.parquet').query("k==2 and protocol=='first_k' and drug_id=='mavoglurant'");prediction=np.concatenate([y[i,b.query_mask[i]].numpy() for i in range(len(te))]);assert np.allclose(prediction,stored.y_pred_lnc.to_numpy(),atol=1e-9)
 rows.append(dict(model=c,fold=cfg['fold'],seed=1,queries=len(base),pre_correction_min=float(base.min()),pre_correction_max=float(base.max()),pre_correction_at_floor_fraction=float(np.mean(abs(base-np.log(1e-5))<1e-10)),support_at_floor_fraction=float(np.mean(abs(support-np.log(1e-5))<1e-10)),mean_absolute_output_shift=float(np.mean(abs(prediction-base))),scope='Seed 1 near-tie diagnostic only; final clamped curve before observed scalar shift, not an internal solver floor count'))
pd.DataFrame(rows).to_csv(S/'E02_floor_probe.csv',index=False);print(rows)
