"""Private, minimal support/dose inputs for reproducible manuscript curves."""
from pathlib import Path
import sys,json,hashlib
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from mhap_pk.data.traceable import load_profiles
S=R/'results/summaries/hybrid_primary_20261002';ps,_,_=load_profiles();out=[]
for p in ps:
 out.append({k:str(p[k]) for k in ['source_id','drug_id','participant_id','profile_id']}|dict(events=p['events'].tolist(),support_k2=p['episodes'][2].support[['observation_id','time','target_log_dv']].to_dict('records'),support_k4=p['episodes'][3].support[['observation_id','time','target_log_dv']].to_dict('records')))
(S/'plot_support_doses_LOCAL_ONLY.json').write_text(json.dumps(dict(dataset='source_traceable_20260925',units=dict(time='h',amount='umol',concentration='umol/L',stored_concentration='natural log'),event_columns=['time','amount','route','duration'],route_codes={'0':'oral','1':'IV bolus','2':'IV infusion'},profiles=out),indent=2,allow_nan=False))
print('Saved',len(out),'profiles: support and dosing inputs; private review use only')
