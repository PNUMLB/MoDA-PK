from pathlib import Path
import pandas as pd
S=Path(__file__).resolve().parents[1]/'results/summaries/molecular_identity_20261002'
p=pd.read_parquet(S/'canonical_predictions.parquet');p['sq']=(p.y_pred_log-p.y_true_log)**2
keys=['track','condition','seed','source_id','drug_id','participant_id','profile_id']
g=p.groupby(keys+['k']).agg(mse=('sq','mean'),queries=('sq','size')).reset_index()
h=g[g.k==2].merge(g[g.k==0],on=keys,suffixes=('','_k0'),validate='one_to_one');h['delta_mse']=h.mse-h.mse_k0;h['worsened']=h.delta_mse>1e-10;h.to_csv(S/'history_effect_by_profile.csv',index=False)
keys.remove('condition');pairs=[]
for ref in ['molecular','drug_id']:
 a=g[(g.k==2)&(g.condition=='molecular_identity')].merge(g[(g.k==2)&(g.condition==ref)],on=keys,suffixes=('','_reference'),validate='one_to_one');a['reference']=ref;a['delta_mse']=a.mse-a.mse_reference;a['worsened']=a.delta_mse>1e-10;pairs.append(a)
a=pd.concat(pairs,ignore_index=True);a.to_csv(S/'paired_profiles.csv',index=False)
with pd.ExcelWriter(S/'molecular_identity_results.xlsx',mode='a',engine='openpyxl',if_sheet_exists='replace') as w:
 h.to_excel(w,sheet_name='Profile history effect',index=False);a.to_excel(w,sheet_name='Paired profiles',index=False)
