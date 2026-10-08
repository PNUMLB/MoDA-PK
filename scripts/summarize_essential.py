"""Aggregate Essential E01/E02 from recorded predictions; never selects a primary model."""
from pathlib import Path
import sys,json,hashlib
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from mhap_pk.evaluation.quality_metrics import extended_metrics
S=R/'results/summaries/essential_20261002';B=R/'results/runs/ESSENTIAL_20261002'
p=pd.read_parquet(R/'results/summaries/ten_seeds_20261001/canonical_predictions.parquet');p=p[(p.track=='Known-drug')&(p.k==2)&(p.protocol=='first_k')&(p.seed<=5)]
frames=[];sels=[];refs=[];checks=[]
fig,axs=plt.subplots(1,5,figsize=(15,3),sharey=True)
for seed in range(1,6):
 f=B/f'E01__s{seed}';frames.append(pd.read_csv(f/'metrics.csv'));sel=json.loads((f/'selection.json').read_text());sels.append(dict(seed=seed,V0_epoch=sel['V0']['epoch'],V1_epoch=sel['V1']['epoch'],changed=sel['V0']['epoch']!=sel['V1']['epoch'],V0_checkpoint_sha256=sel['V0']['checkpoint_sha256'],V1_checkpoint_sha256=sel['V1']['checkpoint_sha256']))
 old=p[(p.seed==seed)&(p.condition=='molecular_relation')];new=pd.read_parquet(f/'predictions.parquet').query("selection_rule=='V0'");keys=['source_id','drug_id','participant_id','profile_id','observation_id'];old=old.copy();new=new.copy()
 for k in keys:old[k]=old[k].astype(str);new[k]=new[k].astype(str)
 a=old.merge(new,on=keys,validate='one_to_one',suffixes=('_old','_new'));assert len(a)==len(old)==len(new)
 diff=float(abs(a.y_pred_log_old-a.y_pred_log_new).max());assert diff<1e-12;checks.append(dict(seed=seed,queries=len(a),max_prediction_difference=diff))
 base=p[(p.seed==seed)&(p.condition=='poppk_vi_map')]
 for scope,g in [('all',base),('without_S5',base[base.drug_id!='fluorouracil'])]:refs.append(dict(seed=seed,rule='PopPK fixed',assessment=scope,**extended_metrics(g)))
 v=pd.read_csv(f/'validation.csv')
 for rule,g in v.groupby('rule'):axs[seed-1].plot(g.epoch,g.validation_sb_hrmse,label=rule);axs[seed-1].scatter(sel[rule]['epoch'],sel[rule]['validation_sb_hrmse'],s=35)
 axs[seed-1].set(title=f'Seed {seed}',xlabel='Epoch');axs[seed-1].grid(alpha=.2)
axs[0].set_ylabel('Validation SB-hRMSE');axs[-1].legend();fig.tight_layout();fig.savefig(S/'E01_validation_selection.pdf');fig.savefig(S/'E01_validation_selection.png',dpi=180);plt.close(fig)
m=pd.concat(frames,ignore_index=True);ref=pd.DataFrame(refs);allm=pd.concat([m,ref],ignore_index=True);metrics=['sb_hrmse','log_rmse','log_mae','within_twofold_pct','signed_bias','p90_abs_log_error'];s=allm.groupby(['rule','assessment'])[metrics].agg(['mean','std']);s.columns=['_'.join(x) for x in s.columns];s=s.reset_index()
m.to_csv(S/'E01_metrics_by_seed.csv',index=False);ref.to_csv(S/'E01_PopPK_fixed_reference.csv',index=False);s.to_csv(S/'E01_2x2_summary.csv',index=False);pd.DataFrame(sels).to_csv(S/'E01_selection_summary.csv',index=False);pd.DataFrame(checks).to_csv(S/'E01_prediction_replay_checks.csv',index=False)
a=pd.read_csv(S/'E02_training_audit.csv');a['initial_selected']=a.chosen_epoch==0;training=a.groupby('condition').agg(fits=('seed','size'),initial_selected=('initial_selected','sum'),ceiling_selected=('selected_at_max','sum'));training.to_csv(S/'E02_learning_summary.csv')
with pd.ExcelWriter(S/'Essential_results.xlsx') as w:
 for name,d in [('E01_summary',s),('E01_seeds',m),('Selection_epochs',pd.DataFrame(sels)),('Fixed_PopPK',ref),('Replay_checks',pd.DataFrame(checks)),('Learning_summary',training.reset_index()),('Drug_metrics',pd.read_csv(S/'E02_drug_metrics_by_seed.csv')),('Table_checks',pd.read_csv(S/'E02_table_checks.csv')),('Mavoglurant',pd.read_csv(S/'E02_mavoglurant_ties.csv')),('Floor_probe',pd.read_csv(S/'E02_floor_probe.csv'))]:d.to_excel(w,sheet_name=name,index=False)
print(s.to_string(index=False))
