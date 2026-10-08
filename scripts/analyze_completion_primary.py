"""Drug-wise diagnostics for the exact pooled primary panel, with no model fitting."""
from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from mhap_pk.evaluation.drug_metrics import balanced_metrics
O=R/'results/summaries/completion_suite_20261002';I=R/'results/summaries/ten_seeds_20261001/canonical_predictions.parquet'
MODS=['molecular_relation','poppk_vi_map','dcm','unipk','deepct'];N=dict(zip(MODS,['MoDE-PK (Ours)','PopPK + MAP','DCM','Uni-PK','DeepCt']));C=dict(zip(MODS,['#b2182b','#2166ac','#8856a7','#1b7837','#d95f02']))
DN=lambda s:'R-lipoic acid' if s=='r_lipoic_acid' else s.replace('_',' ').capitalize()
p=pd.read_parquet(I);p=p[(p.k==2)&(p.protocol=='first_k')&p.condition.isin(MODS)].copy();keys=['source_id','drug_id','participant_id','profile_id','observation_id'];assert not p.duplicated(['track','condition','seed']+keys).any()
for (track,seed),g in p.groupby(['track','seed']):
 refs=[h.set_index(keys)[['query_time','y_true_log','support_end_time']].sort_index() for _,h in g.groupby('condition')]
 for ref in refs[1:]:pd.testing.assert_frame_equal(refs[0],ref)
assert p.groupby(['track','condition']).seed.nunique().eq(10).all()
p['horizon_h']=p.query_time-p.support_end_time;assert (p.horizon_h>0).all();bins=['0–1','1–4','4–12','12–24','>24'];p['horizon_bin']=pd.cut(p.horizon_h,[0,1,4,12,24,np.inf],labels=bins)
def metricframe(keys):
 return pd.DataFrame([dict(zip(keys,k))|balanced_metrics(g.y_true_log,g.y_pred_log,g.source_id+'::'+g.participant_id) for k,g in p.groupby(keys,observed=True)])
d=metricframe(['track','drug_id','condition','seed']);h=metricframe(['track','drug_id','condition','seed','horizon_bin']);d.to_csv(O/'primary_drug_metrics_by_seed.csv',index=False);h.to_csv(O/'primary_horizon_metrics_by_seed.csv',index=False)
fields=['log_rmse','log_mae','within_twofold','signed_log_bias','linear_rmse','linear_mae','r2_log','r2_linear'];s=d.groupby(['track','drug_id','condition'])[fields].agg(['mean','std','count']);s.columns=['_'.join(x) for x in s.columns];s=s.reset_index();s.to_csv(O/'primary_drug_metrics_summary.csv',index=False)
hs=h.groupby(['track','drug_id','condition','horizon_bin'],observed=True).log_rmse.agg(['mean','std']).reset_index();hs.to_csv(O/'primary_horizon_summary.csv',index=False)
scopes=['Known-drug','Joint drug–source'];drugs=sorted(p.drug_id.unique())
with PdfPages(O/'Primary_drug_prediction_atlas.pdf') as pdf:
 for track in scopes:
  for drug in drugs:
   g=p[(p.track==track)&(p.drug_id==drug)&(p.seed==1)];lo=min(g.y_true_log.min(),g.y_pred_log.min());hi=max(g.y_true_log.max(),g.y_pred_log.max());pad=max((hi-lo)*.05,.1);vmin=np.log1p(g.horizon_h.min());vmax=np.log1p(g.horizon_h.max());fig,axs=plt.subplots(2,3,figsize=(12,7.8))
   for ax,mod in zip(axs.flat,MODS):
    q=g[g.condition==mod]
    if q.empty:ax.axis('off');continue
    im=ax.scatter(q.y_true_log,q.y_pred_log,c=np.log1p(q.horizon_h),vmin=vmin,vmax=vmax,cmap='viridis',s=15,alpha=.7);ax.plot([lo-pad,hi+pad],[lo-pad,hi+pad],'k--',lw=.8);ax.set(xlim=(lo-pad,hi+pad),ylim=(lo-pad,hi+pad),xlabel='Observed log concentration',ylabel='Predicted log concentration',title=N[mod]);ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.15)
   axs.flat[-1].axis('off');axs.flat[-1].text(.04,.55,'Identical queries and axes\nNatural log of µmol/L\nNo averaging across seeds\nColor: log(1 + horizon / h)',transform=axs.flat[-1].transAxes,fontsize=11);fig.colorbar(im,ax=axs.flat[-1],fraction=.3,shrink=.75)
   fig.suptitle(f'{DN(drug)} | {track} | k=2 | seed 1');fig.tight_layout(rect=(0,0,1,.96));pdf.savefig(fig);plt.close(fig)
with PdfPages(O/'Primary_drug_horizon_atlas.pdf') as pdf:
 for drug in drugs:
  fig,axs=plt.subplots(1,2,figsize=(12,4.6))
  for ax,track in zip(axs,scopes):
   for mod in MODS:
    t=hs[(hs.drug_id==drug)&(hs.track==track)&(hs.condition==mod)].set_index('horizon_bin').reindex(bins)
    if t['mean'].notna().any():ax.errorbar(range(5),t['mean'],yerr=t['std'],color=C[mod],label=N[mod],marker='o',capsize=2,alpha=.85)
   counts=h[(h.drug_id==drug)&(h.track==track)&(h.condition=='molecular_relation')&(h.seed==1)].set_index('horizon_bin').reindex(bins)
   labels=[f'{b}\n'+(f'{int(counts.loc[b,"participants"])} / {int(counts.loc[b,"queries"])}' if pd.notna(counts.loc[b,'queries']) else '—') for b in bins]
   ax.set_xticks(range(5),labels);ax.set(xlabel='Horizon (h); participants / future measurements',ylabel='Participant-balanced log-RMSE',title=track);ax.grid(alpha=.15)
  handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=5,fontsize=8,frameon=False);fig.suptitle(DN(drug)+' | mean ± SD of 10 seed scores');fig.tight_layout(rect=(0,.07,1,.96));pdf.savefig(fig);plt.close(fig)
# Keep the primary pooled panel distinct from the archived single-drug metric tables.
rows=[]
for drug in drugs:
 for mode in MODS:
  g=s[(s.track=='Known-drug')&(s.drug_id==drug)&(s.condition==mode)].iloc[0]
  def fmt(metric):
   v=g[metric+'_mean']
   if pd.isna(v):return '--'
   if metric=='linear_rmse' and 0<abs(v)<.001:
    mantissa,exponent=f'{v:.2e}'.split('e');return '$'+mantissa+r'\times10^{'+str(int(exponent))+'}$'
   return f'{v:.3f}'
  rows.append(DN(drug)+' & '+N[mode]+' & '+' & '.join(fmt(f) for f in ['linear_rmse','r2_linear','r2_log','log_mae','within_twofold'])+r'\\')
tex=r'''\begin{longtable}{llrrrrr}
\caption{Complementary drug-level metrics for the pooled Known-drug primary panel, $k=2$. Means of ten seed scores; participants receive equal total measurement weight within each drug. Concentration RMSE is in $\mu$mol/L; compare it within a drug. Undefined $R^2$ values are shown as --. Within-twofold is a percentage, not predictive-interval coverage.}\label{tab:completion-primary-metrics}\\
\toprule Drug & Model & RMSE$_C$ & $R^2_C$ & $R^2_{\log C}$ & log-MAE & Twofold \\\midrule\endfirsthead
\toprule Drug & Model & RMSE$_C$ & $R^2_C$ & $R^2_{\log C}$ & log-MAE & Twofold \\\midrule\endhead
'''+ '\n'.join(rows)+r'\bottomrule\end{longtable}'
(O/'Primary_known_additional.tex').write_text(tex)
(O/'primary_diagnostics_manifest.json').write_text(json.dumps(dict(status='completed',input=str(I.relative_to(R)),input_sha256=hashlib.sha256(I.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),seeds=list(range(1,11)),k=2,scope='Original pooled primary neural models, drug-specific PopPK; separate from single-drug neural benchmark',new_fits=0,query_checks=True,files={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in O.glob('Primary_*')}),indent=2))
print('Primary pooled drug, horizon, and observation-prediction diagnostics complete')
