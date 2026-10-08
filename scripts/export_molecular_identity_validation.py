"""Export controlled input comparison and matching Known/Joint main figures."""
from pathlib import Path
import re,json,shutil,hashlib,sys
import pandas as pd,numpy as np
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'src'),str(R/'scripts')]
S=R/'results/summaries/molecular_identity_20261002';D=R/'manuscript/MoDE-PK_EJPS_identity_validation_20261002';B=R/'manuscript/MoDE-PK_EJPS_visual_consistency_20261002'
assert json.loads((S/'verification.json').read_text())['status']=='completed'
D.mkdir(exist_ok=False);shutil.copytree(B,D,dirs_exist_ok=True)
# Shared display design from the original verified query-connection plotter.
src=(R/'scripts/restructure_paper_20261002.py').read_text()
imports=src[:src.index("D=R/'manuscript/")];setup=src[src.index("mods=['molecular_relation'"):src.index('metrics=pd.read_csv')]
data=src[src.index('p=pd.read_parquet'):src.index('fig,axs=plt.subplots(3,2')]
loop=src[src.index('for ax,c in zip(axs.flat,cases):'):src.index('handles=[Line2D')]
legend=src[src.index('handles=[Line2D'):src.index('fig.legend(handles=handles')]
ns={'__file__':str(R/'scripts/restructure_paper_20261002.py')};exec(imports+setup+data,ns)
p=ns['p'];keys=ns['keys'];cases=[c for c in ns['cases'] if c['track']=='Known-drug']
for drug in ['warfarin','xanthohumol']:
 a=p[(p.track=='Known-drug')&(p.drug_id==drug)&(p.condition=='molecular_relation')&(p.seed==1)].copy();a['sq']=(a.y_pred_log-a.y_true_log)**2
 g=a.groupby(keys).agg(mse=('sq','mean'),n=('query_time','nunique')).reset_index();e=g[g.n>=8] if (g.n>=8).any() else g[g.n==g.n.max()]
 e=e.assign(distance=abs(e.mse-e.mse.median())).sort_values(['distance']+keys)
 cases.append(dict(track='Known-drug',**e.iloc[0][keys].to_dict(),selection='Median Known seed1 MoDE-PK MSE among >=8 distinct future times or maximum available; reused identical profile for Joint, without selecting Joint error'))
# Same profile and y-axis limits across each Known/Joint pair; never rescale to conceal bias.
limits={}
for c in cases:
 g=p[(p[keys]==pd.Series({k:c[k] for k in keys})).all(axis=1)]
 assert set(g.track)=={'Known-drug','Joint drug–source'}
 support=ns['lookup'][tuple(c[k] for k in keys)]['episodes'][2].support.target_log_dv.to_numpy()
 means=g.groupby(['track','condition','observation_id']).y_pred_log.mean().to_numpy()
 vals=np.exp(np.r_[means,g.y_true_log.to_numpy(),support]);limits[c['drug_id']]=(vals.min()/1.5,vals.max()*1.5)
loop=loop.replace("ax.grid(alpha=.15);check.append", "ax.set_ylim(limits[c['drug_id']]);ax.grid(alpha=.15);check.append")
ns['limits']=limits;allchecks=[];allscores=[];allcases=[]
for track,name in [('Known-drug','Known_PK_curves'),('Joint drug–source','Joint_PK_curves')]:
 ns['cases']=[dict(c,track=track) for c in cases];ns['fig'],ns['axs']=ns['plt'].subplots(3,2,figsize=(11,10));ns['check']=[];ns['case_scores']=[]
 exec(loop+legend,ns);handles=ns['handles']
 if track!='Known-drug':handles=[h for h in handles if h.get_label()!='PopPK + MAP']
 fig=ns['fig'];fig.legend(handles=handles,loc='upper center',ncol=4,fontsize=9);fig.tight_layout(rect=(0,0,1,.93))
 for ext in ['pdf','png']:fig.savefig(S/(name+'.'+ext),dpi=170)
 ns['plt'].close(fig);shutil.copy2(S/(name+'.pdf'),D/(name+'.pdf'));allchecks.extend(ns['check']);allscores.extend(ns['case_scores']);allcases.extend(ns['cases'])
pd.DataFrame(allchecks).to_csv(S/'main_curve_checks.csv',index=False);pd.DataFrame(allscores).to_csv(S/'main_curve_ensemble_metrics.csv',index=False);pd.DataFrame(allcases).to_csv(S/'main_curve_selection_LOCAL_ONLY.csv',index=False)
# Cross-track identity and query checks in addition to within-track/seed/model checks.
for c in cases:
 g=p[(p[keys]==pd.Series({k:c[k] for k in keys})).all(axis=1)&(p.seed==1)&(p.condition=='molecular_relation')]
 a=g[g.track=='Known-drug'].set_index('observation_id')[['query_time','y_true_log','support_end_time']].sort_index();b=g[g.track=='Joint drug–source'].set_index('observation_id')[a.columns].sort_index();pd.testing.assert_frame_equal(a,b)
cs=pd.read_csv(S/'cohort_summary.csv');cs=cs[cs.k==2].set_index(['condition','track'])
def val(cond,track):
 x=cs.loc[(cond,track)];return f'{x.sb_hrmse_mean:.4f} $\\pm$ {x.sb_hrmse_std:.4f}'
main=(D/'main.tex').read_text();start=main.index('Figure~\\ref{fig:curves} compares');end=main.index('\\subsection{Molecular conditioning',start)
main=main[:start]+r'''Figures~\ref{fig:curves} and \ref{fig:joint-curves} show the same six participant profiles under Known-drug and joint drug--source evaluation, respectively. Neural models use pooled multi-drug training in both settings; the target drug and source are excluded only in the latter. Model colors, case order, time axes, and concentration limits are matched between figures. The original four Known-drug cases are retained; Warfarin and Xanthohumol use the profile nearest the median Known-drug seed-1 MoDE-PK MSE among profiles with at least eight future times, or the maximal available count otherwise. The identical profiles are then shown under joint holdout without selecting by Joint error. These are post hoc descriptive cases, not a representative accuracy estimate. Original Joint failure examples and pathway diagnostics remain supplementary. The matched Joint examples show that a lower relative concentration error can coexist with substantial errors in trajectory level and shape; relative model ranking alone does not establish adequate forecasting for a held-out drug.
\begin{figure}[!htbp]\centering\includegraphics[width=.96\linewidth]{Known_PK_curves.pdf}
\caption{Known-drug concentration forecasts after pooled multi-drug training, $k=2$. Panels match the drugs, participants and axes of Figure~\ref{fig:joint-curves}. Each colored line is the geometric ensemble of ten seeds; PopPK + MAP is a drug-specific reference. Squares are observed history, open circles are future measurements, and dashed lines mark the final supplied observation. Gray marks show doses and infusion intervals. Lines connect saved query predictions, not dense solver trajectories; ensemble curves are distinct from mean seed-wise scores.}\label{fig:curves}
\end{figure}
\begin{figure}[!htbp]\centering\includegraphics[width=.96\linewidth]{Joint_PK_curves.pdf}
\caption{Joint drug--source holdout forecasts for the same six profiles as Figure~\ref{fig:curves}, with identical colors, markers, ten-seed aggregation, and axis limits. Each target drug and its source were excluded from training. Drug-specific PopPK + MAP is not applicable. Warfarin and Xanthohumol illustrate large concentration-level errors and imperfect temporal prediction; all-drug quantitative results are in Table~\ref{tab:joint}.}\label{fig:joint-curves}
\end{figure}

'''+main[end:]
# Append hybrid only within the relation-OFF input block of the ablation table.
lines=main.splitlines();ix=next(i for i,l in enumerate(lines) if l.startswith('Uni-Mol &'))
lines.insert(ix+1,'Uni-Mol + drug identity & '+val('molecular_identity','Known-drug')+' & '+val('molecular_identity','Joint drug–source')+r'\\')
main='\n'.join(lines)+'\n'
paragraph='The additional input-concatenation control combines the 512-dimensional frozen Uni-Mol vector with a 512-dimensional trainable drug-identity vector before the shared 32-dimensional context projection. Both neighborhood losses are OFF. An unseen drug retains its molecular input but receives the mean training-drug identity vector. This control preserves the original PK dynamics and observer and uses the same 120-epoch validation selection, but has more trainable parameters than either single-input control. It is distinct from the drug-specific parameter-offset control.\n\n'
main=main.replace('\\subsection{Molecular conditioning and history representations}', '\\subsection{Molecular conditioning and history representations}\n'+paragraph)
main=main.replace('\\subsection{Representative concentration--time predictions}', '\\FloatBarrier\n\\subsection{Representative concentration--time predictions}').replace('\\subsection{Molecular conditioning and history representations}', '\\FloatBarrier\n\\subsection{Molecular conditioning and history representations}')
(D/'main.tex').write_text(main)
supp=(D/'supplementary.tex').read_text();supp=re.sub(r'\\clearpage\s*\\section\{Joint drug--source concentration forecasts\}.*?(?=\\input\{completion_results\})','',supp,flags=re.S)
# Preserve previous two failure cases explicitly, not as the matched main cases.
shutil.copy2(B/'Joint_PK_curves.pdf',D/'Original_joint_failure_cases.pdf')
extra=r'''
\clearpage\section{Molecular and drug-identity input validation}
A fixed concatenation architecture was registered before its 100 fits (ten seeds for Known and nine joint folds). The 200 single-input control fits were reused only after raw prediction hashes, checkpoint hashes, fitting partitions, population initializations, scalers and learning settings were verified. Checkpoints were selected using validation SB-hRMSE at $k=2$; no test-driven architecture search was performed. The cohort was previously used in model development, so this remains exploratory internal evidence. The additional trainable capacity precludes attributing every difference solely to complementary information.
\begin{table}[!htbp]\centering\small\caption{Relation-OFF input controls. SB-hRMSE mean $\pm$ SD over ten seeds.}\label{tab:identity-validation}\begin{tabular}{lcc}\toprule Input & Known-drug & Joint drug--source\\\midrule
'''
for cond,label in [('molecular','Uni-Mol'),('drug_id','Drug identity'),('molecular_identity','Uni-Mol + drug identity')]:extra+=label+' & '+val(cond,'Known-drug')+' & '+val(cond,'Joint drug–source')+r'\\'+'\n'
extra+=r'\bottomrule\end{tabular}\end{table}'+'\n'
drug=pd.read_csv(S/'drug_summary.csv');drug=drug[drug.k==2]
for track in ['Known-drug','Joint drug–source']:
 extra+=r'\begin{table}[!htbp]\centering\footnotesize\caption{'+track+r' input-control drug-level participant-balanced log-RMSE, ten-seed mean $\pm$ SD. Both relation losses are OFF.}\begin{tabular}{lccc}\toprule Drug & Uni-Mol & Drug identity & Combined\\\midrule'+'\n'
 for drugid,g in drug[drug.track==track].groupby('drug_id'):
  g=g.set_index('condition');extra+=drugid.replace('_',' ').capitalize()+' & '+' & '.join(f'{g.loc[c,"rmse_mean"]:.4f} $\\pm$ {g.loc[c,"rmse_std"]:.4f}' for c in ['molecular','drug_id','molecular_identity'])+r'\\'+'\n'
 extra+=r'\bottomrule\end{tabular}\end{table}'+'\n'
extra+=r'''
The accompanying workbook reports drug-level MAE, bias, twofold accuracy, tail error, R-squared and linear-scale errors, together with paired differences and the same-checkpoint $k=0$ versus $k=2$ history effect. The profile-wise identity $\operatorname{MSE}=\overline e^{\,2}+\overline{(e-\overline e)^2}$ decomposes future residual error descriptively. Subtracting future-label bias is not a deployable correction and does not demonstrate a correct trajectory shape.
\begin{figure}[!htbp]\centering\includegraphics[width=\linewidth]{Original_joint_failure_cases.pdf}\caption{Previously fixed Joint failure examples, preserved independently of the new matched Known/Joint main figures. These are the original Joint-selected median-error cases used by the update-path diagnostics. Plot conventions are unchanged.}\label{fig:original-joint-failures}\end{figure}
'''
supp=supp.replace('\\input{completion_results}',extra+'\n\\input{completion_results}');(D/'supplementary.tex').write_text(supp)
c=(D/'completion_results.tex').read_text().replace('supplementary Joint curves','original supplementary Joint failure cases');(D/'completion_results.tex').write_text(c)
for name in ['cohort_summary.csv','drug_summary.csv','paired_drug_summary.csv','run_inventory.csv']:shutil.copy2(S/name,D/'tables'/('identity_'+name))
for script in ['run_molecular_identity_validation.py','summarize_molecular_identity_validation.py','export_molecular_identity_validation.py']:shutil.copy2(R/'scripts'/script,D/'revision_code'/script)
manifest=dict(parent=str(B.relative_to(R)),summary=str(S.relative_to(R)),new_fits=100,reused_fits=200,curve_identity_checks=6,curve_model_seed_checks=allchecks,main_figures='Known and Joint separately; same six profiles and axis limits',primary_model_replaced=False)
(D/'identity_validation_manifest.json').write_text(json.dumps(manifest,indent=2));print(D)
