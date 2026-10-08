"""Drug-first pooled benchmark, explicit ablation, and matched overlaid trajectories."""
from pathlib import Path
import sys,shutil,re,json,hashlib
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from mhap_pk.data.traceable import load_profiles
D=R/'manuscript/MoDE-PK_EJPS_drug_first_20261002';S=R/'results/summaries/drug_first_editorial_20261002';S.mkdir(exist_ok=True)
if not D.exists():shutil.copytree(R/'manuscript/MoDE-PK_EJPS_essential_20261002',D)
mods=['molecular_relation','poppk_vi_map','dcm','unipk','deepct'];names=['MoDE-PK (Ours)','PopPK + MAP','DCM','Uni-PK','DeepCt'];colors=['#b2182b','#2166ac','#8856a7','#1b7837','#d95f02']
label=lambda x: {'r_lipoic_acid':'R-lipoic acid'}.get(x,x.capitalize())
metrics=pd.read_csv(R/'results/summaries/ten_seeds_20261001/drug_metrics_summary.csv');known=metrics[(metrics.track=='Known-drug')&(metrics.k==2)&metrics.condition.isin(mods)].copy();known.to_csv(S/'main_known_drug_table.csv',index=False)
rows=[]
for drug,g in known.groupby('drug_id',sort=True):
 x=g.set_index('condition').loc[mods];best=x.rmse.min();vals=[('\\textbf{'+f'{v:.3f}'+'}') if v==best else f'{v:.3f}' for v in x.rmse];rows.append(label(drug)+f' & {int(x.participants.iloc[0])} & {int(x.queries.iloc[0])} & '+' & '.join(vals)+r'\\')
table=r'''\begin{table}[!htbp]\centering\footnotesize\setlength{\tabcolsep}{3pt}
\caption{Known-drug forecasting after pooled multi-drug training, with two observed concentration times. Neural methods share parameters across training drugs; PopPK + MAP is a drug-specific reference. Participant-balanced drug log-RMSE is averaged over ten seeds; bold marks the lowest mean within each drug. $N_p$ counts test participants and $N_q$ counts future measurements, excluding supplied history and counted once across seeds. The separate single-drug training benchmark is supplementary.}\label{tab:known}
\begin{adjustbox}{max width=\linewidth}\begin{tabular}{lrrrrrrr}\toprule
Drug & $N_p$ & $N_q$ & MoDE-PK (Ours) & PopPK + MAP & DCM & Uni-PK & DeepCt\\\midrule
'''+ '\n'.join(rows)+r'\bottomrule\end{tabular}\end{adjustbox}\end{table}'
# Same stored queries, means in log space, and fixed representative selection.
p=pd.read_parquet(R/'results/summaries/ten_seeds_20261001/canonical_predictions.parquet');p=p[(p.k==2)&(p.protocol=='first_k')&p.condition.isin(mods)]
keys=['source_id','drug_id','participant_id','profile_id'];qkeys=keys+['observation_id'];ps,meta,_=load_profiles();lookup={tuple(x[k] for k in keys):x for x in ps};selected=pd.read_csv(R/'results/summaries/training_scope_extension_20261001/single_curve_selection.csv');cases=[]
for drug in ['buprenorphine','empagliflozin','mavoglurant','remifentanil']:cases.append(dict(track='Known-drug',**selected[selected.drug_id==drug].iloc[0][keys].to_dict(),selection='Previously fixed profile; median single-drug seed-1 error among time-count eligible profiles; not reselected for pooled accuracy'))
for drug in ['warfarin','xanthohumol']:
 a=p[(p.track=='Joint drug–source')&(p.drug_id==drug)&(p.condition=='molecular_relation')&(p.seed==1)].copy();a['sq']=(a.y_pred_log-a.y_true_log)**2;g=a.groupby(keys).agg(mse=('sq','mean'),n=('query_time','nunique')).reset_index();e=g[g.n>=8] if (g.n>=8).any() else g[g.n==g.n.max()];e=e.assign(distance=abs(e.mse-e.mse.median())).sort_values(['distance']+keys);cases.append(dict(track='Joint drug–source',**e.iloc[0][keys].to_dict(),selection='Median seed-1 MoDE-PK MSE among >=8 future times or maximal time-count eligibility; fixed before plotting'))
fig,axs=plt.subplots(3,2,figsize=(11,10));check=[];case_scores=[]
for ax,c in zip(axs.flat,cases):
 g=p[(p.track==c['track'])&(p[keys]==pd.Series({k:c[k] for k in keys})).all(axis=1)];available=[m for m in mods if m in set(g.condition)];reference=None
 for (mode,seed),h in g.groupby(['condition','seed']):
  v=h.set_index(qkeys)[['query_time','y_true_log','support_end_time']].sort_index()
  if reference is None:reference=v
  else:pd.testing.assert_frame_equal(reference,v)
 for mode in available:assert set(g[g.condition==mode].seed)==set(range(1,11))
 profile=lookup[tuple(c[k] for k in keys)];support=profile['episodes'][2].support.groupby('time').target_log_dv.mean();truth=g[g.condition=='molecular_relation'].query('seed==1').sort_values(['query_time','observation_id'])
 for mode,name,col in zip(mods,names,colors):
  if mode not in available:continue
  e=g[g.condition==mode].groupby('observation_id').agg(query_time=('query_time','first'),y_true_log=('y_true_log','first'),y_pred_log=('y_pred_log','mean')).sort_values(['query_time']);ax.plot(e.query_time,np.exp(e.y_pred_log),color=col,lw=2 if mode=='molecular_relation' else 1.4,marker='.',ms=3,zorder=4 if mode=='molecular_relation' else 3);case_scores.append(dict(track=c['track'],drug_id=c['drug_id'],condition=mode,ensemble_log_rmse=float(np.sqrt(np.mean((e.y_pred_log-e.y_true_log)**2)))))
 ax.scatter(support.index,np.exp(support),marker='s',c='black',s=32,zorder=6);ax.scatter(truth.query_time,np.exp(truth.y_true_log),facecolors='white',edgecolors='black',s=30,zorder=7);ax.axvline(support.index.max(),color='black',ls='--',lw=.8)
 for time,amount,route,duration in profile['events']:
  ax.axvline(time,color='gray',alpha=.4,lw=.7)
  if duration>0:ax.axvspan(time,time+duration,color='gray',alpha=.12)
 ax.set(yscale='log',xlabel='Time (h)',ylabel='Concentration (µmol/L)',title=label(c['drug_id'])+' | '+('Known-drug, pooled' if c['track']=='Known-drug' else 'Joint holdout'));ax.grid(alpha=.15);check.append(dict(track=c['track'],drug_id=c['drug_id'],models=len(available),seeds=10,queries=len(truth),support_times=len(support),same_queries=True))
handles=[Line2D([0],[0],color=co,lw=2,label=n) for co,n in zip(colors,names)]+[Line2D([0],[0],color='black',marker='s',ls='',label='Observed history'),Line2D([0],[0],color='black',marker='o',mfc='white',ls='',label='Future measurements')]
fig.legend(handles=handles,loc='upper center',ncol=4,fontsize=9);fig.tight_layout(rect=(0,0,1,.92));fig.savefig(S/'Overlaid_PK_curves.pdf');fig.savefig(S/'Overlaid_PK_curves.png',dpi=160);plt.close(fig);shutil.copy2(S/'Overlaid_PK_curves.pdf',D/'Overlaid_PK_curves.pdf');pd.DataFrame(cases).to_csv(S/'case_selection_LOCAL_ONLY.csv',index=False);pd.DataFrame(check).to_csv(S/'curve_checks.csv',index=False);pd.DataFrame(case_scores).to_csv(S/'case_ensemble_metrics.csv',index=False)
# Build ablation groups from ten-seed matched published summaries.
a=pd.read_csv(R/'results/summaries/ten_seeds_20261001/ablation_summary.csv');blocks=[('Observation-update pathways; molecular input and both relations retained',[('component_base','Without observation updates'),('component_parameter','Parameter updates only'),('component_state','State updates only'),('molecular_relation','MoDE-PK (Ours)')]),('Input representations; both relation losses OFF',[('no_molecule','Without molecular or identity features'),('drug_id','Drug identity (UNK for held-out drugs)'),('ecfp','ECFP'),('molecular','Uni-Mol')]),('Neighborhood losses; Uni-Mol input retained',[('molecular','Neither loss'),('chemical_only','Chemical only'),('pk_only','Empirical PK only'),('molecular_relation','Both (MoDE-PK)')])];ar=[]
for heading,items in blocks:
 ar.append(r'\multicolumn{3}{l}{\textit{'+heading+r'}}\\')
 for condition,name in items:
  v=a[a.condition==condition].set_index('track');ar.append(name+' & '+' & '.join(f"{v.loc[t,'sb_hrmse_mean']:.4f} $\\pm$ {v.loc[t,'sb_hrmse_std']:.4f}" for t in ['Known-drug','Joint drug–source'])+r'\\')
 ar.append(r'\addlinespace')
ablation=r'''\begin{table}[!htbp]\centering\footnotesize
\caption{Ablation study under pooled training, $k=2$, SB-hRMSE mean $\pm$ SD over ten seeds. Each variant is retrained with matched partitions and selection procedures. Comparisons are within blocks: the input block disables both neighborhood losses, whereas the update block retains them. Lower is better. Drug-level and additional metrics are supplementary.}\label{tab:ablation}
\begin{adjustbox}{max width=\linewidth}\begin{tabular}{lcc}\toprule Configuration & Known-drug & Joint holdout\\\midrule
'''+ '\n'.join(ar)+r'\bottomrule\end{tabular}\end{adjustbox}\end{table}'
BASE=R/'manuscript/MoDE-PK_EJPS_essential_20261002'
main=(BASE/'main.tex').read_text();supp=(BASE/'supplementary.tex').read_text();moved=[]
# Move diagnostic figures but retain clear study findings in main.
for labelid in ['fig:drugmetrics','fig:embedding','fig:lowdata']:
 pattern=r'\\begin\{figure\}.*?\\end\{figure\}'
 for match in list(re.finditer(pattern,main,re.S)):
  if '\\label{'+labelid+'}' in match.group():moved.append(match.group());main=main[:match.start()]+main[match.end():];break
 main=main.replace('Figure~\\ref{'+labelid+'}',{'fig:drugmetrics':'The supplementary complementary-metric figure','fig:embedding':'The supplementary within-drug representation figure','fig:lowdata':'The supplementary matched-budget pooling figure'}[labelid])
# Preserve the single-drug table already extensively reported in supplement; replace main only.
main=re.sub(r'\\begin\{table\}.*?\\label\{tab:known\}.*?\\end\{table\}',table,main,count=1,flags=re.S) if False else main
for tag,replacement in [('tab:known',table),('tab:ablation',ablation)]:
 for match in re.finditer(r'\\begin\{table\}.*?\\end\{table\}',main,re.S):
  if '\\label{'+tag+'}' in match.group():main=main[:match.start()]+replacement+main[match.end():];break
start=main.index('\\subsection{Single-drug forecasting performance}');end=main.index('\\begin{table}',start)
main=main[:start]+r'''\subsection{Known-drug forecasting after pooled training}
Table~\ref{tab:known} evaluates the shared multi-drug neural models on unseen participants receiving known drugs. MoDE-PK attained the lowest mean participant-balanced log-RMSE for seven of eleven drugs. PopPK + MAP uses drug-specific training and provides a conventional reference. This table differs from the supplementary single-drug benchmark, in which every neural method is trained separately for each drug.

'''+main[end:]
old='The secondary SB-hRMSE was 0.5129';start=main.index(old);end=main.index('\n\n',start)
main=main[:start]+'Pooled MoDE-PK achieved SB-hRMSE 0.4609. Drug-wise results retain the large difference between fluorouracil (one future measurement) and remifentanil (455). The separate single-drug benchmark, with its own per-drug checkpoint selection, is reported in the Supplementary Material; its MoDE-PK SB-hRMSE is 0.5129. These training scopes must not be combined into a single ranking.'+main[end:]
start=main.index('\\subsection{Complementary drug-level prediction measures}');end=main.index('Table~\\ref{tab:known} assesses',start)
main=main[:start]+r'''\subsection{Complementary drug-level prediction measures}
Participant-balanced log-MAE, predictions within twofold error, signed bias, concentration-scale RMSE, and $R^2$ are provided separately for every drug and training scope in the Supplementary Material. Undefined single-measurement $R^2$ values are retained as undefined. These measures complement drug log-RMSE without treating different drugs or repeated measurements as interchangeable independent samples.

'''+main[end:]
main=main.replace('\\subsection{Effects of parameter and state updates}','\\subsection{Ablation study: observation updates, molecular inputs, and relation losses}')
start=main.index('\\subsection{Representative single-drug concentration--time predictions}');end=main.index('\\subsection{Molecular conditioning within individual drugs}',start)
main=main[:start]+r'''\subsection{Representative concentration--time predictions}
Figure~\ref{fig:curves} overlays the pooled models within each profile, including four Known-drug cases and the two drugs where MoDE-PK underperforms in joint holdout. The Known-drug profiles are unchanged from the preceding time-count and median-error selection, rather than reselected for pooled accuracy. For each Joint drug, we select the profile nearest the median seed-1 MoDE-PK MSE among profiles with at least eight future times, or among those with the largest time count if none qualifies. These are post hoc descriptive examples, not cases selected to beat a comparator.
\begin{figure}[!htbp]\centering\includegraphics[width=\linewidth]{Overlaid_PK_curves.pdf}
\caption{Matched-profile concentration forecasts at $k=2$. Each colored line is the geometric ensemble of ten seed predictions for one model. Top two rows: pooled Known-drug neural models and drug-specific PopPK + MAP. Bottom row: joint drug--source holdout for Warfarin and Xanthohumol; drug-specific PopPK is not applicable. Squares are supplied observations, open circles are future measurements, and dashed vertical lines mark the last supplied observation. Gray marks indicate dosing times and infusion intervals. Lines connect saved query predictions, not densely evaluated solver trajectories. Ensemble curves differ from the mean individual-seed errors in Tables~\ref{tab:known}--\ref{tab:joint}. The single-drug per-model ten-seed atlas remains supplementary.}\label{fig:curves}
\end{figure}

'''+main[end:]
# Condense representation explanation in main; retain full figure/caption in supplement.
start=main.index('\\subsection{Molecular conditioning within individual drugs}');end=main.index('\\subsection{Influence of observation count}',start)
main=main[:start]+r'''\subsection{Molecular conditioning and history representations}
The input ablation in Table~\ref{tab:ablation} provides the quantitative test of molecular conditioning. Drug-level paired effects and UMAP displays are supplementary: the displays compare the same profiles with and without molecular features while both neighborhood losses are OFF. They describe changes in observer representations, not prediction accuracy or validated chemical similarity. The UMAP drugs are selected by having at least ten test profiles, independently of curve-case selection. Separate projections cannot be interpreted as aligned model-comparison axes.

'''+main[end:]
# Retain pooling finding once, remove duplicated section.
start=main.index('\\subsection{Benefit and harm from adding other drugs\' training data}');end=main.index('\\subsection{Prior quality and source-weighting sensitivity}',start);main=main[:start]+main[end:]
# Include existing failure analysis evidence, no new tuning or biological causal claim.
anchor='\\subsection{Complementary drug-level prediction measures}'
failure=r'''The two adverse holdout cases were examined separately. In a post hoc profile-level decomposition, squared mean log residual accounted for approximately 99.8\% of Warfarin MSE and 96.6\% of Xanthohumol MSE, indicating predominantly persistent concentration-level bias over the observed future interval. Support-based scale/rate diagnostics reduced Warfarin error from 4.5457 to 3.4011; scale-only correction reduced Xanthohumol error from 3.1771 to 2.8347. Neither surpassed the strongest comparator, and additional correction worsened the other nine held-out drugs. These diagnostics were not adopted into MoDE-PK. They characterize an unresolved failure pattern without attributing it to a specific biological or assay mechanism.

'''
main=main.replace(anchor,failure+anchor)
# Add moved figures under clear supplementary scope and retain original single-drug captions.
section=r'\clearpage\section{Additional displays of accuracy, representations, and data sharing}'+'\n'
for f in moved:
 f=f.replace('as the primary comparison','as the single-drug Known benchmark and joint-holdout comparison').replace('N/A denotes an unavailable reduced budget; Full varies by drug.','A dash denotes an infeasible reduced budget; Full denotes the complete drug-specific training cohort. A cell marked Full repeats that full-cohort budget and is not a missing experiment.')
 section+=f+'\n\n'
section+=r'''\paragraph{How to read the representations.} The left pair of UMAP panels compares molecular features present versus absent, colored by the last observed concentration. The right pair repeats those coordinates colored by future prediction error. Within each drug the same profiles are shown; axes are independently fitted. These drugs meet a count-based display criterion and need not match trajectory-case drugs.
\paragraph{Pooling budgets.} Columns count target-drug training participants, not inference observations. Fluorouracil has four training participants, so its four-participant condition is the Full condition. Indomethacin has three training participants, making budgets four, eight, and sixteen infeasible. Missing reduced-budget entries do not denote failed predictions or zero effects.
\paragraph{Failure diagnostics.} Warfarin and Xanthohumol holdout errors were predominantly persistent profile-level log bias. The decomposition uses future labels only for explanation; those labels do not enter support-based scale/rate corrections. The corresponding fixed-checkpoint analysis covers ten seeds and all folds. Corrections reduced errors for these two drugs while worsening the remaining nine, so they were not adopted. Details and all-drug metrics accompany the scale-shape analysis package.
'''
if '\\section{Additional displays of accuracy, representations, and data sharing}' not in supp:supp=supp.replace('\\end{document}',section+'\n\\end{document}')
main=main.replace('Across ten seeds, single-drug MoDE-PK achieved the lowest participant-balanced log-RMSE for 8 of eleven drugs among five methods trained on the same drug-specific data.','Across ten seeds, pooled MoDE-PK achieved the lowest participant-balanced log-RMSE for seven of eleven known drugs among the evaluated neural models and the drug-specific population PK reference. A separate single-drug benchmark favored MoDE-PK for eight drugs.')
main=main.replace('\\begin{document}','\\setlength{\\emergencystretch}{2em}\n\\begin{document}')
(D/'main.tex').write_text(main);(D/'supplementary.tex').write_text(supp)
# Resolve N/A into infeasible versus full-budget duplicate in the relocated figure.
low=pd.read_csv(R/'results/summaries/sample_size_effects_20261002/paired_by_seed.csv');low=low[low.k==2].copy();low['budget']=np.where(low.target_n==low.full_target_n,'Full',low.target_n.astype(str));low['gain']=-low.delta_rmse;drugs=sorted(low.drug_id.unique());full=low.groupby('drug_id').full_target_n.first();lim=low.groupby(['condition','drug_id','budget']).gain.mean().abs().max();fig,axs=plt.subplots(1,2,figsize=(12,6))
for ax,mode,title in zip(axs,['pooled_molecular','pooled_molecular_relation'],['Data pooling; relations OFF','Data pooling + both relations']):
 w=low[low.condition==mode].groupby(['drug_id','budget']).gain.mean().unstack().reindex(index=drugs,columns=['2','4','8','16','Full']);v=w.to_numpy();im=ax.imshow(np.ma.masked_invalid(v),cmap='RdBu',vmin=-lim,vmax=lim,aspect='auto')
 for i,d in enumerate(drugs):
  for j,b in enumerate(w.columns):
   value=v[i,j];txt=f'{value:+.3f}' if np.isfinite(value) else ('Full' if b!='Full' and int(b)==full[d] else '—');ax.text(j,i,txt,ha='center',va='center',fontsize=8,color='white' if np.isfinite(value) and abs(value)>lim*.6 else 'black')
 ax.set_yticks(range(len(drugs)),[f'{label(d)} (n={int(full[d])})' for d in drugs]);ax.set_xticks(range(5),w.columns);ax.set_title(title);ax.set_xlabel('Target-drug training participants');fig.colorbar(im,ax=ax,shrink=.7,label='Single-drug minus pooled log-RMSE')
fig.tight_layout();fig.savefig(D/'Pooling_benefit.pdf');fig.savefig(S/'Pooling_benefit.pdf');plt.close(fig)
(S/'manifest.json').write_text(json.dumps(dict(new_training=False,seeds=list(range(1,11)),primary_change='Pooled Known table replaces single-drug main table; all original results retained separately',curve_selection='Known cases fixed; Joint failures selected with explicit median-error and time-count rule',code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
print(D)
