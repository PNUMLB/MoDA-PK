"""Build a consistent manuscript from verified hybrid-primary predictions only."""
from pathlib import Path
import json,re,shutil,hashlib
import pandas as pd,numpy as np
R=Path(__file__).resolve().parents[1];S=R/'results/summaries/hybrid_primary_20261002';OLD=R/'manuscript/MoDE-PK_EJPS_identity_validation_20261002';D=R/'manuscript/MoDE-PK_EJPS_hybrid_primary_20261002'
TITLE='MoDE-PK: Integrating Molecular Structure, Drug Identity, and Observation History for Pharmacokinetic Forecasting'
N={'ours':'MoDE-PK (Ours)','poppk_vi_map':'PopPK + MAP','dcm':'DCM','unipk':'Uni-PK','deepct':'DeepCt','without_updates':'Without observation updates','parameter_only':'Parameter updates only','state_only':'State updates only','without_molecule':'Without molecular input','without_identity':'Without identity input','chemical_only':'Chemical regularization','pk_only':'Empirical PK regularization','both_relations':'Both regularizers','ecfp':'ECFP + identity','chemberta_mlm':'ChemBERTa MLM + identity','chemberta_mtr':'ChemBERTa MTR + identity','memoryless':'Memoryless observer','budget240':'MoDE-PK, 240 epochs','ours_240':'MoDE-PK, 240 epochs','dcm_240':'DCM, 240 epochs','unipk_240':'Uni-PK, 240 epochs','deepct_240':'DeepCt, 240 epochs'}
K='Known-drug';J='Joint drug–source'
def esc(s):return str(s).replace('–','--').replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')
def drug(s):return 'R-lipoic acid' if s=='r_lipoic_acid' else s.capitalize()
def f(x,n=3):return 'NA' if pd.isna(x) else f'{x:.{n}f}'
def pm(row,key='log_rmse',n=4):return f(row[key+'_mean'],n)+r' $\pm$ '+f(row[key+'_std'],n)
def block(s,start,end):return s[s.index(start):s.index(end,s.index(start))]
def tab(caption,label,heads,rows,spec=None,small=True):
 return '\n'+r'\begin{table}[!htbp]\centering'+('\\footnotesize' if small else '\\small')+'\n'+r'\caption{'+caption+'}'+r'\label{'+label+'}\n'+r'\begin{adjustbox}{max width=\linewidth}\begin{tabular}{'+(spec or 'l'+'r'*(len(heads)-1))+'}\\toprule\n'+' & '.join(heads)+r'\\\midrule'+'\n'+'\n'.join(' & '.join(map(str,row))+r'\\' for row in rows)+'\n'+r'\bottomrule\end{tabular}\end{adjustbox}\end{table}'+'\n'
def longtab(caption,label,heads,rows):
 return '\n'+r'\begingroup\scriptsize\setlength{\tabcolsep}{3pt}\begin{longtable}{l'+'r'*(len(heads)-1)+'}\n'+r'\caption{'+caption+r'}\label{'+label+r'}\\\toprule'+'\n'+' & '.join(heads)+r'\\\midrule\endfirsthead'+'\n'+r'\multicolumn{'+str(len(heads))+r'}{l}{\tablename\ \thetable\ (continued)}\\\toprule'+'\n'+' & '.join(heads)+r'\\\midrule\endhead\bottomrule\endfoot'+'\n'+'\n'.join(' & '.join(map(str,row))+r'\\' for row in rows)+'\n'+r'\end{longtable}\endgroup'+'\n'
def fig(file,caption,label,width='.98'):
 return '\n'+r'\begin{figure}[!htbp]\centering\includegraphics[width='+width+r'\linewidth]{'+file+'}\n'+r'\caption{'+caption+r'}\label{'+label+r'}\end{figure}'+'\n'

def main():
 assert json.loads((S/'analysis_manifest.json').read_text())['status']=='completed'
 D.mkdir(exist_ok=True);(D/'tables').mkdir(exist_ok=True);(D/'revision_code').mkdir(exist_ok=True)
 for name in ['references.bib','elsarticle.cls','elsarticle-harv.bst']:
  if (OLD/name).exists():shutil.copy2(OLD/name,D/name)
 shutil.copy2(OLD/'tables/drug_data_inventory.tex',D/'tables/drug_data_inventory.tex')
 for p in S.glob('*.pdf'):shutil.copy2(p,D/p.name)
 for p in S.glob('*.csv'):
  if 'LOCAL_ONLY' not in p.name:shutil.copy2(p,D/'tables'/p.name)
 mainold=(OLD/'main.tex').read_text();suppold=(OLD/'supplementary.tex').read_text()
 dr=pd.read_csv(S/'primary_drug_summary.csv');dr=dr[(dr.protocol=='first_k')&(dr.k==2)];cs=pd.read_csv(S/'primary_cohort_summary.csv');cs=cs[(cs.protocol=='first_k')&(cs.k==2)]
 ac=pd.read_csv(S/'ablation_cohort_summary.csv');ac=ac[ac.k==2];ad=pd.read_csv(S/'ablation_drug_summary.csv');ad=ad[ad.k==2]
 def cv(model,track):return cs[(cs.condition==model)&(cs.track==track)].iloc[0]
 def av(model,track):return ac[(ac.condition==model)&(ac.track==track)].iloc[0]
 wins={}
 for track in [K,J]:
  q=dr[dr.track==track].pivot(index='drug_id',columns='condition',values='log_rmse_mean');wins[track]=int((q.idxmin(axis=1)=='ours').sum())
 low=pd.read_csv(S/'pooling_paired_summary.csv');two=low[(low.cap==2)&(low.k==2)];improved=int((two.delta_log_rmse_mean<0).sum())
 oldtitle=re.search(r'\\title\{([^}]+)\}',mainold).group(1)
 pre=mainold[:mainold.index('\\begin{abstract}')].replace(oldtitle,TITLE)
 abstract=('Small drug-specific cohorts motivate shared models that combine drug-level information with sparse individual observations. We propose MoDE-PK, integrating a frozen molecular representation and a trainable drug-identity representation with analytic dose-event compartment dynamics and a recurrent parameter--state observer. An unseen drug uses its molecular vector with the mean training-drug identity vector. Evaluation used 482 profiles from 342 source-scoped participants, 11 drugs, and nine source groups. Across ten seeds, MoDE-PK achieved the lowest participant-balanced log-RMSE for '+str(wins[K])+' of eleven known drugs in the primary comparison and '+str(wins[J])+' of eleven drugs under joint drug--source holdout. Source-balanced hierarchical log-RMSE was '+f(cv('ours',K).sb_hrmse_mean,4)+' and '+f(cv('ours',J).sb_hrmse_mean,4)+', respectively. In matched low-data experiments with two target training participants, adding other-drug records improved '+str(improved)+' of eleven drugs. Retrained component and input analyses assess the contributions of drug conditioning and observation updates. Strong identity-only transfer results and large errors for some withheld drugs limit attribution of generalization to molecular structure alone. These internal results support an integrated forecasting framework while revealing heterogeneous benefits of cross-drug learning.')
 main=pre+'\\begin{abstract}\n'+abstract+'\n\\end{abstract}\n'+block(mainold,'\\begin{keyword}','\\section{Introduction}').replace(oldtitle,TITLE)
 intro=block(mainold,'\\section{Introduction}','\\section{Materials and methods}')
 intro=intro[:intro.index('We introduce Molecularly')]+r'''We introduce Molecularly Conditioned Dose-Event Pharmacokinetics (MoDE-PK), integrating molecular structure and learned drug identity in a shared context that conditions both a compartment prior and a sequential parameter--state observer. The structural representation provides information for a previously unobserved drug; the identity representation allows training drugs to carry information not captured by the frozen structural features. Explicit dose events control propagation, while separate observation updates adjust effective clearance, central volume, and central drug amount. Multi-drug learning here shares forecasting components across compounds; it does not model drug--drug interactions.

We evaluate known-drug prediction in unseen participants, the benefit of adding other-drug records at fixed target-data budgets, and prediction for drugs withheld together with their source. Component retraining separates molecular and identity inputs, parameter and state updates, recurrent memory, and optional neighborhood regularization. Drug-level results and matched concentration trajectories complement cohort summaries, so gains for common or densely sampled drugs cannot conceal failure on individual compounds.

'''
 main+=intro+'\\section{Methods}\n'
 problem=block(mainold,'\\subsection{Multi-drug forecasting','\\subsection{Cohort assembly')
 problem=problem[:problem.index('\\begin{figure}')]
 problem=problem.replace('let $z_d$ be its molecular representation,','let $z_d$ be its molecular representation, $e_d$ its identity representation,').replace('z_d,\\mathcal D','z_d,e_d,\\mathcal D').replace('through $z_d$','through $[z_d;e_d]$').replace('target drug representation','target drug representations')
 main+=problem+fig('Figure1.pdf',r'MoDE-PK architecture. The standardized frozen Uni-Mol vector and a learned drug-identity vector are concatenated before a shared context projection. For a withheld drug, the identity branch uses the mean training-drug vector. The context conditions an effective PK prior and the GRU observer. Recorded doses drive compartment propagation; available concentrations drive clearance/volume and central-amount updates. Future concentrations are prediction targets only. Neighborhood regularization is inactive in the primary model.','fig:framework')
 main+=r'''\subsection{Joint molecular and identity conditioning}
The frozen Uni-Mol vector $z_d\in\mathbb R^{512}$ is standardized using training drugs only. Each fitting partition learns a table $E\in\mathbb R^{n_D\times512}$ over its training drugs. The identity input is
\begin{equation}
e_d=\begin{cases}E_{\iota(d)},&d\in\mathcal D_{\mathrm{train}},\\
 n_D^{-1}\sum_{j=1}^{n_D}E_j,&d\notin\mathcal D_{\mathrm{train}}.
\end{cases}
\end{equation}
The fallback is an average embedding, not a population PK fit for the withheld drug. Its own molecular input is retained. The shared prior is
\begin{align}
c_d&=\operatorname{SiLU}(W_c[z_d;e_d]+b_c),\\
v_d&=W_\mu c_d+b_\mu,\qquad \mu_d=\beta+v_d,\\
\theta&=(\log k_a,\log CL,\log V_c,\log Q,\log V_p).
\end{align}
The context $c_d$ has 32 dimensions. The trainable base $\beta$ is initialized from training-only population calibration; the residual head starts at zero. Exponentiation gives positive parameters. These are effective predictive quantities, because oral bioavailability and individual physiological parameters are not independently identified. Identity entries, the context projection, and the observer are optimized jointly; Uni-Mol remains frozen.

'''
 dynamics=block(mainold,'\\subsection{Compartment dynamics','\\subsection{Joint optimization')
 dynamics=dynamics.replace('molecular context','joint drug context').replace('molecular prior','drug-conditioned prior').replace('h_d','c_d')
 main+=dynamics+r'''\subsection{Training objective and model comparisons}
For source $s$, drug $d$, participant $p$, and future measurement $q$, let $w_{sdpq}=(|S||D_s||P_{sd}||Q_{sdp}|)^{-1}$. Repeated profiles are pooled within the participant. The primary objective is
\begin{equation}
\mathcal L=\sum_{sdpq}w_{sdpq}(\widehat y-y)^2+10^{-3}R_\delta+10^{-4}R_v.
\end{equation}
$R_\delta$ averages squared two-component parameter offsets across available observation times; $R_v$ averages the five components of $v_d^2$. Both neighborhood regularizers are OFF in the primary model. Their additions are separate retrained comparisons. AdamW uses learning rate 0.003, weight decay $10^{-4}$, and gradient clipping at 5. Training samples $k\in\{0,1,2,4\}$ per profile and epoch. Validation SB-hRMSE at $k=2$ selects a checkpoint at epochs 0, 10, \ldots, 120. Ten fitting seeds (1--10) are retained without selecting by test error.

The concatenation configuration was adopted after inspecting earlier internal input comparisons. Accordingly, this cohort is development and internal evaluation evidence, not untouched confirmation of model selection. The architecture and settings were then fixed for the analyses reported here; no ablation with a favorable test score replaces the declared primary model. The earlier molecular-only model used both neighborhood losses. Comparisons against it therefore change both inputs and regularization; the relation-OFF controls separate these factors.

Update ablations retrain the prior and observer with parameter updates only, state updates only, or neither. Input-removal ablations zero one 512-dimensional branch while retaining the 1024-to-32 projection. This controls projection shape; an unused identity table does not contribute effective capacity. Optional chemical and empirical PK regularizers, alternative molecular encoders, and a memoryless observer are assessed separately. A 240-epoch sensitivity analysis changes the training ceiling while preserving validation selection.

'''
 comparator=block(mainold,'\\subsection{Comparator implementations for multi-drug','\\subsection{Drug-specific benchmarking')
 comparator=comparator.replace('The single-drug Known-drug panel','The primary comparison panel')
 comparator=comparator[:comparator.index('The Supplementary Material retains')]+r'''Neural-ODE-PK~\citep{lu2021}, mixed-effects DCM~\citep{janssen2024}, and selective prior flattening~\citep{hughes2021} are relevant prior approaches, but no validated matched-cohort score is assigned without a completed implementation. Public native examples do not substitute for human-cohort comparisons.

'''
 main+=comparator+r'''\subsection{Single-drug and matched low-data comparisons}
The separate single-drug benchmark fits each neural model to one drug's training participants and selects checkpoints using that drug's validation participants. Molecular and identity inputs are constant within such a fit. This scope differs from the primary globally selected multi-drug fit.

Controlled transfer uses nested target training subsets of 2, 4, 8, or 16 participants when below the available total, plus the full training count. Target-only and pooled fitting use identical target subsets, target validation participants, and test queries. Other drugs contribute their original training participants; all linked records of omitted target participants are removed. Each pooled fit is selected using target-drug validation. Population initialization, feature scaling, and identity mapping use that resulting training partition. Both arms use the primary concatenation architecture, no neighborhood losses, and the same optimization settings. Ten seeds vary both subset sampling and initialization. Validation counts remain fixed, so two training participants do not imply only two labelled target participants available for development.

'''
 evaluation=block(mainold,'\\subsection{Participant-grouped','\\section{Results}')
 evaluation=evaluation.replace('relation targets,','identity mapping, optional relation targets,').replace('Bootstrap intervals~\\citep{efron1986} preserve participant clustering~\\citep{field2007}, and source-omission analyses retain the available grouping; both remain exploratory.','Exploratory source-cluster bootstrap intervals~\\citep{efron1986,field2007} resample source-level paired differences after seed averaging. They are conditional on these fitted models and observed sources; they do not account for prior model-selection inspection.')
 main+=evaluation+'\\section{Results}\n'+block(mainold,'\\subsection{Cohort assembly','\\subsection{Molecular conditioning')
 main+='\\subsection{Known-drug prediction after multi-drug training}\n'+f'MoDE-PK had the lowest mean participant-balanced log-RMSE for {wins[K]} of eleven drugs in the primary panel (Table~\\ref{{tab:known}}). Its SB-hRMSE was {pm(cv("ours",K),"sb_hrmse")}, compared with {pm(cv("poppk_vi_map",K),"sb_hrmse")} for the drug-specific population reference. Drug-level counts and errors are retained rather than replacing them with a pooled ranking.\n'
 def drugtable(track,label,caption):
  models=['ours','poppk_vi_map','dcm','unipk','deepct'] if track==K else ['ours','dcm','unipk','deepct'];rows=[]
  for d,g in dr[dr.track==track].groupby('drug_id'):
   g=g.set_index('condition');best=g.log_rmse_mean.min();vals=[]
   for m in models:
    v=f(g.loc[m,'log_rmse_mean']);vals.append(r'\textbf{'+v+'}' if np.isclose(g.loc[m,'log_rmse_mean'],best) else v)
   rows.append([drug(d),str(int(g.participants.iloc[0])),str(int(g.queries.iloc[0]))]+vals)
  return tab(caption,label,['Drug','$N_p$','$N_q$']+[N[m] for m in models],rows)
 main+=drugtable(K,'tab:known',r'Known-drug prediction after pooled multi-drug neural training, $k=2$. Values are ten-seed mean participant-balanced log-RMSE; bold marks the lowest mean per drug. PopPK + MAP is a drug-specific reference. $N_p$ is the test participant count and $N_q$ the future-measurement count, counted once across seeds. Seed SD and other drug-level measures are supplementary.')
 main+='\\subsection{Prediction for withheld drugs and sources}\n'+f'The joint evaluation withholds one source and its drugs from fitting and another source from selection. MoDE-PK achieved the lowest mean error for {wins[J]} of eleven drugs among the four applicable methods (Table~\\ref{{tab:joint}}). SB-hRMSE was {pm(cv("ours",J),"sb_hrmse")}, versus {pm(cv("unipk",J),"sb_hrmse")} for Uni-PK. The primary $k=2$ setting permits two concentrations from the target profile at inference; it is not a concentration-free forecast.\n'
 main+=drugtable(J,'tab:joint',r'Joint drug--source holdout, $k=2$, ten-seed mean participant-balanced log-RMSE. Each drug is predicted only by the fold excluding its source from fitting and validation. Bold marks the lowest mean. Drug-specific PopPK + MAP is inapplicable without drug-specific training records.')
 bias=pd.read_csv(S/'profile_bias_shape.csv');rows=[]
 for d in ['warfarin','xanthohumol']:
  g=bias[(bias.track==J)&(bias.drug_id==d)];frac=100*g.bias_squared.sum()/g.mse.sum();rows.append((d,frac))
 main+='Warfarin and Xanthohumol remain important failures. Squared profile-mean log residual accounts for '+f(rows[0][1],1)+r'\% and '+f(rows[1][1],1)+r'''\% of their summed profile MSE across seeds, respectively. This decomposition indicates persistent concentration-level mismatch, not proof that temporal shape is correct. Removing an error mean calculated from future labels is a diagnostic calculation, not an available forecasting correction. Drug and source novelty are confounded in this cohort, so these failures cannot be assigned to molecular properties alone.

'''
 main+='\\subsection{Component and input analyses}\n'+r'''Table~\ref{tab:ablation} uses the same concatenation backbone for retrained update and input-removal controls. All update controls retain a trainable prior; ``without updates'' does not mean without a prior network. Drug-level results are supplied separately to reveal heterogeneity behind the hierarchical summaries. Optional relation terms and encoder replacements are supplementary.
'''
 rows=[]
 abmodels=['ours','without_updates','parameter_only','state_only','without_molecule','without_identity','memoryless']
 for m in abmodels:
  vals=[]
  for t in [K,J]:
   x=av(m,t);value=pm(x,'sb_hrmse');best=min(av(a,t).sb_hrmse_mean for a in abmodels);vals.append(r'\textbf{'+value+'}' if np.isclose(x.sb_hrmse_mean,best) else value)
  rows.append([N[m]]+vals)
 main+=tab(r'Retrained MoDE-PK components, $k=2$. SB-hRMSE mean $\pm$ seed SD. Bold marks the lowest displayed mean in each evaluation. Both neighborhood terms are OFF. Removed inputs are zeroed in the same-width projection; removed update mechanisms are disabled during training and inference. Each variant uses ten seeds and the same partitions and checkpoint-selection rule.','tab:ablation',['Configuration',K,'Joint holdout'],rows)
 bestk=ac[ac.condition.isin(['ours','without_updates','parameter_only','state_only'])& (ac.track==K)].sort_values('sb_hrmse_mean').iloc[0];bestj=ac[ac.condition.isin(['ours','without_updates','parameter_only','state_only'])&(ac.track==J)].sort_values('sb_hrmse_mean').iloc[0]
 main+=f'Among the update configurations, {N[bestk.condition].lower()} had the lowest Known-drug error ({f(bestk.sb_hrmse_mean,4)}), and {N[bestj.condition].lower()} had the lowest joint error ({f(bestj.sb_hrmse_mean,4)}). These results constrain which update benefits can be claimed.\n\n'
 main+='Removing the molecular branch increased Known-drug error but reduced joint holdout error, whereas removing identity increased error in both settings. The molecular contribution therefore depends on the evaluation distribution; complementary input benefits cannot be inferred from the concatenation design alone.\n\n'
 main+=f'The optional combination of chemical and empirical PK regularization achieved SB-hRMSE {pm(av("both_relations",K),"sb_hrmse")} in Known-drug evaluation and {pm(av("both_relations",J),"sb_hrmse")} under joint holdout. These exploratory values are lower than the prespecified relation-OFF primary configuration in this internal evaluation. They motivate a separately validated regularization choice, rather than a claim that the primary recipe is optimal across all tested variants.\n\n'
 ids=pd.read_csv(R/'results/summaries/molecular_identity_20261002/cohort_summary.csv');idj=ids[(ids.condition=='drug_id')&(ids.track==J)&(ids.k==2)].iloc[0]
 main+=f'The separately trained compact identity-only reference had joint SB-hRMSE {pm(idj,"sb_hrmse")}, below the primary model. Its withheld-drug input uses the training-identity fallback. This reflects a transferable population representation and observed history, not recognition of an unseen identity. Thus, favorable transfer relative to the main comparator panel does not establish superiority to every internal input variant or an independent molecular benefit. The input-removal controls and compact single-input references have different projection widths and are reported separately.\n\n'
 main+='\\subsection{Benefit of other-drug training records}\n'+f'At a fixed budget of two target training participants, pooled fitting reduced drug log-RMSE for {improved} of eleven target drugs (Supplementary Table~\\ref{{tab:pooling}}). The comparison holds target training, validation, and evaluation records constant; benefits are not uniform. Increasing the target budget and training each model separately per drug provide additional reference points. Because target validation remains available, this experiment addresses reduced training data rather than fully unlabelled development.\n\n'
 rr0=pd.read_csv(S/'resampling_cohort_summary.csv');rr0=rr0[(rr0.condition=='ours')&(rr0.k==2)].set_index('evaluation')
 bb0=pd.read_csv(S/'budget_cohort_summary.csv');bb0=bb0[(bb0.condition=='ours_240')&(bb0.k==2)].set_index('track')
 main+='\\subsection{Partition and optimization sensitivity}\n'
 main+=f'With out-of-fold predictions concatenated before scoring, five-fold participant-grouped validation gave MoDE-PK SB-hRMSE {pm(rr0.loc["CV5"],"sb_hrmse")}. Alternative 70/10/20 and 80/10/10 partitions gave {pm(rr0.loc["ratio_712"],"sb_hrmse")} and {pm(rr0.loc["ratio_811"],"sb_hrmse")}. Under a 240-epoch ceiling, Known-drug and joint errors were {pm(bb0.loc[K],"sb_hrmse")} and {pm(bb0.loc[J],"sb_hrmse")}. The same extended ceiling was applied to the neural comparators. These internal checks assess partition and optimization sensitivity without replacing the original primary setting; drug-level results are supplementary.\n\n'
 main+=r'''\FloatBarrier
\subsection{Representative concentration--time predictions}
Figures~\ref{fig:known-curves} and \ref{fig:joint-curves} show identical six profiles under Known-drug and joint drug--source evaluation. The cases were fixed in the preceding model analysis and were not reselected to favor the concatenation model. Four original descriptive cases are retained; Warfarin and Xanthohumol were selected near the median previous-model Known seed-1 MSE among profiles with at least eight future times, or the maximum available count otherwise. These are post hoc explanatory examples, not a representative accuracy estimate. Matching cases, colors and axes reveals both amplitude and temporal errors under the two training conditions.
'''
 main+=fig('Known_PK_curves.pdf',r'Known-drug forecasts after pooled training, $k=2$. Each colored line is the geometric ensemble of seeds 1--10, evaluated on identical queries. Squares are observed history, open circles are future measurements, and the dashed line marks the last supplied observation. Gray marks indicate doses and infusion intervals. Lines connect saved query predictions and are not dense solver trajectories. Ensemble curves differ from mean seed-wise scores. The same six cases and axes appear in Figure~\ref{fig:joint-curves}.','fig:known-curves')
 main+=fig('Joint_PK_curves.pdf',r'Joint drug--source forecasts for the same six profiles and axes as Figure~\ref{fig:known-curves}. The target drug and source are absent from fitting and selection. PopPK + MAP is omitted because its drug-specific prior cannot be estimated in this condition. Curves use the same ten-seed aggregation and query-point connections.','fig:joint-curves')
 main+=r'''\FloatBarrier
\subsection{Observation history and forecasting behavior}
The initial-observation protocol evaluates $k=0,1,2,3,4$ on identical future queries (Figure~\ref{fig:history}). Additional observations also move the final observed time forward. The supplementary fixed-origin analysis fixes the latest observation at $t_4$ for nonempty histories to separate observation count from recency. Horizon-specific errors are reported within each drug with participant and query counts, rather than pooling different time ranges across drugs.
'''
 cfull=pd.read_csv(S/'primary_cohort_summary.csv')
 for track in [K,J]:
  q=cfull[(cfull.track==track)&(cfull.condition=='ours')&(cfull.protocol=='first_k')].set_index('k');main+=esc(track)+' SB-hRMSE changed from '+f(q.loc[0,'sb_hrmse_mean'],4)+' without observations to '+f(q.loc[2,'sb_hrmse_mean'],4)+' with two. '
 main+='The same-checkpoint history contrast is distinct from retraining without update modules.\n'+fig('Initial_history.pdf',r'Initial-observation sensitivity. All settings evaluate the same measurements after $t_4$; $k$ is the number of supplied distinct observation times. Points and bars show mean and SD across ten seeds. $k=3$ uses the same checkpoints and was not sampled during training.','fig:history')
 main+=r'''Within-drug UMAP displays compare the final observer states of MoDE-PK and its input-removal controls on the same profiles. Concentration and error are coloring variables, not projection inputs. Projection coordinates are independently fitted and do not share a distance scale. Original-space associations across ten seeds complement the single-seed displays; neither cluster separation nor projection trustworthiness measures predictive superiority (Supplementary Figures~\ref{fig:umap-observed}--\ref{fig:umap-error}).

\section{Discussion}
MoDE-PK combines shared cross-drug forecasting with a drug-specific identity representation and profile-specific observation processing. Molecular context and identity enter both the effective PK prior and recurrent observer, while an explicit dosing schedule drives compartment propagation. This organization connects information shared across drugs with information specific to a training drug and an observed trajectory.

The primary comparisons show favorable drug-balanced performance for several compounds, but pooling and observation updates do not help every drug. Identity-only transfer is a strong internal reference, and large holdout errors remain for Warfarin and Xanthohumol. Consequently, the results support the evaluated integrated forecasting formulation rather than universal superiority or molecular generalization independent of source effects. Optional regularization and update mechanisms must be judged by their retrained contrasts, not inferred to be beneficial because they are included in a model.

\subsection{Limitations and future work}
Eleven drugs provide limited structural coverage. Each drug is associated with one source group, so joint withholding changes molecular identity together with population, assay, dose, and sampling conditions. Provenance checks improve traceability without making all measurements equivalent; positive-only targets, uncertain historical doses, event order, and analyte definitions remain limitations. The cohort has been used repeatedly for development, including the choice of concatenated inputs. Ten seeds characterize fitting variability, not independent biological replication or external validation.

Comparator implementations preserve stated components within a common human forecasting interface but do not establish complete native-paper reproduction. Their support correction and molecular inputs differ from some original workflows. The 240-epoch sensitivity analysis assesses optimization budget; it cannot substitute for missing native data or a different inference mechanism. Sparse target validation also limits model selection. Prediction intervals, dose optimization, and clinical utility have not been validated.

Further work should expand independently sourced drug cohorts and test the frozen method on new studies, including the same drug across multiple studies. Richer measured covariates and verified dosing histories may help distinguish sources of concentration-scale and shape error. Such information should be added through explicit matched-input evaluations. The present effective parameter updates should not be interpreted as recovery of individual physiological clearance or volume.

\section{Conclusion}
MoDE-PK integrates molecular structure, learned drug identity, dose-event compartment propagation, and sequential observation updates. Drug-level comparisons and matched low-data experiments characterize where shared training is useful, while update and representation controls expose settings where simpler alternatives remain stronger. Joint drug--source errors and identity-only results define the current limits of the molecular-transfer claim. The framework provides a traceable basis for broader independent validation rather than a clinically validated forecasting tool.
\FloatBarrier
'''
 ending=mainold[mainold.index('\\section*{Data and code availability}'):]
 ending=ending.replace('A public repository address has not yet been assigned.','The publication package includes code, configuration and artifact manifests; row-level redistribution follows the listed source conditions.')
 main+=ending;main=main.replace(r'\end{table}',r'\end{table}'+chr(10)+r'\FloatBarrier');(D/'main.tex').write_text(main)
 # Supplement: data and mechanistic details first; results all regenerated below.
 supp=suppold[:suppold.index('\\section{Data provenance}')].replace(oldtitle,TITLE)+block(suppold,'\\section{Data provenance}','\\section{Model specification}')
 supp+=r'''\section{Model and optimization details}
The primary model concatenates 512 standardized frozen Uni-Mol dimensions and 512 learned identity dimensions before a 32-dimensional Linear--SiLU context. The trainable identity table contains training drugs only. A withheld identity is the arithmetic mean of training rows and retains its own molecular vector. The primary loss has no neighborhood regularization. Core dynamics, event handling, observation features and bounded updates follow the main equations.

'''
 supp+=block(suppold,'\\subsection{Frozen molecular features','\\subsection{Hierarchical training objective')
 supp=supp.replace('\\subsection{Construction of chemical and pharmacokinetic neighborhoods}','\\subsection{Optional neighborhood regularizers}\nThese terms are OFF in the primary model. The chemical-only, PK-only and combined conditions are independently retrained with coefficients 0.03 and 0.1, respectively.\n')
 supp+=r'''\subsection{Training, input controls and observation processing}
AdamW uses learning rate 0.003, weight decay $10^{-4}$, clipping norm 5 and double-precision PK propagation. Every ten epochs, including initialization, validation SB-hRMSE at $k=2$ selects the checkpoint. The main ceiling is 120 epochs; the budget sensitivity uses 240. Molecular and parameter residual heads start at zero; state gain starts at 0.5. Frozen molecular encoders do not receive PK gradients.

Branch-removal ablations keep a 1024-wide projection and zero only the designated input. They differ from compact single-input controls with a 512-wide projection. The identity mean for unseen drugs uses the resulting training partition only. ECFP substitutes 512 radius-2 fingerprint bits. ChemBERTa MLM and MTR supply 384-dimensional cached features padded with 128 zeros to match the molecular interface. All feature scaling is train-fold-specific. The memoryless control replaces the GRU with a last-observation MLP and is retrained; it retains dose features, the prior and both update mechanisms.

'''
 obs=block(suppold,'\\subsection{Observer inputs','\\section{Comparator implementations}').replace('molecular context','joint drug context').replace('molecular prior','drug-conditioned prior').replace('from $z_d$','from $[z_d;e_d]$').replace('h_d','c_d')
 supp+=obs+block(suppold,'\\section{Comparator implementations}','\\section{Evaluation metrics')
 supp=supp.replace(', distinct from the separate nlmixr2 estimator in Extended Analyses','')
 supp+=r'''\section{Drug-level performance and complementary metrics}
All scores are recalculated per seed from canonical query predictions. Within a drug, each participant receives equal total weight. Log-RMSE, log-MAE, concentration-scale RMSE, signed log error, within-twofold percentage and weighted $R^2$ use the main definitions. Single-query or constant-target $R^2$ is undefined (NA); negative values remain visible. Population and neural methods use identical query IDs and true concentrations. Cross-drug concentration-scale pooling is not used to rank drug-level performance.

'''
 for track in [K,J]:
  rows=[]
  for (d,m),g in dr[dr.track==track].groupby(['drug_id','condition']):
   a=g.iloc[0];rows.append([drug(d)+' / '+N[m],pm(a,'log_rmse',3),f(a.log_mae_mean),f(a.within_twofold_mean,1),f(a.signed_log_bias_mean),f(a.r2_log_mean)])
  supp+=longtab(esc(track)+r': drug-level ten-seed mean $\pm$ SD for log-RMSE and mean complementary scores. Twofold is a percentage, not prediction-interval coverage.','tab:detail-'+('known' if track==K else 'joint'),['Drug / model','Log-RMSE','MAE','Twofold','Bias','$R^2_{\log}$'],rows)
 for track in [K,J]:
  rows=[]
  for (d,m),g in dr[dr.track==track].groupby(['drug_id','condition']):
   a=g.iloc[0];rows.append([drug(d)+' / '+N[m],f(a.linear_rmse_mean),f(a.linear_mae_mean),f(a.r2_linear_mean)])
  supp+=longtab(esc(track)+r': participant-balanced concentration-scale RMSE and MAE ($\mu$mol/L), and concentration-scale weighted $R^2$.','tab:linear-'+('known' if track==K else 'joint'),['Drug / model','RMSE','MAE','$R^2$'],rows)
 rows=[[esc(t)+' / '+N[m],pm(g.iloc[0],'sb_hrmse'),pm(g.iloc[0],'log_rmse'),f(g.iloc[0].within_twofold_pct_mean,1)] for (t,m),g in cs.groupby(['track','condition'])]
 supp+=tab(r'Secondary cohort summaries, $k=2$. Pooled log-RMSE and pooled twofold percentage weight measurements equally and are distinct from drug-balanced quantities.','tab:cohort',['Evaluation / model','SB-hRMSE','Pooled log-RMSE','Twofold'],rows)
 boot=pd.read_csv(S/'primary_source_bootstrap.csv');rows=[[esc(a.track)+' / '+N[a.reference],f(a.mean_delta,4),f(a.ci025,4),f(a.ci975,4),int(a.source_count)] for a in boot.itertuples()]
 supp+=tab(r'Exploratory source-bootstrap differences: MoDE-PK minus reference. Seed-mean paired source scores are resampled; intervals are conditional on fitted models and the reused sources, not independent post-selection confirmation.','tab:bootstrap',['Track / reference','Difference','2.5\%','97.5\%','Sources'],rows)
 supp+=r'''\FloatBarrier\section{Component and representation analyses}
These are end-to-end retrained conditions of the primary concatenation model. Existing single-input controls are reported separately. The entire drug-level matrix and all history counts are also in the accompanying workbook.
'''
 for track in [K,J]:
  models=['ours','without_updates','parameter_only','state_only','without_molecule','without_identity'];rows=[]
  for d,g in ad[(ad.track==track)&ad.condition.isin(models)].groupby('drug_id'):
   g=g.set_index('condition');rows.append([drug(d)]+[pm(g.loc[m],n=3) for m in models])
  supp+=tab(esc(track)+r': retrained update and branch-removal effects, drug log-RMSE. Abbreviations: full, no updates, parameter only, state only, no molecular input, no identity input.','tab:component-'+('known' if track==K else 'joint'),['Drug','Full','No upd.','Param.','State','No mol.','No ID'],rows)
 rows=[[N[m],pm(av(m,K),'sb_hrmse'),pm(av(m,J),'sb_hrmse')] for m in ['ours','chemical_only','pk_only','both_relations','ecfp','chemberta_mlm','chemberta_mtr','memoryless']]
 supp+=tab(r'Regularization, molecular representation and memory controls on the concatenation backbone. SB-hRMSE mean $\pm$ SD over ten seeds.','tab:representation',['Configuration',K,'Joint holdout'],rows)
 for track in [K,J]:
  models=['ours','chemical_only','pk_only','both_relations','ecfp','chemberta_mlm','chemberta_mtr','memoryless'];rows=[]
  for d,g in ad[(ad.track==track)&ad.condition.isin(models)].groupby('drug_id'):
   g=g.set_index('condition');rows.append([drug(d)]+[f(g.loc[m,'log_rmse_mean']) for m in models])
  supp+=tab(esc(track)+r': drug-level mean log-RMSE for optional losses and representation controls. MLM/MTR denote the two ChemBERTa caches.','tab:extra-'+('known' if track==K else 'joint'),['Drug','Full','Chem.','PK','Both','ECFP','MLM','MTR','MLP'],rows)
 ids=ids[ids.k==2];rows=[]
 for m,name in [('molecular','Uni-Mol only'),('drug_id','Drug identity only'),('molecular_identity','Uni-Mol + identity')]:
  rows.append([name]+[pm(ids[(ids.condition==m)&(ids.track==t)].iloc[0],'sb_hrmse') for t in [K,J]])
 supp+=tab(r'Compact input references with relation losses OFF. These retained ten-seed fits are separately trained architectures; single-input projections have 512 inputs and concatenation has 1024. The concatenation row is the primary model.','tab:compact',['Input',K,'Joint holdout'],rows)
 rev=pd.read_csv(S/'primary_revision_by_seed.csv');rows=[]
 for (t,c),g in rev.groupby(['track','configuration']):rows.append([esc(t)+' / '+c,f(g.sb_hrmse.mean(),4)+r' $\pm$ '+f(g.sb_hrmse.std(),4)])
 supp+=tab(r'Primary-model revision on identical queries and seeds, $k=2$. The earlier molecular-only configuration includes both relation losses, whereas the current concatenation model excludes them. This is a multi-factor configuration comparison, distinct from the matched branch-removal ablation.','tab:revision',['Evaluation / configuration','SB-hRMSE'],rows)
 supp+=r'''\FloatBarrier\section{Single-drug and low-data transfer analyses}
All target-only MoDE-PK fits use the concatenation architecture. Unchanged comparator predictions retain their original fit identities. Each same-query comparison is verified before aggregation. Target-specific validation selects both arms of each pooling contrast; the main pooled model uses a different, global selection objective.
'''
 single=pd.read_csv(S/'single_drug_summary.csv');single=single[single.k==2];rows=[]
 for d,g in single.groupby('drug_id'):
  g=g.set_index('condition');rows.append([drug(d)]+[pm(g.loc[m],n=3) for m in ['ours','poppk_vi_map','dcm','unipk','deepct']])
 supp+=tab(r'Separate drug-specific training, $k=2$, participant-balanced log-RMSE mean $\pm$ SD. Every neural method is fitted separately to each drug.','tab:single',['Drug','MoDE-PK','PopPK + MAP','DCM','Uni-PK','DeepCt'],rows)
 counts=pd.read_csv(R/'results/summaries/single_drug_20261001/partition_counts.csv');cols=[c for c in counts if 'participant' in c];rows=[[drug(a.drug_id)]+[int(getattr(a,c)) for c in cols] for a in counts.itertuples()]
 supp+=tab('Participant counts in the original target-specific roles. Low-data budgets reduce training only; validation and test roles remain fixed.','tab:target-counts',['Drug']+[esc(c.replace('_participants','')) for c in cols],rows)
 lowd=pd.read_csv(S/'pooling_drug_summary.csv');lowd=lowd[lowd.k==2];rows=[]
 for (d,cap),g in lowd.groupby(['drug_id','cap']):
  g=g.set_index('scope');diff=low[(low.drug_id==d)&(low.cap==cap)&(low.k==2)].iloc[0];rows.append([drug(d),int(cap),pm(g.loc['target'],n=3),pm(g.loc['pooled'],n=3),pm(diff,'delta_log_rmse',3)])
 supp+=longtab(r'Matched target-data budgets. Drug log-RMSE mean $\pm$ SD, $k=2$. Difference is pooled minus target-only; a negative value favors other-drug training. The largest listed budget per drug is its full training count.','tab:pooling',['Drug','Train $N$','Target only','With donors','Difference'],rows)
 if (S/'Pooling_by_drug.pdf').exists():supp+=fig('Pooling_by_drug.pdf','Matched target-only and pooled learning across target training budgets. Identical target validation and test records are used. Error bars show ten-seed SD.','fig:pooling')
 supp+=r'''\FloatBarrier\section{Partition and training-budget sensitivity}
The five-fold analysis evaluates each eligible query once out of fold per seed. Fold predictions are concatenated before computing participant-balanced metrics; fold RMSEs are not averaged. Additional participant-grouped 70/10/20 and 80/10/10 partitions assess sensitivity to the original split. The 240-epoch experiment retains the original validation selection rule and initializations. These internal partitions do not create independent external studies.
'''
 rr=pd.read_csv(S/'resampling_cohort_summary.csv');rr=rr[rr.k==2];rows=[[esc(e)+' / '+N[m],pm(g.iloc[0],'sb_hrmse'),pm(g.iloc[0],'log_rmse')] for (e,m),g in rr.groupby(['evaluation','condition'])];supp+=tab(r'Partition sensitivity, ten-seed mean $\pm$ SD, $k=2$. CV5 uses concatenated out-of-fold predictions.','tab:resampling',['Partition / model','SB-hRMSE','Pooled log-RMSE'],rows)
 rd=pd.read_csv(S/'resampling_drug_summary.csv');rd=rd[rd.k==2]
 for e,g0 in rd.groupby('evaluation'):
  rows=[]
  for d,g in g0.groupby('drug_id'):
   g=g.set_index('condition');rows.append([drug(d)]+[pm(g.loc[m],n=3) for m in ['ours','poppk_vi_map','dcm','unipk','deepct']])
  supp+=tab(esc(e)+r': drug-level log-RMSE mean $\pm$ SD.','tab:cv-'+str(e),['Drug','MoDE-PK','PopPK + MAP','DCM','Uni-PK','DeepCt'],rows)
 b=pd.read_csv(S/'budget_cohort_summary.csv');b=b[b.k==2];rows=[[esc(t)+' / '+N[m],pm(g.iloc[0],'sb_hrmse'),pm(g.iloc[0],'log_rmse')] for (t,m),g in b.groupby(['track','condition'])];supp+=tab(r'Extended optimization ceiling of 240 epochs, $k=2$. Checkpoint selection remains validation-only.','tab:budget',['Evaluation / model','SB-hRMSE','Pooled log-RMSE'],rows)
 supp+=r'''\FloatBarrier\section{Observation timing, horizons and update diagnostics}
The initial protocol supplies the first $k$ distinct observed times and fixes queries after $t_4$. The fixed-origin protocol supplies $\{t_4\}$, $\{t_2,t_4\}$ or $\{t_1,t_2,t_3,t_4\}$ for $k=1,2,4$, respectively. Its empty-history prediction provides an additional reference. These analyses use the same selected checkpoints without retraining.
'''+fig('Fixed_origin.pdf','Fixed-origin observation sensitivity. Nonempty histories all end at the fourth observed time; identical future queries are evaluated. Mean and SD are over ten seeds.','fig:fixed')
 rows=[]
 for (t,m,protocol),g in cfull.groupby(['track','condition','protocol']):
  g=g.set_index('k');rows.append([esc(t)+' / '+N[m]+' / '+('initial' if protocol=='first_k' else 'fixed')]+[f(g.loc[k,'sb_hrmse_mean'],4) if k in g.index else '--' for k in range(5)])
 supp+=longtab('Observation-history sensitivity, mean SB-hRMSE. An absent fixed-origin condition is marked -- rather than assigned a value.','tab:history',['Evaluation / model / protocol']+[f'$k={k}$' for k in range(5)],rows)
 hh=pd.read_csv(S/'horizon_by_seed.csv');hh=hh[(hh.condition=='ours')&(hh.seed==1)];rows=[[esc(a.track)+' / '+drug(a.drug_id),esc(a.horizon_bin),int(a.participants),int(a.queries)] for a in hh.itertuples()]
 supp+=longtab('Forecast-horizon stratum counts. Horizons are hours after the final supplied observation at initial $k=2$; counts are identical across comparison models and seeds. The separate drug-horizon atlas reports errors with these denominators.','tab:horizon-counts',['Evaluation / drug','Horizon (h)','$N_p$','$N_q$'],rows)
 supp+=fig('History_effect_by_drug.pdf','Same-checkpoint effect of two observations relative to no observations on identical queries. Participant MSE is averaged within each drug. Negative differences favor history; bars show the mean and SD across ten seeds. This is distinct from retraining without observation updates.','fig:history-effect')
 he=pd.read_csv(S/'history_effect_drug_by_seed.csv');rows=[]
 for (t,d),g in he.groupby(['track','drug_id']):rows.append([esc(t)+' / '+drug(d),f(g.participant_mean_delta_mse.mean(),4),f(100*g.worsened_participant_fraction.mean(),1),f(100*g.worsened_profile_fraction.mean(),1)])
 supp+=tab('Observation effects by drug, mean over seeds. Positive MSE changes indicate worse future prediction. Worsening thresholds use 1e-10; participant and raw profile fractions are distinct.','tab:history-effect',['Evaluation / drug','MSE difference','Worse people (\%)','Worse profiles (\%)'],rows)
 st=pd.read_csv(S/'stage_drug_metrics.csv');rows=[]
 for (t,d),g in st.groupby(['track','drug_id']):
  x=g.groupby('mode').log_rmse.mean();rows.append([esc(t)+' / '+drug(d)]+[f(x[m]) for m in ['prior','parameter_path','state_path','full']])
 supp+=tab('Same-checkpoint inference interventions, mean drug log-RMSE. These are not retrained ablations: the selected full-model prior and weights are fixed while update paths are disabled.','tab:paths',['Evaluation / drug','Prior','Parameter','State','Full'],rows)
 supp+=r'''The identity $\mathrm{MSE}_i=\bar e_i^2+\overline{(e_i-\bar e_i)^2}$ separates profile-constant log bias from centered residual variation. It uses future labels only for analysis. Neither a low centered error nor a large bias fraction proves correct physiological dynamics. Update traces and negative history effects are supplied as diagnostics rather than calibrated uncertainty.
\FloatBarrier\section{Representation analysis}
Known-drug representations are extracted from the final 32-dimensional GRU state after two observations for the primary model and matched input-removal controls. Features are standardized using training profiles only. Each drug with at least ten test profiles is displayed: mavoglurant, remifentanil and xanthohumol. These counts include repeated profiles and are not independent participant counts. UMAP uses model seed 1, projection seed 17, five neighbors, minimum distance 0.1, random initialization and 500 epochs. Neither observed concentration nor future error is supplied to UMAP. The same coordinates are colored two ways. Axes are independent across panels. Trustworthiness reports neighborhood preservation, not predictive accuracy or globally calibrated distances.
'''+fig('UMAP_observed.pdf','Within-drug UMAP of final observer states, colored by the last observed natural-log concentration. Rows share drug and profiles, columns differ in input configuration. This color reflects information available to the observer.','fig:umap-observed')+fig('UMAP_future_error.pdf','The same coordinates as Figure~\\ref{fig:umap-observed}, colored by future profile log-RMSE. Future labels are used only for this post hoc coloring. Cluster separation is not a model-performance criterion.','fig:umap-error')
 supp+=r'''\subsection{Matched-profile comparator representations}
The same 101 Known-drug profiles are compared for all five primary methods at seed 1 and $k=2$. MoDE-PK uses its concatenation-model observer state. DCM and DeepCt combine their effective log parameters with the support-only concentration offset; Uni-PK combines its 64-dimensional dynamics context at zero concentration with that offset; PopPK uses the five effective parameters after support-only MAP. These are heterogeneous predictive summaries, not homologous encoder layers. Feature scaling uses training profiles. All projections are independently fitted, with 15 neighbors, minimum distance 0.1, seed 17 and 500 epochs. Baseline checkpoint hashes and the existing prediction-replay evidence are verified against the retained comparator fits. Future labels do not enter the feature or projection construction.
'''+fig('Cross_model_UMAP.pdf','Model-specific predictive features for the same 101 Known-drug profiles. Colors denote drugs. Axes and cluster distances are not shared across models; this descriptive plot does not rank forecast quality.','fig:cross-model')
 rep=pd.read_csv(S/'representation_original_space.csv');rows=[]
 for (d,m),g in rep.groupby(['drug_id','model']):
  rows.append([drug(d)+' / '+({'full':'MoDE-PK','without_molecule':'No molecule','without_identity':'No identity'}[m]),f(g.distance_association_last_concentration.mean()),f(g.distance_association_future_error.mean())])
 supp+=tab('Mean Spearman association over ten seeds between original standardized state distances and pairwise differences in last concentration or future error. Pairs are dependent; these are descriptive associations, not significance tests.','tab:representation-distance',['Drug / input','Observed concentration','Future error'],rows)
 supp+=r'''\FloatBarrier\section{Reproducibility and analysis lineage}
The primary concatenation analysis reuses 100 completed fits after verifying their checkpoint and prediction hashes. The updated component, representation, memory and budget panel contains 1,300 new fits; partition sensitivity adds 70, matched low-data conditions 750, and single-drug learning 110. All 2,230 new fits are tracked separately from the reused primary fits. The unchanged baseline predictions retain their original run identities. The accompanying manifests record code snapshots, fitting partitions, scalers, population initialization, configurations, checkpoint selection, prediction checksums and exact checkpoint replay checks. Query identity, true concentration, participant grouping and temporal eligibility are checked before paired aggregation.

Every current MoDE-PK table, trajectory and representation uses the declared concatenation configuration or an explicitly named retrained variant. Earlier molecular-only, fixed-prior and exploratory simulation analyses remain archived separately and are not relabeled as evidence for this model. Fitting times are wall-clock execution diagnostics under concurrent workloads, not isolated inference-latency benchmarks. The local reproducibility package includes the current table and figure generators and derived predictions subject to source redistribution conditions.

'''
 inv=pd.read_csv(S/'new_run_inventory.csv');rows=[]
 for (kind,variant),g in inv.groupby(['kind','variant']):rows.append([esc(kind)+' / '+N.get(variant,esc(variant)),len(g),f(g.epoch.median(),0),int((g.epoch==0).sum()),int((g.epoch==(240 if variant=='budget240' else 120)).sum())])
 supp+=tab('New fitting-job accounting and validation selection. The ceiling is 120 epochs except the 240-epoch budget condition. Initialization may be selected when later candidates worsen validation.','tab:fit-audit',['Family / condition','Fits','Median epoch','Epoch 0','At ceiling'],rows)
 supp+='\\bibliographystyle{elsarticle-harv}\n\\bibliography{references}\n\\end{document}\n';(D/'supplementary.tex').write_text(supp)
 # Cross-document supplementary refs use xr; include only after main/supp aux exist.
 main=(D/'main.tex').read_text().replace('\\begin{document}','\\usepackage{xr}\n\\externaldocument{supplementary_labels}\n\\begin{document}',1);(D/'main.tex').write_text(main)
 for p in R.joinpath('scripts').glob('*hybrid*.py'):shutil.copy2(p,D/'revision_code'/p.name)
 (D/'analysis_lineage.json').write_text(json.dumps({'primary':'molecular+identity concatenation; both updates; relations OFF','summary':str(S.relative_to(R)),'new_fits':2230,'reused_primary_fits':100,'source':str(OLD.relative_to(R)),'primary_prediction_sha256':hashlib.sha256((S/'primary_predictions.parquet').read_bytes()).hexdigest(),'previous_analyses_not_relabelled':True},indent=2));print(D)
if __name__=='__main__':main()
