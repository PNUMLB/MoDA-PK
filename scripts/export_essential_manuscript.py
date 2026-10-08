"""Add prediction-derived Essential results to an immutable manuscript revision."""
from pathlib import Path
import shutil,pandas as pd
R=Path(__file__).resolve().parents[1];S=R/'results/summaries/essential_20261002';D=R/'manuscript/MoDE-PK_EJPS_essential_20261002'
if not D.exists():shutil.copytree(R/'manuscript/MoDE-PK_EJPS_release_20261002',D)
s=pd.read_csv(S/'E01_2x2_summary.csv').set_index(['rule','assessment'])
rows=[]
for rule,label in [('V0','All-source selection'),('V1','Selection excluding S5'),('PopPK fixed','PopPK (fixed reference)')]:
 for scope,sl in [('all','All'),('without_S5','Excluding S5')]:
  v=s.loc[(rule,scope)];rows.append(f"{label} & {sl} & {v.sb_hrmse_mean:.4f} $\\pm$ {v.sb_hrmse_std:.4f} & {v.log_rmse_mean:.4f} $\\pm$ {v.log_rmse_std:.4f}\\\\")
section=r'''
\section{Validation-source selection sensitivity and implementation checks}\label{sec:essential}
This exploratory analysis concerns the pooled MoDE-PK reference configuration and its original training seeds 1--5, not the ten-seed single-drug benchmark. The S5 fluorouracil validation source has one future measurement and receives one ninth of the source-balanced validation weight. We compared V0 (all validation sources) with V1 (validation sources excluding S5), without changing training, participant partitions, inputs, or the assessment measurements. Each rule selected the lowest validation SB-hRMSE from epochs 0, 10, $\ldots$, 120; exact ties retained the earlier epoch. Because historical candidate checkpoints were unavailable, the five training trajectories were replayed using their original configurations. The V0 selected parameter tensors and assessment predictions reproduced the original files exactly; validation differences were below $3\times10^{-16}$.

V0 selected epochs 120, 80, 50, 110, and 100 for seeds 1--5, respectively. V1 selected 120, 110, 120, 110, and 100. Selection therefore changed in two of five runs. Table~\ref{tab:essential-selection} separates the selection effect from assessment reweighting. PopPK retains its original drug-specific selections and is only a fixed reference; no global S5-exclusion rule is imposed on it. V1 is not adopted as a new primary selection rule, and this sensitivity does not establish comparator-wide rank stability or external validation.

\begin{table}[!htbp]\centering\footnotesize
\caption{Pooled Known-drug selection sensitivity at $k=2$, mean $\pm$ SD across seeds 1--5. Assessment excluding S5 re-aggregates the same stored predictions. The single-drug main table and ten-seed primary results are unchanged.}\label{tab:essential-selection}
\begin{adjustbox}{max width=\linewidth}\begin{tabular}{llcc}\toprule
Selection & Assessment sources & SB-hRMSE & Pooled log-RMSE\\\midrule
'''+ '\n'.join(rows)+r'''
\bottomrule\end{tabular}\end{adjustbox}\end{table}

\paragraph{Source-record verification.}
The pinned PKdata distributor records contain 48 fluorouracil concentration rows; 46 occur in the current canonical source table. The two unmatched rows are zero concentrations, leaving the same 38 positive records. Concentration conversion from $\mu$g/mL to $\mu$mol/L, minutes to hours, and weight-linked mg/kg to $\mu$mol was verified using molecular weight 130.08. The eligible Known-drug partition contains four training profiles, one validation profile, and one assessment profile, each with one future query under the registered protocol. The distributor describes plasma and a 0.5-minute IV administration; the existing bolus approximation is retained. This file-level verification does not establish the original assay or the full clinical-record chain, nor does it reclassify zero records as measured BLQ values.

\paragraph{Prediction and checkpoint integrity.}
Checks covered 850 existing run directories representing 950 fits across pooled Known-drug, joint drug--source holdout, and single-drug comparisons. The 154 drug--model cells in the primary tables were recovered from predictions with maximum unrounded discrepancy below $9\times10^{-16}$. Within each comparison, query identifiers, targets, and times matched across models; source-scoped participant partitions, observation boundaries, and fitting-only preprocessing records were checked. Final log and concentration outputs were finite. Historical logs do not universally record internal solver failures or floor applications, so these counts remain unverified rather than assumed zero.

\paragraph{Near-ties and optimization.}
For Mavoglurant joint holdout, DeepCt and Uni-PK select epoch zero in all ten runs. Their initialization maps the same training-derived population parameters into their respective dynamics, explaining nearly identical initial predictions; the stored predictions are not bit-identical and checkpoint files are distinct. Training logs record weight changes, but later candidates did not improve validation in this fold. A seed-1 re-inference found no support or query output at the concentration floor before the observation-based scalar correction. This is a targeted diagnostic, not a solver-wide floor audit. The existing equal-budget seed-1 comparison at 120 and 240 epochs is retained: pooled Known-drug SB-hRMSE changed from 1.1100 to 0.7265 for DCM, 1.9479 to 1.6475 for DeepCt, and 1.0903 to 0.8128 for Uni-PK. The corresponding MoDE-PK values were 0.4227 and 0.4277. These results demonstrate sensitivity to optimization budget; the primary panel is a comparison under its stated budget, not a claim that each method has reached its optimal attainable performance. No additional budget sweep was used to select favorable assessment scores.
'''
(D/'essential_results.tex').write_text(section)
p=D/'supplementary.tex';t=p.read_text();mark='\\section{Single-drug learning and weighting sensitivity}';t=t.replace(mark,'\\clearpage\n\\input{essential_results}\n\n'+mark) if '\\input{essential_results}' not in t else t
t=t.replace('Matrix unknown; $\\mu$g/mL, mg/kg, min','Plasma per distributor; $\\mu$g/mL, mg/kg, min');p.write_text(t)
p=D/'main.tex';t=p.read_text();mark='\\subsection{Forecasting under joint drug--source holdout}'
paragraph='The pooled reference model was also examined for sensitivity to the single-query fluorouracil validation source. Across its original five seeds, excluding that source from checkpoint selection changed two selected epochs; full-assessment SB-hRMSE changed from 0.4850 to 0.4411, and pooled log-RMSE from 0.5027 to 0.4629. This exploratory selection analysis is separate from the ten-seed single-drug benchmark and does not replace the primary selection rule (Supplementary Material, Validation-source selection sensitivity and implementation checks).\n\n'
if paragraph not in t:t=t.replace(mark,paragraph+mark)
mark='\\section{Discussion}'
paragraph2='The implementation audit reproduced all primary drug-level table entries from their underlying predictions. It also identified validation-selected initial checkpoints for DeepCt and Uni-PK in the Mavoglurant holdout fold. Together with the existing training-budget sensitivity results, this qualifies comparisons as performance under the stated fitting and selection procedure rather than fully optimized limits of each architecture.\n\n'
if paragraph2 not in t:t=t.replace(mark,mark+'\n'+paragraph2)
p.write_text(t)
shutil.copy2(S/'E01_validation_selection.pdf',D/'E01_validation_selection.pdf')
print(D)
