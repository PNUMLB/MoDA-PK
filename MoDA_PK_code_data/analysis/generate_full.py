"""Portable current-manuscript tables and data figures from the results package."""
from pathlib import Path
import argparse,json,re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NAMES={'ours':'MoDA-PK','poppk_vi_map':'PopPK+MAP','dcm':'DCM','unipk':'Uni-PK','deepct':'DeepCt'}
VARIANTS=['FULL','NO-UPDATE','PARAM-ONLY','AMOUNT-ONLY','ID-ONLY','MOL-ONLY']

def metrics(frame):
 d=frame.copy();d['se']=(d.y_pred_log-d.y_true_log)**2;d['ae']=(d.y_pred_log-d.y_true_log).abs();d['linear_se']=(np.exp(d.y_pred_log)-np.exp(d.y_true_log))**2
 keys=['condition','seed','source_id','drug_id']
 p=d.groupby(keys+['participant_id'])[['se','ae','linear_se']].mean().reset_index()
 g=p.groupby(keys)[['se','ae','linear_se']].mean().reset_index();g['log_rmse']=np.sqrt(g.se);g['log_mae']=g.ae;g['linear_rmse']=np.sqrt(g.linear_se)
 r=[]
 for key,v in d.groupby(keys):
  w=1/v.groupby('participant_id').y_true_log.transform('size');mu=np.average(v.y_true_log,weights=w);den=((v.y_true_log-mu)**2*w).sum()
  r.append(dict(zip(keys,key),r2_log=1-(v.se*w).sum()/den if den>0 else np.nan))
 return g.merge(pd.DataFrame(r),on=keys,validate='one_to_one')

def latex_table(data,columns,path):
 # data indexed by drug, columns (mean/std, model)
 lines=[r'\begin{tabular}{l'+'r'*len(columns)+r'}\toprule','Drug & '+' & '.join(NAMES.get(c,c) for c in columns)+r'\\\midrule']
 for drug,row in data.iterrows():
  best=min(row[('mean',c)] for c in columns);cells=[]
  for c in columns:
   val=f"{row[('mean',c)]:.3f} $\\pm$ {row[('std',c)]:.3f}"
   if row[('mean',c)]==best:val=r'\textbf{'+val+'}'
   cells.append(val)
  lines.append(drug.replace('_',' ').capitalize()+' & '+' & '.join(cells)+r'\\')
 lines.append(r'\bottomrule\end{tabular}');path.write_text('\n'.join(lines)+'\n')

def savefig(fig,path):
 fig.savefig(path.with_suffix('.pdf'));fig.savefig(path.with_suffix('.png'),dpi=200);plt.close(fig)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,required=True);ap.add_argument('--output',type=Path,default=Path('generated'));ap.add_argument('--tex-source',type=Path);ap.add_argument('--include-dose-controller-test',action='store_true',help='Plot the optional 64D controller test');ap.add_argument('--include-transfer',action='store_true',help='Generate the archived transfer plot, excluded from the current manuscript')
 a=ap.parse_args();root=a.results.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
 primary=root/'01_primary_11drugs';essential=root/'07_essential_ablation_11drugs'
 plt.rcParams.update({'font.size':8,'pdf.fonttype':42,'ps.fonttype':42})
 counts=pd.read_csv(primary/'dataset_counts_k3.csv');counts['route']=counts.drug_id.map({'buprenorphine':'IV infusion','fentanyl':'IV infusion','mavoglurant':'IV infusion','remifentanil':'IV infusion','fluorouracil':'IV bolus','indomethacin':'IV bolus','empagliflozin':'Oral','r_lipoic_acid':'Oral','veralipride':'Oral','warfarin':'Oral','xanthohumol':'Oral'});counts.to_csv(out/'Table1_dataset.csv',index=False)
 scopes={};tables={}
 for scope,file in [('single','single_predictions.parquet'),('integrated','full_predictions.parquet'),('low','low_predictions.parquet')]:
  raw=pd.read_parquet(primary/file);raw=raw[raw.k==3]
  if scope=='low':target=metrics(raw[raw.scope=='target']);raw=raw[raw.scope=='pooled']
  d=metrics(raw);models=list(NAMES) if scope=='single' else ['ours','dcm','unipk','deepct'];d=d[d.condition.isin(models)];scopes[scope]=d
  table=d.groupby(['drug_id','condition']).log_rmse.agg(['mean','std']).unstack('condition')
  table.to_csv(out/f'Table{dict(single=2,integrated=3,low=4)[scope]}_{scope}.csv')
  latex_table(table,models,out/f'Table{dict(single=2,integrated=3,low=4)[scope]}_{scope}.tex');tables[scope]=table
 target_table=target.groupby(['drug_id','condition']).log_rmse.agg(['mean','std']).unstack('condition');target_table.to_csv(out/'TableS6_target_only.csv');latex_table(target_table,list(NAMES),out/'TableS6_target_only.tex')
 # Observation-count tables use the archived canonical hierarchical aggregate outputs.
 for num,name in [(3,'single'),(4,'full'),(5,'low')]:
  d=pd.read_csv(primary/f'{name}_cohort_summary.csv')
  if 'scope' in d:d=d[d.scope=='pooled']
  d.to_csv(out/f'TableS{num}_history.csv',index=False)
 # Main Figures 6--8: participant-weighted metric summaries over the five original fits.
 drugs=sorted(counts.drug_id)
 for number,metric,title in [(6,'log_mae','Log-MAE'),(7,'linear_rmse','Concentration-scale RMSE'),(8,'r2_log','Log-concentration R²')]:
  fig,axs=plt.subplots(1,3,figsize=(13,5.5),layout='constrained')
  for ax,(scope,d) in zip(axs,scopes.items()):
   models=list(d.condition.unique());n=len(models);y=np.arange(len(drugs));height=.75/n
   for j,c in enumerate(models):
    gg=d[d.condition==c].groupby('drug_id')[metric].agg(['mean','min','max']).reindex(drugs);center=y+(j-(n-1)/2)*height
    ax.barh(center,gg['mean'],height=height,label=NAMES[c],xerr=np.maximum(0,np.vstack([gg['mean']-gg['min'],gg['max']-gg['mean']])),error_kw={'elinewidth':.5})
   ax.set_yticks(y,[x.replace('_',' ') for x in drugs]);ax.invert_yaxis();ax.set_title(scope);ax.set_xlabel(title)
   if metric=='linear_rmse':ax.set_xscale('log')
   if metric=='r2_log':ax.set_xscale('symlog',linthresh=1);ax.axvline(0,color='black',lw=.5)
  axs[-1].legend(fontsize=7);savefig(fig,out/f'Figure{number}_{metric}')
 # PK case figures use the previously selected cases, never choose new favourable profiles.
 curves=pd.read_parquet(primary/'curve_predictions.parquet');supports=pd.read_csv(primary/'curve_supports.csv');allobs=pd.read_parquet(primary/'single_predictions.parquet');allobs=allobs[(allobs.condition=='ours')&(allobs.seed==1)&(allobs.k==0)]
 for number,regime in [(3,'single'),(4,'integrated'),(5,'low')]:
  fig,axs=plt.subplots(3,4,figsize=(12,8),layout='constrained')
  for ax,drug in zip(axs.flat,drugs):
   d=curves[(curves.drug_id==drug)&(curves.data_regime==regime)];d=d[d.scope=='pooled'] if regime=='low' and (d.scope=='pooled').any() else d;pid=d.profile_id.iloc[0];support=supports[supports.drug_id==drug].iloc[0];ids=set(json.loads(support.support_ids))
   obs=allobs[(allobs.drug_id==drug)&(allobs.profile_id==pid)].drop_duplicates('observation_id');is_support=obs.observation_id.isin(ids)
   ax.scatter(obs.loc[is_support,'query_time'],np.exp(obs.loc[is_support,'y_true_log']),marker='s',color='black',s=12,label='Supplied')
   ax.scatter(obs.loc[~is_support,'query_time'],np.exp(obs.loc[~is_support,'y_true_log']),facecolors='none',edgecolors='black',s=12,label='Future')
   for model,g in d.groupby('condition'):
    line=g.groupby('query_time').y_pred_log.mean();ax.plot(line.index,np.exp(line),label=NAMES.get(model,model),lw=1)
   ax.axvline(support.cutoff,ls=':',color='grey',lw=.8);ax.set_yscale('log');ax.set_title(drug.replace('_',' '));ax.set_xlabel('Time (h)');ax.set_ylabel('Concentration (µmol/L)')
  axs.flat[-1].axis('off');handles,labels=axs.flat[0].get_legend_handles_labels();axs.flat[-1].legend(handles,labels,fontsize=7,loc='center');savefig(fig,out/f'Figure{number}_PK_{regime}')
 # Essential component/conditioning summaries and plots, from participant-level scores.
 d=pd.read_csv(essential/'ablation_drug_scores.csv');s=pd.read_csv(essential/'ablation_overall_scores.csv');summary=s.groupby('variant').sb_hrmse.agg(['mean','std']).reindex(VARIANTS);summary.to_csv(out/'Table5_essential_ablation.csv')
 table5=summary[['mean','std']].copy();table5.columns=pd.MultiIndex.from_tuples([('mean','SB-hRMSE'),('std','SB-hRMSE')]);latex_table(table5,['SB-hRMSE'],out/'Table5_essential_ablation.tex')
 for num,variants in [(7,VARIANTS[:4]),(8,['FULL','ID-ONLY','MOL-ONLY'])]:
  table=d[d.variant.isin(variants)].groupby(['drug_id','variant']).log_rmse.agg(['mean','std']).unstack('variant');table.to_csv(out/f'TableS{num}_essential.csv');latex_table(table,variants,out/f'TableS{num}_essential.tex')
 fig,ax=plt.subplots(figsize=(7,3.8),layout='constrained');ax.errorbar(summary['mean'],range(6),xerr=summary['std'],fmt='o',capsize=4);ax.set_yticks(range(6),VARIANTS);ax.invert_yaxis();ax.set_xlabel('SB-hRMSE (lower is better)');ax.set_title('Mean ± sample fitting SD; five seeds');savefig(fig,out/'FigureS1_essential_ablation')
 if a.include_transfer:
  delta=pd.read_csv(essential/'cross_drug_transfer_delta.csv');fig,ax=plt.subplots(figsize=(7,4.6),layout='constrained');ax.barh(delta.drug_id.str.replace('_',' '),delta.delta_mean,color=['#0072B2' if x<0 else '#D55E00' for x in delta.delta_mean]);ax.axvline(0,color='black',lw=.7);ax.invert_yaxis();ax.set_xlabel('Integrated − target-only log-RMSE');ax.set_title('Matched two-participant training; five-seed mean difference');savefig(fig,out/'transfer_delta_optional')
 if a.include_dose_controller_test:
  folder=root/'08_dose_controller_subset_test_11drugs';summary_dose=pd.read_csv(folder/'summary.csv').set_index('variant');dd=pd.read_csv(folder/'drug_mean_differences.csv')
  summary_dose.to_csv(out/'dose_controller_comparison.csv')
  fig,axs=plt.subplots(1,2,figsize=(9,4.5),layout='constrained');axs[0].errorbar(summary_dose['mean'],range(2),xerr=summary_dose['std'],fmt='o',capsize=4);axs[0].set_yticks(range(2),summary_dose.index);axs[0].set_xlabel('SB-hRMSE; five-seed mean ± fitting SD')
  axs[1].barh(dd.drug_id.str.replace('_',' '),dd.delta,color=['#0072B2' if x<0 else '#D55E00' for x in dd.delta]);axs[1].axvline(0,color='black',lw=.7);axs[1].set_xlabel('64D gate minus FULL log-RMSE');axs[1].invert_yaxis();savefig(fig,out/'dose_controller_subset_optional')
 # Verify recomputed table cells against the current manuscript when provided.
 checked=0
 if a.tex_source:
  main=(a.tex_source/'main.tex').read_text();su=(a.tex_source/'supplementary.tex').read_text()
  for text,label,table,models in [(main,'tab:full-single',tables['single'],list(NAMES)),(main,'tab:full-integrated',tables['integrated'],['ours','dcm','unipk','deepct']),(main,'tab:low-integrated',tables['low'],['ours','dcm','unipk','deepct']),(su,'tab:low-single',target_table,list(NAMES))]:
   block=next(b for b in re.findall(r'\\begin\{table\}.*?\\end\{table\}',text,re.S) if '\\label{'+label+'}' in b)
   for line in block.splitlines():
    drug=line.split('&')[0].strip().lower().replace('r-lipoic acid','r_lipoic_acid')
    if drug not in table.index:continue
    pairs=re.findall(r'(\d+\.\d+)\s*\$\\pm\$\s*(\d+\.\d+)',line);assert len(pairs)==len(models)
    for c,(mean,sd) in zip(models,pairs):assert mean==f"{table.loc[drug,('mean',c)]:.3f}" and sd==f"{table.loc[drug,('std',c)]:.3f}";checked+=1
  extra=(a.tex_source/'table_essential_ablation.tex').read_text()
  for line in extra.splitlines():
   v=line.split('&')[0].strip()
   if v in VARIANTS:
    pair=re.findall(r'(\d+\.\d+)\s*\$\\pm\$\s*(\d+\.\d+)',line);assert len(pair)==1
    assert pair[0]==(f"{summary.loc[v,'mean']:.3f}",f"{summary.loc[v,'std']:.3f}");checked+=1
  for filename,variants in [('table_essential_components.tex',VARIANTS[:4]),('table_essential_conditioning.tex',['FULL','ID-ONLY','MOL-ONLY'])]:
   vv=d[d.variant.isin(variants)].groupby(['drug_id','variant']).log_rmse.agg(['mean','std']).unstack('variant')
   for line in (a.tex_source/filename).read_text().splitlines():
    key=line.split('&')[0].strip().lower().replace('r-lipoic acid','r lipoic acid');drug=next((x for x in vv.index if x.replace('_',' ')==key),None)
    if drug is None:continue
    pairs=re.findall(r'(\d+\.\d+)\s*\$\\pm\$\s*(\d+\.\d+)',line);assert len(pairs)==len(variants)
    for v,(mean,sd) in zip(variants,pairs):assert mean==f"{vv.loc[drug,('mean',v)]:.3f}" and sd==f"{vv.loc[drug,('std',v)]:.3f}";checked+=1
 (out/'verification.json').write_text(json.dumps(dict(status='PASS',manuscript_mean_sd_pairs_checked=checked,essential_seed_count=int(s.seed.nunique()),training_required=False,source='saved result predictions and canonical summaries',note='Numerical content reproduced; layouts are regenerated, not pixel-identical. Figures 1/2 are conceptual diagrams provided in the TeX package. PK plots connect recorded future prediction times; no dense interpolation or dose annotations are invented.'),indent=2)+'\n')
 print('Generated tables and figures:',out,'; verified mean/SD pairs:',checked)
if __name__=='__main__':main()
