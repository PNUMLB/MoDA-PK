"""Public aggregate-only figures/tables; no patient records or training needed."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]

def table(frame,models,path):
    rows=[r'\begin{tabular}{l'+'r'*len(models)+r'}\toprule','Drug & '+' & '.join(models)+r'\\\midrule']
    for drug,g in frame.groupby('drug_id'):
        indexed=g.set_index('condition');best=g.log_rmse_mean.min();cells=[]
        for model in models:
            r=indexed.loc[model];text=f'{r.log_rmse_mean:.3f} $\\pm$ {r.log_rmse_std:.3f}'
            if r.log_rmse_mean==best:text=r'\textbf{'+text+'}'
            cells.append(text)
        rows.append(drug.replace('_',' ')+' & '+' & '.join(cells)+r'\\')
    rows.append(r'\bottomrule\end{tabular}');path.write_text('\n'.join(rows)+'\n')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=ROOT/'outputs');args=ap.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    p=ROOT/'data/primary';e=ROOT/'data/essential'
    for num,filename,models in [(2,'single_drug_summary.csv',['ours','poppk_vi_map','dcm','unipk','deepct']),(3,'full_drug_summary.csv',['ours','dcm','unipk','deepct']),(4,'low_drug_summary.csv',['ours','dcm','unipk','deepct'])]:
        d=pd.read_csv(p/filename);d=d[d.k==3]
        if 'scope' in d:d=d[d.scope=='pooled']
        d=d[d.condition.isin(models)];d.to_csv(out/f'Table{num}.csv',index=False);table(d,models,out/f'Table{num}.tex')
    low=pd.read_csv(p/'low_drug_summary.csv');low=low[(low.k==3)&(low.scope=='target')];low.to_csv(out/'TableS6.csv',index=False);table(low,['ours','poppk_vi_map','dcm','unipk','deepct'],out/'TableS6.tex')
    d=pd.read_csv(e/'ablation_drug_scores.csv');s=pd.read_csv(e/'ablation_overall_scores.csv')
    assert set(s.seed)=={1,2,3,4,5} and s.groupby('variant').seed.nunique().eq(5).all()
    expected=pd.read_csv(e/'ablation_summary.csv').set_index('variant');summary=s.groupby('variant').sb_hrmse.agg(['mean','std','count']).reindex(expected.index)
    np.testing.assert_allclose(summary[['mean','std']],expected[['mean','std']],rtol=1e-12,atol=1e-12);summary.to_csv(out/'Table5.csv')
    for num,variants in [(7,['FULL','NO-UPDATE','PARAM-ONLY','AMOUNT-ONLY']),(8,['FULL','ID-ONLY','MOL-ONLY'])]:
        sub=d[d.variant.isin(variants)].groupby(['drug_id','variant']).log_rmse.agg(['mean','std']).unstack('variant');sub.to_csv(out/f'TableS{num}.csv')
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42});fig,ax=plt.subplots(figsize=(7,4),layout='constrained');ax.errorbar(summary['mean'],range(len(summary)),xerr=summary['std'],fmt='o',capsize=4);ax.set_yticks(range(len(summary)),summary.index);ax.invert_yaxis();ax.set_xlabel('SB-hRMSE; mean ± fitting SD, five seeds')
    for ext in ['png','pdf']:fig.savefig(out/f'FigureS1_ablation.{ext}',dpi=200)
    plt.close(fig)
    paired=s.pivot(index='seed',columns='variant',values='sb_hrmse');interaction=paired['PARAM-ONLY']+paired['AMOUNT-ONLY']-paired['NO-UPDATE']-paired.FULL;interaction.to_csv(out/'interaction_by_seed.csv',header=['I'])
    transfer=pd.read_csv(e/'cross_drug_transfer_delta.csv');transfer.to_csv(out/'transfer_delta.csv',index=False)
    # This plot is an optional numerical diagnostic, excluded from the current manuscript.
    fig,ax=plt.subplots(figsize=(7,4.5),layout='constrained');ax.barh(transfer.drug_id.str.replace('_',' '),transfer.delta_mean,color=['#0072B2' if v<0 else '#D55E00' for v in transfer.delta_mean]);ax.axvline(0,color='black',lw=.7);ax.invert_yaxis();ax.set_xlabel('Integrated − target-only log-RMSE');fig.savefig(out/'transfer_diagnostic.png',dpi=200);plt.close(fig)
    panels=pd.read_csv(p/'publication_metric_panels.csv')
    for num,metric in [(6,'log_mae'),(7,'linear_rmse'),(8,'r2_log')]:
        fig,axes=plt.subplots(1,3,figsize=(13,5),layout='constrained')
        for ax,scope in zip(axes,['single','integrated','low']):
            x=panels[(panels.experiment==scope)&(panels.k==3)];drugs=sorted(x.drug_id.unique());models=list(x.condition.unique());height=.8/len(models)
            for j,model in enumerate(models):
                g=x[x.condition==model].groupby('drug_id')[metric].agg(['mean','min','max']).reindex(drugs);err=np.maximum(0,np.vstack([g['mean']-g['min'],g['max']-g['mean']]))
                ax.barh(np.arange(len(drugs))+(j-(len(models)-1)/2)*height,g['mean'],height=height,xerr=err,label=model,error_kw={'elinewidth':.5})
            ax.set_yticks(range(len(drugs)),[v.replace('_',' ') for v in drugs]);ax.invert_yaxis();ax.set_title(scope);ax.set_xlabel(metric)
            if metric=='linear_rmse':ax.set_xscale('log')
            if metric=='r2_log':ax.set_xscale('symlog',linthresh=1)
        axes[-1].legend(fontsize=7);fig.savefig(out/f'Figure{num}_{metric}.pdf');plt.close(fig)
    (out/'verification.json').write_text(json.dumps(dict(status='PASS',essential_seeds=5,aggregate_ablation_values_reproduced=True,row_level_data_required=False),indent=2)+'\n')
    print('PASS: aggregate tables and figures generated in',out)
if __name__=='__main__':main()
