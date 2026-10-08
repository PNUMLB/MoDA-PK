"""Drug-first plots of the fixed input comparison; no outcome-based case selection."""
from pathlib import Path
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];S=R/'results/summaries/molecular_identity_20261002'
p=pd.read_csv(S/'paired_drug_summary.csv');p=p[p.k==2];d=pd.read_csv(S/'drug_summary.csv');d=d[d.k==2]
drugs=sorted(d.drug_id.unique());names=[x.replace('_',' ').capitalize() for x in drugs]
fig,axs=plt.subplots(1,2,figsize=(11,6),sharey=True)
for ax,track in zip(axs,['Known-drug','Joint drug–source']):
 for offset,ref,color,label in [(-.13,'molecular','#2166ac','Combined minus Uni-Mol'),(.13,'drug_id','#d95f02','Combined minus Drug-ID')]:
  g=p[(p.track==track)&(p.reference==ref)].set_index('drug_id').loc[drugs]
  ax.errorbar(g.delta_rmse_mean,np.arange(len(drugs))+offset,xerr=g.delta_rmse_std,fmt='o',color=color,capsize=2,ms=4,label=label)
 ax.axvline(0,color='black',lw=.8);ax.set_yticks(range(len(drugs)),names);ax.set_title(track);ax.set_xlabel('Difference in drug log-RMSE\nNegative favors the combined input');ax.grid(axis='x',alpha=.2)
axs[0].invert_yaxis();axs[1].legend(loc='lower right',fontsize=8);fig.tight_layout();fig.savefig(S/'Input_paired_drug_effects.pdf');fig.savefig(S/'Input_paired_drug_effects.png',dpi=170);plt.close(fig)
# All values and counts are reported, rather than selecting favorable drugs.
fig,axs=plt.subplots(1,2,figsize=(10,6),sharey=True)
for ax,track in zip(axs,['Known-drug','Joint drug–source']):
 a=d[d.track==track].pivot(index='drug_id',columns='condition',values='rmse_mean').loc[drugs,['molecular','drug_id','molecular_identity']]
 im=ax.imshow(a,cmap='YlOrRd',aspect='auto',vmin=0);ax.set_xticks(range(3),['Uni-Mol','Drug-ID','Combined']);ax.set_yticks(range(11),names);ax.set_title(track)
 for i in range(len(drugs)):
  for j in range(3):ax.text(j,i,f'{a.iloc[i,j]:.3f}',ha='center',va='center',fontsize=8,color='white' if a.iloc[i,j]>a.to_numpy().max()*.6 else 'black')
 fig.colorbar(im,ax=ax,label='Drug log-RMSE')
fig.tight_layout();fig.savefig(S/'Input_drug_metrics.pdf');fig.savefig(S/'Input_drug_metrics.png',dpi=170);plt.close(fig)
