"""Render additional experiment summaries at readable publication sizes."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];O=R/'results/summaries/completion_suite_20261002'
ds=pd.read_csv(O/'drug_summary.csv');su=pd.read_csv(O/'simulation_summary.csv')
plt.rcParams.update({'axes.titlesize':10,'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9})
# All-drug paired means; no favorable drug selection.
fig,axs=plt.subplots(2,1,figsize=(7.5,9));modes=['last_mlp','lstm','deep_sets','attention','synthetic','residual_centered'];base_d=ds[(ds.k==2)&(ds.model=='molecular_relation_120')].set_index(['track','drug_id']).log_rmse_mean
for ax,track in zip(axs,['Known-drug','Joint drug–source']):
 t=ds[(ds.track==track)&(ds.k==2)&ds.model.isin(modes)].pivot(index='drug_id',columns='model',values='log_rmse_mean').reindex(columns=modes);diff=t.sub(base_d.loc[track],axis=0);v=diff.to_numpy();lim=max(abs(v).max(),.01);im=ax.imshow(v,cmap='RdBu_r',vmin=-lim,vmax=lim,aspect='auto');ax.set_xticks(range(len(modes)),['MLP','LSTM','Deep Sets','Attention','Synthetic','Centered'],rotation=35,ha='right');ax.set_yticks(range(len(t)),['R-lipoic acid' if x=='r_lipoic_acid' else x.capitalize() for x in t.index]);ax.set_title(track)
 for i in range(len(t)):
  for j in range(len(modes)):ax.text(j,i,(f'{v[i,j]:+.3f}' if abs(v[i,j])<.005 else f'{v[i,j]:+.2f}'),ha='center',va='center',fontsize=8,color='white' if abs(v[i,j])>.6*lim else 'black')
 fig.colorbar(im,ax=ax,shrink=.7,label='Variant minus MoDE-PK drug log-RMSE')
fig.tight_layout();fig.savefig(O/'Additional_drug_effects.pdf');plt.close(fig)
for omitted in [False,True]:
 fig,axs=plt.subplots(2,4,figsize=(10,6),sharey=True)
 for row,timing in enumerate(['early','late']):
  for col,(shift,noise) in enumerate([(0.,.05),(0.,.3),(.6,.05),(.6,.3)]):
   ax=axs[row,col];t=su[(su.prior_shift==shift)&(su.noise==noise)&(su.omitted_dose==omitted)&(su.timing==timing)]
   for mode,g in t.groupby('model'):
    label={'cl_vc_map':'CL/Vc MAP','combined':'Parameter + state','parameter':'Parameter only','prior':'Prior','state':'State only'}[mode]
    ax.plot(g.k,g.latent_log_rmse_mean,marker={'cl_vc_map':'s','combined':'o','parameter':'^','prior':'x','state':'d'}[mode],label=label)
   ax.set(title=f'{timing.capitalize()}: shift {shift}, noise {noise}',xlabel='Observed time points k');ax.grid(alpha=.2);ax.set_xticks([0,1,2,4])
   if col==0:ax.set_ylabel('Latent log-RMSE')
 handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=5,fontsize=10,frameon=False);fig.tight_layout(rect=(0,.05,1,1));fig.savefig(O/('Synthetic_omitted_dose.pdf' if omitted else 'Synthetic_positive_control.pdf'));plt.close(fig)
