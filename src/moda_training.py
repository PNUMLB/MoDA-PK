"""Original full-batch fitting loop for externally prepared authorized tensors."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from moda_pk import build_model
FLAGS={'FULL':(1,1,1,1),'NO-UPDATE':(1,1,0,0),'PARAM-ONLY':(1,1,1,0),'AMOUNT-ONLY':(1,1,0,1),'ID-ONLY':(0,1,1,1),'MOL-ONLY':(1,0,1,1)}

def inputs(b,slots):
    rows=torch.arange(len(b['ids']))
    return (b['molecular'],b['events'],b['support_t'][rows,slots],b['support_y'][rows,slots],b['support_mask'][rows,slots],b['query_t'],b['ids'])

def score(pred,b,metadata):
    values=[]
    for i,row in metadata.iterrows():
        mask=b['query_mask'][i];se=(pred[i,mask]-b['query_y'][i,mask]).square()
        values.append(dict(source=row.source_id,drug=row.drug_id,participant=row.participant_id,total=float(se.sum()),n=len(se)))
    d=pd.DataFrame(values).groupby(['source','drug','participant'])[['total','n']].sum();d['mse']=d.total/d.n
    return float(d.groupby(['source','drug']).mse.mean().pow(.5).groupby('source').mean().mean())

def fit(payload,variant,seed,output):
    torch.set_num_threads(1);torch.manual_seed(seed);rng=np.random.default_rng(1000+seed)
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    tr,va=payload['training'],payload['validation'];meta=pd.DataFrame(payload['validation_metadata'])
    flags=dict(zip(['use_molecular','use_drug_id','use_cl_vc_update','use_central_amount_update'],FLAGS[variant]))
    model=build_model(payload['n_drugs'],payload['population'],**flags);optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.0001)
    assert abs(float(tr['weights'].sum())-1)<1e-10
    assert tr['support_t'].shape[1]==4 and va['support_t'].shape[1]==4
    best=float('inf');history=[];training=[]
    for epoch in range(121):
        if epoch:
            model.train();optimizer.zero_grad(set_to_none=True);slots=torch.tensor(rng.integers(0,4,len(tr['ids'])));o=model(*inputs(tr,slots));w=tr['weights'].sum(1);_,_,dev=model.prior(tr['molecular'],tr['ids'])
            loss=((o['mean']-tr['query_y']).square()*tr['weights']).sum()+.001*(o['regularization']*w).sum()+.0001*(dev.square().mean(1)*w).sum()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);optimizer.step();training.append(dict(epoch=epoch,loss=float(loss.detach()),gradient_norm=float(grad)))
        if epoch%10==0:
            model.eval()
            with torch.no_grad():value=score(model(*inputs(va,torch.full((len(va['ids']),),2,dtype=torch.long)))['mean'],va,meta)
            if not np.isfinite(value):raise FloatingPointError('Nonfinite validation score')
            history.append(dict(epoch=epoch,validation_sb_hrmse=value))
            if value<best:best=value;selected=epoch;torch.save(model.state_dict(),out/'selected.pt')
    pd.DataFrame(training).to_csv(out/'training.csv',index=False);pd.DataFrame(history).to_csv(out/'validation.csv',index=False)
    (out/'selection.json').write_text(json.dumps(dict(variant=variant,seed=seed,epoch=selected,validation_sb_hrmse=best,test_evaluated=False),indent=2)+'\n')
