"""Training-only parametric PK bootstrap; not independent clinical observations."""
from dataclasses import replace
import torch,json
from ..relational import floor_curve
from ..evaluation.simulation_map import adapt

def generate(ps,meta,tr,b,pop,seed,out):
    theta=pop[None].repeat(len(tr),1)
    st=torch.cat([b.support_t[:,3],b.query_t],1);sy=torch.cat([b.support_y[:,3],b.query_y],1);mask=torch.cat([b.support_mask[:,3],b.query_mask],1)
    fit,diag=adapt(theta,b.events,st,sy,mask,steps=100,prior_sd=.6,noise_sd=.3)
    with torch.no_grad():residual=sy-floor_curve(fit,b.events,st)
    rng=torch.Generator().manual_seed(20261002+seed);scale=torch.zeros(len(tr),2,dtype=torch.double);noise=torch.zeros(len(tr),dtype=torch.double);stats={}
    for drug in sorted(meta.iloc[tr].drug_id.unique()):
        ix=torch.tensor(meta.iloc[tr].drug_id.to_numpy()==drug);sd=fit[ix,1:3].std(0,unbiased=False).clamp(.05,.3);ns=residual[ix][mask[ix]].std(unbiased=False).clamp(.05,.5);scale[ix]=sd;noise[ix]=ns;stats[drug]=dict(parameter_sd=sd.tolist(),noise_sd=float(ns),training_profiles=int(ix.sum()))
    perturbed=fit+torch.nn.functional.pad(torch.randn((len(tr),2),generator=rng,dtype=torch.double)*scale,(1,2))
    with torch.no_grad():
        h=floor_curve(perturbed,b.events,b.support_t[:,3]);h=h+noise[:,None]*torch.randn(h.shape,generator=rng,dtype=torch.double);q=floor_curve(perturbed,b.events,b.query_t);q=q+noise[:,None]*torch.randn(q.shape,generator=rng,dtype=torch.double)
    assert torch.isfinite(q[b.query_mask]).all()
    # Every k uses a prefix of the same generated first-four time blocks.
    synew=b.support_y.clone()
    for ki in range(4):synew[:,ki]=h
    (out/'synthetic_generator.json').write_text(json.dumps(dict(fitting_indices=tr.tolist(),seed=seed,training_only=True,teacher='Population plus train-profile CL/Vc MAP from train concentrations only; diagonal jitter and noise clipped to declared bounds',statistics=stats,loss='0.5 real + 0.5 synthetic prediction loss; original real regularization and relation losses',synthetic_profiles=len(tr),no_validation_or_test_fitting=True),indent=2))
    torch.save(dict(teacher_parameters=fit,generated_parameters=perturbed,support=h,query=q),out/'synthetic_training_LOCAL_ONLY.pt')
    return replace(b,support_y=synew,query_y=q)
