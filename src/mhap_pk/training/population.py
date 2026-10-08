"""Train-partition-only population initialization, matching earlier calibration."""
import numpy as np
import torch
from ..relational import floor_curve

def fit_population(batch,steps=120):
    best=float('inf');best_theta=None
    for volume in [10.,100.,1000.]:
        initial=torch.tensor([0.,0.,np.log(volume),0.,np.log(volume)],dtype=torch.double);theta=initial.clone().requires_grad_(True);opt=torch.optim.Adam([theta],lr=.05)
        for ep in range(steps+1):
            y=floor_curve(theta[None].expand(len(batch.indices),-1),batch.events,batch.query_t);mse=((y-batch.query_y).square()*batch.weights).sum();loss=mse+.0001*(theta-initial).square().mean()
            if not torch.isfinite(loss):raise FloatingPointError('population initialization')
            if float(mse.detach())<best:best=float(mse.detach());best_theta=theta.detach().clone()
            if ep==steps:break
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_([theta],5.);opt.step()
            with torch.no_grad():theta.clamp_(-12.,16.)
    return best_theta,dict(fitting_indices=batch.indices.tolist(),steps=steps,starts=3,train_mse=best,log_parameters=best_theta.tolist())
