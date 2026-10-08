"""Positive-control simulation: train observers with known prior and independent subjects."""
from pathlib import Path
import sys,json,time,subprocess,os
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np,pandas as pd,torch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from mhap_pk.relational_model import RelationalPK
from mhap_pk.relational import floor_curve
from mhap_pk.evaluation.simulation_map import adapt
B=R/'results/runs/COMPLETION_SUITE/20261002/simulation_crossed'
def save(p,o):p.write_text(json.dumps(o,indent=2))
def generate(n,seed,shift=None,noise=None,timing=None,mismatch=False):
 g=torch.Generator().manual_seed(seed);r=lambda *shape:torch.randn(shape,generator=g,dtype=torch.double)
 mu=torch.tensor([0.,0.,np.log(10),0.,np.log(15)],dtype=torch.double)[None]+.3*r(n,5)
 shifts=(torch.rand(n,generator=g,dtype=torch.double)>.5).double()*.6 if shift is None else torch.full((n,),shift,dtype=torch.double)
 theta=mu+torch.nn.functional.pad(r(n,2)*shifts[:,None],(1,2));events=torch.zeros(n,1,4,dtype=torch.double);events[:,:,1]=100;events[:,0,2]=(torch.arange(n)%2).double()
 early=torch.tensor([.25,.5,1.,2.],dtype=torch.double);late=torch.tensor([2.,3.,4.,6.],dtype=torch.double)
 timingmask=torch.rand(n,generator=g)>.5 if timing is None else torch.full((n,),timing=='early')
 st=torch.where(timingmask[:,None],early,late);qt=torch.tensor([8.,10.,12.,16.,20.,24.],dtype=torch.double).repeat(n,1);truth_events=events
 if mismatch:
  extra=events.clone();extra[:,:,0]=-1.;extra[:,:,1]=50.;truth_events=torch.cat([extra,events],1)
 with torch.no_grad():h=floor_curve(theta,truth_events,st);y=floor_curve(theta,truth_events,qt)
 ns=torch.where(torch.rand(n,generator=g)>.5,.05,.3).double() if noise is None else torch.full((n,),noise,dtype=torch.double)
 return dict(prior_shift=shifts,noise_sd=ns,early=timingmask,mu=mu,theta=theta,z=torch.zeros(n,512,dtype=torch.double),events=events,st=st,sy=h+ns[:,None]*r(n,4),qt=qt,truth=y,observed=y+ns[:,None]*r(n,6))
def args(d,k):
 mask=torch.arange(4)[None]<torch.as_tensor(k)[:,None] if not isinstance(k,int) else (torch.arange(4)[None]<k).repeat(len(d['mu']),1)
 return d['z'],d['events'],d['st'],d['sy'],mask,d['qt']
def pv(d):return d['mu'],torch.zeros(len(d['mu']),32,dtype=torch.double),torch.zeros_like(d['mu'])
def worker(seed):
 torch.set_num_threads(1);out=B/f'seed{seed}';out.mkdir(exist_ok=False);save(out/'manifest.json',dict(status='running',seed=seed));start=time.time();train=generate(192,10000+seed);val=generate(96,20000+seed);models={};log=[]
 for role,d in [('train',train),('validation',val)]:pd.DataFrame(dict(route=d['events'][:,0,2].numpy(),prior_shift=d['prior_shift'].numpy(),noise_sd=d['noise_sd'].numpy(),early=d['early'].numpy())).to_csv(out/(role+'_factor_design.csv'),index=False)
 for name,param,state in [('parameter',True,False),('state',False,True),('combined',True,True)]:
  torch.manual_seed(seed);m=RelationalPK(parameter_update=param,state_update=state).double()
  for p in m.prior.parameters():p.requires_grad_(False)
  opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.003,weight_decay=.0001);rng=np.random.default_rng(1000+seed);best=float('inf')
  for ep in range(121):
   if ep:
    m.train();k=torch.tensor(rng.choice([0,1,2,4],len(train['mu'])));o=m(*args(train,k),prior_values=pv(train));loss=(o['mean']-train['observed']).square().mean()+.001*o['regularization'].mean();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5,error_if_nonfinite=True);opt.step()
   if ep%10==0:
    m.eval()
    with torch.no_grad():v=(m(*args(val,2),prior_values=pv(val))['mean']-val['observed']).square().mean().item()
    log.append(dict(model=name,epoch=ep,validation_mse=v))
    if v<best:best=v;chosen=ep;checkpoint={k:x.clone() for k,x in m.state_dict().items()}
  m.load_state_dict(checkpoint);m.eval();torch.save(checkpoint,out/(name+'.pt'));models[name]=m;save(out/(name+'_selection.json'),dict(epoch=chosen,validation_mse=best,train_profiles=192,validation_profiles=96,prior_fixed=True))
 pd.DataFrame(log).to_csv(out/'validation.csv',index=False);save(out/'selection_complete.json',dict(before_test=True));rows=[];preds=[]
 for mismatch in [False,True]:
  for shift in [0.,.6]:
   for noise in [.05,.3]:
    for timing in ['early','late']:
     d=generate(64,30000+seed,shift,noise,timing,mismatch);base=floor_curve(d['mu'],d['events'],d['qt'])
     for k in [0,1,2,4]:
      aa=args(d,k)
      with torch.no_grad():est={'prior':base}|{n:m(*aa,prior_values=pv(d))['mean'] for n,m in models.items()}
      theta,_=adapt(d['mu'],d['events'],d['st'],d['sy'],aa[4],steps=50,noise_sd=.3,prior_sd=.6);est['cl_vc_map']=floor_curve(theta,d['events'],d['qt'])
      for name,y in est.items():
       y=y.detach();assert torch.isfinite(y).all();mse=(y-d['truth']).square().mean(1);prior_mse=(base-d['truth']).square().mean(1);rows.append(dict(seed=seed,model=name,prior_shift=shift,noise=noise,timing=timing,omitted_dose=mismatch,k=k,latent_log_rmse=float(mse.mean().sqrt()),observed_log_rmse=float((y-d['observed']).square().mean().sqrt()),harm_fraction=float((mse>prior_mse+1e-10).double().mean()),mse_difference=float((mse-prior_mse).mean())))
       for i in range(len(y)):
        for j in range(y.shape[1]):preds.append(dict(seed=seed,model=name,prior_shift=shift,noise=noise,timing=timing,omitted_dose=mismatch,k=k,synthetic_subject=i,query_time=float(d['qt'][i,j]),truth=float(d['truth'][i,j]),observed=float(d['observed'][i,j]),prediction=float(y[i,j])))
 pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False);pd.DataFrame(preds).to_parquet(out/'predictions.parquet',index=False);save(out/'manifest.json',dict(status='completed',seed=seed,seconds=time.time()-start,synthetic_only=True,training_factor_design='independently sampled noise, timing, and mismatch; balanced routes',scope='Mechanism check with fixed known priors; no real-cohort performance claim; omitted prior dose is held-out generator mismatch'))
if __name__=='__main__':
 B.mkdir(parents=True,exist_ok=True)
 if len(sys.argv)>1:worker(int(sys.argv[1]))
 else:
  save(B/'plan.json',dict(seeds=list(range(1,11)),train=192,validation=96,test_per_scenario=64,steps=120,known_prior=True,training_factor_design='independently sampled noise, timing, and mismatch; balanced routes',models=['prior','parameter','state','combined','cl_vc_map'],axes=dict(shift=[0,.6],noise=[.05,.3],timing=['early','late'],k=[0,1,2,4],omitted_dose=[False,True]),selection='k2 validation noisy concentrations; no test tuning; fixed train simulation distribution'))
  def go(seed):
   with (B/f'seed{seed}.log').open('w') as f:r=subprocess.run([sys.executable,__file__,str(seed)],stdout=f,stderr=subprocess.STDOUT,env=os.environ|dict(OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1'));return dict(seed=seed,returncode=r.returncode)
  with ThreadPoolExecutor(max_workers=8) as ex:
   rows=list(ex.map(go,range(1,11)))
  save(B/'status.json',dict(status='completed' if not any(x['returncode'] for x in rows) else 'failed',jobs=rows));print(rows)
