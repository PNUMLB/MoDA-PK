"""Synthetic shape/finite-value check, not a clinical prediction example."""
import torch
from moda_pk import build_model

def main():
    torch.set_num_threads(1);torch.manual_seed(1)
    population=torch.log(torch.tensor([1.,1.,10.,1.,10.],dtype=torch.double))
    model=build_model(11,population).eval()
    z=torch.zeros((2,512),dtype=torch.double)
    events=torch.tensor([[[0.,1.,0.,0.]],[[0.,1.,1.,0.]]],dtype=torch.double)
    times=torch.tensor([[.25,.5,1.],[.25,.5,1.]],dtype=torch.double)
    labels=torch.log(torch.full((2,3),.05,dtype=torch.double))
    mask=torch.ones((2,3),dtype=torch.bool);query=torch.tensor([[2.,4.],[2.,4.]],dtype=torch.double);ids=torch.tensor([0,1])
    with torch.no_grad():result=model(z,events,times,labels,mask,query,ids)
    assert result['mean'].shape==(2,2) and torch.isfinite(result['mean']).all()
    assert sum(p.numel() for p in model.parameters())==56751
    print('PASS: output shape (2,2), finite predictions, 56,751 trainable parameters.')
if __name__=='__main__':main()
