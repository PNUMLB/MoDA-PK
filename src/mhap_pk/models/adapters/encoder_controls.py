"""History encoder controls with unchanged dose dynamics and update heads."""
import torch
from torch import nn
from ...relational_model import RelationalPK

class HistoryCell(nn.Module):
    def __init__(self,kind,inputs=43,hidden=32):
        super().__init__();self.kind=kind;self.hidden=hidden
        if kind=='lstm':self.cell=nn.LSTMCell(inputs,hidden)
        elif kind in ['last_mlp','deep_sets']:self.embed=nn.Sequential(nn.Linear(inputs,hidden),nn.SiLU(),nn.Linear(hidden,hidden))
        elif kind=='attention':
            self.embed=nn.Linear(inputs,hidden);self.block=nn.TransformerEncoderLayer(hidden,2,64,dropout=0.,activation='gelu',batch_first=True)
        else:raise ValueError(kind)
    def reset_history(self):self.tokens=[];self.masks=[];self.cell_state=None
    def set_active(self,active):self.active=active
    def forward(self,x,h):
        if self.kind=='lstm':
            c=torch.zeros_like(h) if self.cell_state is None else self.cell_state
            out,new_c=self.cell(x,(h,c));self.cell_state=torch.where(self.active[:,None],new_c,c);return out
        token=self.embed(x)
        if self.kind=='last_mlp':return token
        self.tokens.append(token);self.masks.append(self.active)
        t=torch.stack(self.tokens,1);mask=torch.stack(self.masks,1)
        if self.kind=='attention':
            # Empty profiles are padded with one harmless dummy token; ignored downstream.
            safe=mask.clone();safe[:,0]|=~mask.any(1);t=self.block(t,src_key_padding_mask=~safe)
        return (t*mask[:,:,None]).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)

class EncoderControlPK(RelationalPK):
    def __init__(self,kind):
        super().__init__();self.gru=HistoryCell(kind)
