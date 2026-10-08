"""Training-anchor residual heads; constants are exploratory, not uncertainty estimates."""
import torch
from torch import nn
from .low_data import MolecularDrugOffset
from ...relational_model import CalibratedPrior

class SimilarityTransferPrior(MolecularDrugOffset):
    """Own residual shrinks to leave-self-out molecular neighbors as n decreases.

    Anchors use the existing train-fold-standardized frozen Uni-Mol coordinates.
    Softmax of cosine / temperature defines convex weights. Unknown IDs never
    index a target-specific trainable parameter. All residuals bounded by 2.
    """
    def __init__(self, anchors, counts, temperature=.25, pseudo_count=4.):
        if anchors.ndim != 2 or anchors.shape[1] != 512 or len(anchors)<1:
            raise ValueError('Require nonempty [training drugs,512] anchors')
        if counts.shape != (len(anchors),) or (counts<=0).any():
            raise ValueError('Positive training participant counts required')
        if temperature<=0 or pseudo_count<=0:raise ValueError('Positive constants required')
        super().__init__(len(anchors))
        self.register_buffer('anchors',anchors.clone())
        self.register_buffer('counts',counts.clone())
        self.temperature=temperature;self.pseudo_count=pseudo_count

    def mixing_weights(self,z,drug_indices):
        if drug_indices is None:raise ValueError('Explicit training ID or -1 required')
        if ((drug_indices < -1)|(drug_indices>=len(self.anchors))).any():raise ValueError('Invalid training ID')
        known=drug_indices>=0;idx=drug_indices.clamp_min(0)
        logits=nn.functional.normalize(z,dim=-1)@nn.functional.normalize(self.anchors,dim=-1).T/self.temperature
        if len(self.anchors)>1:
            logits=logits.masked_fill(nn.functional.one_hot(idx,len(self.anchors)).bool()&known[:,None],-torch.inf)
        donor=logits.softmax(-1)
        own=nn.functional.one_hot(idx,len(self.anchors)).to(z.dtype)*known[:,None]
        strength=torch.where(known,self.counts[idx]/(self.counts[idx]+self.pseudo_count),0.)
        return strength[:,None]*own+(1-strength[:,None])*donor

    def forward(self,z,drug_indices=None):
        mu,context,deviation=CalibratedPrior.forward(self,z,drug_indices)
        offset=self.mixing_weights(z,drug_indices)@(2*self.offset.weight.tanh())
        return mu+offset,context,deviation+offset
