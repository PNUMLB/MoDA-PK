"""Untrained soft-gating prototypes over frozen sequential/information experts.

Two fusion modes share a support-conditioned scalar gate:
  output: convex combination of the two log forecasts (an ensemble);
  dynamics: combine log PK parameters AND aligned state estimates, then one ODE.

This module does not implement a shared learned molecular prior or train the
gate. Its default zero-initialized logistic gate is exactly 0.5. Expert weights
are frozen; a later runner must fit the gate using leakage-free OOF episodes.
"""
import torch
from torch import nn
from .iterated_information import condition as iterated_condition
from .relational import floor_curve, floor_concentration
from .state_ode import advance
from .transfer_calibration import pool_support


FEATURE_NAMES = (
    'log1p_nearest_fitting_embedding_rms_distance', 'log1p_observed_time_blocks',
    'log1p_support_span_h', 'log1p_elapsed_since_first_dose_h',
    'sequential_prior_signed_residual', 'sequential_prior_rms_residual',
    'information_prior_signed_residual', 'information_prior_rms_residual',
    'prior_mean_absolute_disagreement',
)


def mix_dynamics(theta_seq, state_seq, theta_info, state_info, gate):
    """Mix at the SAME time and event convention; states are [depot,central,peripheral].

    Circulating amounts are converted to concentrations using each expert's own
    volumes, mixed, then converted with the hybrid volumes. This retains both
    endpoint models and a convex central concentration at the conditioning time.
    It is a fused state estimate, not physical administration or identified PK.
    """
    if theta_seq.shape != theta_info.shape or theta_seq.shape[-1] != 5:
        raise ValueError('Aligned five-dimensional log PK parameters required')
    if state_seq.shape != state_info.shape or state_seq.shape != (len(theta_seq), 3):
        raise ValueError('Aligned three-dimensional amount states required')
    if gate.shape != (len(theta_seq),) or not torch.isfinite(gate).all() or ((gate < 0)|(gate > 1)).any():
        raise ValueError('One finite gate in [0,1] per profile required')
    g = gate[:, None]
    theta = (1-g)*theta_seq+g*theta_info
    volumes_seq = torch.stack([torch.ones_like(gate), theta_seq[:,2].exp(), theta_seq[:,4].exp()], -1)
    volumes_info = torch.stack([torch.ones_like(gate), theta_info[:,2].exp(), theta_info[:,4].exp()], -1)
    volumes_mix = torch.stack([torch.ones_like(gate), theta[:,2].exp(), theta[:,4].exp()], -1)
    amounts = ((1-g)*(state_seq/volumes_seq)+g*(state_info/volumes_info))*volumes_mix
    return theta, amounts


class FrozenExpertSoftHybrid(nn.Module):
    """Trainable 10-parameter gate; two frozen experts with separate priors.

    Gate input has nine low-dimensional features, not raw 512-D molecular IDs.
    Molecular information still drives both experts and the registry distance.
    Gate features and weights must be fitted using the outer fitting data only.
    This prototype's initialization is not a trained reliability estimator.
    """
    def __init__(self, sequential, information, fitting_vectors, information_config=None, fusion='dynamics'):
        super().__init__()
        if fusion not in ['output','dynamics']:
            raise ValueError('Expected output or dynamics fusion')
        if fitting_vectors.ndim != 2 or fitting_vectors.shape[1] != 512 or not len(fitting_vectors):
            raise ValueError('Nonempty fitting-only molecular registry required')
        if not torch.isfinite(fitting_vectors).all(): raise ValueError('Registry must be finite')
        self.sequential, self.information, self.fusion = sequential, information, fusion
        self.information_config = dict(information_config or dict(name='legacy',kind='legacy'))
        if self.information_config['kind'] not in ['legacy','iterated']:
            raise ValueError('Unknown information observer')
        for expert in [self.sequential, self.information]:
            expert.eval()
            for parameter in expert.parameters(): parameter.requires_grad_(False)
        self.register_buffer('fitting_vectors', fitting_vectors.detach().clone())
        self.gate = nn.Linear(len(FEATURE_NAMES),1).to(dtype=fitting_vectors.dtype,device=fitting_vectors.device)
        nn.init.zeros_(self.gate.weight);nn.init.zeros_(self.gate.bias)

    def train(self, mode=True):
        super().train(mode)
        self.sequential.eval();self.information.eval()
        return self

    def _features(self, z, events, st, sy, sm, mu_seq, mu_info):
        distance = (z[:,None]-self.fitting_vectors[None]).square().mean(-1).min(-1).values.sqrt()
        if st.shape[1] == 0:
            return torch.cat([distance.log1p()[:,None],z.new_zeros((len(z),len(FEATURE_NAMES)-1))],1)
        t,y,mask = pool_support(st,sy,sm)
        n = mask.sum(1);denominator=n.clamp_min(1)
        begin=torch.where(mask,t,torch.inf).min(1).values
        end=torch.where(mask,t,-torch.inf).max(1).values
        span=torch.where(n>0,(end-begin).clamp_min(0),0.)
        first_dose=torch.where(events[...,1]>0,events[...,0],torch.inf).min(1).values
        elapsed=torch.where(n>0,(end-first_dose).clamp_min(0),0.)
        ps,pi=floor_curve(mu_seq,events,t),floor_curve(mu_info,events,t)
        residual_s=torch.where(mask,y-ps,0.);residual_i=torch.where(mask,y-pi,0.)
        disagreement=torch.where(mask,(ps-pi).abs(),0.).sum(1)/denominator
        return torch.stack([distance.log1p(),n.log1p(),span.log1p(),elapsed.log1p(),
            residual_s.sum(1)/denominator,(residual_s.square().sum(1)/denominator).sqrt(),
            residual_i.sum(1)/denominator,(residual_i.square().sum(1)/denominator).sqrt(),disagreement],-1)

    def condition(self,z,events,st,sy,sm,ids=None,gate_override=None):
        # Conditioning and gate features have no access to query times or labels.
        if st.ndim != 2 or st.shape != sy.shape or sm.shape != st.shape or sm.dtype != torch.bool:
            raise ValueError('Aligned support times, labels and boolean mask required')
        if not torch.isfinite(st[sm]).all() or not torch.isfinite(sy[sm]).all():
            raise ValueError('Observed support must be finite')
        # The frozen legacy information expert expects at least one time column.
        # A masked placeholder is equivalent to no observations, with no update.
        if st.shape[1] == 0:
            st = z.new_zeros((len(z),1));sy = torch.zeros_like(st)
            sm = torch.zeros_like(st,dtype=torch.bool)
        else:
            st = torch.where(sm,st,0.);sy = torch.where(sm,sy,0.)
        with torch.no_grad():
            seq=self.sequential.condition(z,events,st,sy,sm,ids)
            mu_info=self.information.prior(z,ids)[0]
            if self.information_config['kind']=='legacy':
                theta_info=self.information.condition(z,events,st,sy,sm,ids)['mean']
            else:
                theta_info=iterated_condition(mu_info,events,st,sy,sm,
                    prior_sd=self.information_config['prior_sd'])['parameters']
            cutoff=seq['time']
            first=torch.where(events[...,1]>0,events[...,0],torch.inf).min(1).values
            start=torch.minimum(first,cutoff)
            info_state=advance(theta_info,z.new_zeros((len(z),3)),start,cutoff[:,None],events,True)[:,0]
            # At k=0 the common state is PRE first event, not post-dose. Otherwise
            # both experts are post-event at the final support time.
            info_state=torch.where(seq['include_start'][:,None],0.,info_state)
            features=self._features(z,events,st,sy,sm,seq['prior_parameters'],mu_info)
        gate=torch.sigmoid(self.gate(features)).squeeze(-1)
        if gate_override is not None:
            gate=torch.as_tensor(gate_override,dtype=z.dtype,device=z.device).expand(len(z))
        parameters,amounts=mix_dynamics(seq['parameters'],seq['state'],theta_info,info_state,gate)
        return dict(fusion=self.fusion,gate=gate,gate_features=features,parameters=parameters,state=amounts,
            time=seq['time'],include_start=seq['include_start'],
            sequential_parameters=seq['parameters'],sequential_state=seq['state'],
            information_parameters=theta_info,information_state=info_state)

    def forward(self,z,events,st,sy,sm,qt,ids=None,variance=False,gate_override=None):
        if variance:raise ValueError('Only point forecasting is implemented')
        cutoff=torch.where(sm,st,-torch.inf).max(1).values if st.shape[1] else z.new_full((len(z),),-torch.inf)
        if qt.ndim!=2 or qt.shape[0]!=len(z) or qt.shape[1]==0 or not torch.isfinite(qt).all() or (qt<=cutoff[:,None]).any():
            raise ValueError('Queries must be finite and strictly after observed history')
        state=self.condition(z,events,st,sy,sm,ids,gate_override)
        if self.fusion=='dynamics':
            amounts=advance(state['parameters'],state['state'],state['time'],qt,events,state['include_start'])
            mean=floor_concentration(state['parameters'],amounts)
        else:
            with torch.no_grad():
                amounts=advance(state['sequential_parameters'],state['sequential_state'],state['time'],qt,events,state['include_start'])
                ys=floor_concentration(state['sequential_parameters'],amounts)
                yi=floor_curve(state['information_parameters'],events,qt)
            g=state['gate'][:,None]
            mean=(1-g)*ys+g*yi
        return dict(mean=mean,state=state)
