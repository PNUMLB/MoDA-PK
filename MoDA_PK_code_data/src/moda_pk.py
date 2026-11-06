"""Convenience API for the original model; zeroed conditioning preserves 1024D."""
from mhap_pk.models.adapters.final_gated_bigru import build_final

def build_model(n_drugs, population, *, use_molecular=True, use_drug_id=True,
                use_cl_vc_update=True, use_central_amount_update=True):
    if not use_molecular and not use_drug_id:
        raise ValueError('Both-conditioning-off is outside the revised ablation matrix')
    variant='without_molecule' if not use_molecular else ('without_identity' if not use_drug_id else 'full')
    model=build_final(variant,n_drugs,population)
    model.parameter_update=use_cl_vc_update
    model.state_update=use_central_amount_update
    return model
