"""Center PK perturbations on actual training observations, preserving source randomness."""
from dataclasses import replace
import json,hashlib
import torch
from ..relational import floor_curve

def generate(ps,meta,tr,b,pop,seed,out,source):
    info=json.loads((source/'synthetic_generator.json').read_text())
    assert info['fitting_indices']==tr.tolist()
    original=torch.load(source/'synthetic_training_LOCAL_ONLY.pt',weights_only=True)
    fit=original['teacher_parameters']
    with torch.no_grad():
        h0=floor_curve(fit,b.events,b.support_t[:,3]);q0=floor_curve(fit,b.events,b.query_t)
        h=b.support_y[:,3]+(original['support']-h0)
        q=b.query_y+(original['query']-q0)
    sy=b.support_y.clone()
    for ki in range(4):sy[:,ki]=h
    assert torch.isfinite(q[b.query_mask]).all()
    record=info|dict(centering='real training observation + original synthetic value - fitted teacher prediction',original_generator_sha256=hashlib.sha256((source/'synthetic_generator.json').read_bytes()).hexdigest(),original_trajectory_sha256=hashlib.sha256((source/'synthetic_training_LOCAL_ONLY.pt').read_bytes()).hexdigest(),source_condition='synthetic',same_parameter_perturbations_and_noise=True,no_validation_or_test_fitting=True)
    (out/'synthetic_generator.json').write_text(json.dumps(record,indent=2))
    torch.save(dict(support=h,query=q),out/'synthetic_training_LOCAL_ONLY.pt')
    return replace(b,support_y=sy,query_y=q)
