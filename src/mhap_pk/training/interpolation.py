"""Same-profile temporal replacement from revealed support only."""
import math
import torch


def event_free_interval(left, right, events):
    """Conservatively exclude starts/stops even at the interval endpoints."""
    active = events[events[:, 1] > 0]
    if not len(active):
        return True
    start = active[:, 0]
    if ((start >= left) & (start <= right)).any():
        return False
    infusion = active[active[:, 2] == 2]
    stop = infusion[:, 0] + infusion[:, 3]
    return not bool(((stop >= left) & (stop <= right)).any())


def interpolated_log_value(left, right, fraction, kind):
    if kind == 'log_linear':
        return (1-fraction)*left + fraction*right
    if kind == 'linear':
        # Equivalent to log((1-lambda)*exp(left)+lambda*exp(right)), stable.
        return torch.logaddexp(left+math.log1p(-fraction), right+math.log(fraction))
    raise ValueError(f'Unknown interpolation {kind}')


def replace_support(times, labels, mask, events, generator, kind,
                    probability=.5, fraction_bounds=(.25,.75)):
    if kind not in ['linear','log_linear'] or not 0 <= probability <= 1:
        raise ValueError('Invalid kind/probability')
    lo,hi=fraction_bounds
    if not 0 < lo <= hi < 1:
        raise ValueError('Interpolation must be strictly inside endpoints')
    out_times=times.clone();out_labels=labels.clone();records=[]
    for row in range(len(times)):
        blocks=torch.unique(times[row,mask[row]],sorted=True)
        eligible=[]
        for left,right in zip(blocks[:-1],blocks[1:]):
            ii=torch.where(mask[row] & (times[row]==left))[0]
            jj=torch.where(mask[row] & (times[row]==right))[0]
            # Do not turn replicate blocks into artificial independent measurements.
            if len(ii)!=1 or len(jj)!=1:
                continue
            if event_free_interval(left,right,events[row]):
                eligible.append((int(ii[0]),int(jj[0])))
        if not eligible or torch.rand((),generator=generator)>=probability:
            continue
        pair=int(torch.randint(len(eligible),(),generator=generator))
        i,j=eligible[pair]
        lam=lo+(hi-lo)*float(torch.rand((),generator=generator))
        t=(1-lam)*times[row,i]+lam*times[row,j]
        y=interpolated_log_value(labels[row,i],labels[row,j],lam,kind)
        out_times[row,i]=t;out_labels[row,i]=y
        records.append(dict(row=row,left_slot=i,right_slot=j,left_time=float(times[row,i]),
            right_time=float(times[row,j]),fraction=lam,synthetic_time=float(t),
            left_log=float(labels[row,i]),right_log=float(labels[row,j]),synthetic_log=float(y)))
    return out_times,out_labels,records
