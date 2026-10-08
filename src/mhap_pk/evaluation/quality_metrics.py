"""Descriptive prediction metrics; natural-log concentrations, no fitted thresholds."""
import numpy as np
from mhap_pk.metrics import metric


def extended_metrics(frame):
    x = frame.copy()
    y = x.y_true_log.to_numpy(dtype=float)
    p = x.y_pred_log.to_numpy(dtype=float)
    if not len(y) or not np.isfinite(np.r_[y, p]).all():
        raise ValueError('Nonempty finite log predictions are required')
    e = p-y
    a = np.abs(e)
    x['sq'] = e**2
    _, sources = metric(x)
    x['ae'] = a
    x['two'] = a <= np.log(2)
    parts = x.groupby(['source_id','drug_id','participant_id']).agg(ae=('ae','mean'), two=('two','mean'))
    balanced = parts.groupby(level=[0,1]).mean().groupby(level=0).mean().mean()
    var = np.mean((y-y.mean())**2)
    den = np.var(y)+np.var(p)+(y.mean()-p.mean())**2
    out = dict(sb_hrmse=float(sources.sb_hrmse.mean()), log_rmse=float(np.sqrt(np.mean(e**2))),
               log_mae=float(a.mean()), signed_bias=float(e.mean()),
               median_abs_log_error=float(np.median(a)), p90_abs_log_error=float(np.quantile(a,.9)),
               p95_abs_log_error=float(np.quantile(a,.95)),
               geometric_mean_fold_error=float(np.exp(a.mean())),
               geometric_bias_ratio=float(np.exp(e.mean())),
               sb_log_mae=float(balanced.ae), sb_within_twofold_pct=float(100*balanced.two),
               pooled_log_r2=float(1-np.mean(e**2)/var) if var>0 else np.nan,
               pooled_log_ccc=float(2*np.mean((y-y.mean())*(p-p.mean()))/den) if den>0 else np.nan,
               under_twofold_pct=float(100*np.mean(e < -np.log(2))),
               over_twofold_pct=float(100*np.mean(e > np.log(2))))
    for factor, name in [(1.25,'1p25'),(1.5,'1p5'),(2,'two'),(3,'three'),(5,'five')]:
        out['within_'+name+'fold_pct'] = float(100*np.mean(a<=np.log(factor)))
    c, pred = np.exp(y), np.exp(p)
    if not np.isfinite(np.r_[c,pred]).all():
        raise ValueError('Nonfinite linear concentrations')
    out['linear_rmse_umol_l'] = float(np.sqrt(np.mean((pred-c)**2)))
    out['linear_mae_umol_l'] = float(np.mean(np.abs(pred-c)))
    return out
