"""Canonical hierarchical metrics used by every result exporter."""

import numpy as np
import pandas as pd


def profile_uid(frame):
    """Return a collision-safe profile key without changing split identities."""
    cols = ['source_id', 'drug_id', 'participant_id', 'profile_id']
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise KeyError(f'missing profile identity columns: {missing}')
    return frame[cols].astype(str).agg('::'.join, axis=1)


def metric(p):
    required = {'condition','seed','source_id','drug_id','participant_id','sq'}
    if not required.issubset(p.columns):
        raise KeyError(f'missing metric columns: {sorted(required-set(p.columns))}')
    # Participant is the individual aggregation unit. profile_id is deliberately
    # not used here because one participant may have multiple drug profiles.
    mse = p.groupby(['condition', 'seed', 'source_id', 'drug_id', 'participant_id'], sort=False).sq.mean()
    drug = mse.groupby(level=[0, 1, 2, 3]).mean().pow(.5).rename('sb_hrmse').reset_index()
    source = drug.groupby(['condition', 'seed', 'source_id']).sb_hrmse.mean().reset_index()
    return drug, source


def secondary_metrics(p):
    """Secondary metrics with the same participant/drug identity contract."""
    required = {'condition','seed','source_id','drug_id','participant_id','y_true_log','y_pred_log'}
    if not required.issubset(p.columns):
        raise KeyError(f'missing secondary metric columns: {sorted(required-set(p.columns))}')
    x = p.copy(); x['error'] = x.y_pred_log - x.y_true_log; x['abs_error'] = x.error.abs()
    key = ['condition','seed','source_id','drug_id','participant_id']
    prof = x.groupby(key, sort=False).agg(
        signed_bias=('error','mean'), median_abs_log_error=('abs_error','median'),
        p90_abs_log_error=('abs_error', lambda v: v.quantile(.9)),
        within_twofold=('abs_error', lambda v: np.mean(v <= np.log(2))),
        query_rows=('error','size')).reset_index()
    return prof.groupby(['condition'], sort=False).agg(
        signed_bias=('signed_bias','mean'), median_abs_log_error=('median_abs_log_error','mean'),
        p90_abs_log_error=('p90_abs_log_error','mean'),
        profile_within_twofold_pct=('within_twofold', lambda v: 100*v.mean()),
        query_rows=('query_rows','sum')).reset_index()
