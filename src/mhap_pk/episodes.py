"""Support/query splitting by distinct time blocks, before tensor construction."""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Episode:
    support: pd.DataFrame
    query: pd.DataFrame
    cutoff: float
    k: int


def split_episode(observations, k, *, common_future_after=None):
    """One held-out profile; k counts distinct times, including all replicates.

    Required columns: observation_id, time, target_log_dv. This function is used
    after participant-level dataset splitting. Query labels are for loss/scoring,
    never passed to the model. Set common_future_after to the fourth observed
    time when comparing k=0/1/2/4 on identical future observations.
    """
    if not isinstance(k, (int, np.integer)) or k < 0:
        raise ValueError('k must be a nonnegative integer')
    required = ['observation_id', 'time', 'target_log_dv']
    frame = observations[required].copy()
    if frame.empty or frame.observation_id.isna().any() or not frame.observation_id.is_unique:
        raise ValueError('Expected a nonempty profile with unique observation IDs')
    if not np.isfinite(frame[['time', 'target_log_dv']].to_numpy()).all():
        raise ValueError('Only finite, eligible observations can form episodes')
    for key in ['participant_id', 'global_subject_id', 'dataset']:
        if key in observations and observations[key].nunique(dropna=False) != 1:
            raise ValueError('Episode cannot mix profiles or participants')
    frame = frame.sort_values(['time', 'observation_id']).reset_index(drop=True)
    times = np.sort(frame.time.unique())
    if k >= len(times):
        raise ValueError('Support must leave at least one future time block')
    cutoff = float(times[k - 1]) if k else float('-inf')
    support = frame[frame.time <= cutoff].copy()
    threshold = cutoff
    if common_future_after is not None:
        if not np.isfinite(common_future_after):
            raise ValueError('Common forecast cutoff must be finite')
        threshold = max(threshold, float(common_future_after))
    query = frame[frame.time > threshold].copy()
    if query.empty:
        raise ValueError('No future observations after the requested cutoff')
    assert not set(support.observation_id) & set(query.observation_id)
    return Episode(support=support, query=query, cutoff=cutoff, k=int(k))


def split_common_origin(observations, support_time_indices, *, origin_index=3):
    """Episode with a fixed forecast origin and varying support information.

    ``support_time_indices`` names distinct time blocks supplied to the model;
    the query always starts strictly after ``origin_index``.  This separates
    the effect of more observations from the shorter forecast horizon in a
    first-k design.
    """
    required = ['observation_id', 'time', 'target_log_dv']
    frame = observations[required].copy().sort_values(['time', 'observation_id']).reset_index(drop=True)
    if frame.empty or frame.observation_id.isna().any() or not frame.observation_id.is_unique:
        raise ValueError('Expected nonempty observations with unique IDs')
    if not np.isfinite(frame[['time', 'target_log_dv']].to_numpy()).all():
        raise ValueError('Only finite observations can form episodes')
    times = np.sort(frame.time.unique())
    if origin_index < 0 or origin_index >= len(times):
        raise ValueError('Forecast origin must be an observed time block')
    indices = sorted(set(int(i) for i in support_time_indices))
    if not indices or indices[-1] > origin_index or indices[0] < 0:
        raise ValueError('Support indices must be nonempty and no later than origin')
    support = frame[frame.time.isin(times[indices])].copy()
    query = frame[frame.time > times[origin_index]].copy()
    if query.empty:
        raise ValueError('No future observations after fixed origin')
    return Episode(support=support, query=query, cutoff=float(times[indices[-1]]), k=len(indices))
