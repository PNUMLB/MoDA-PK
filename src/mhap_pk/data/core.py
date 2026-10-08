"""Auditable participant/source splits and label-separated forecasting episodes."""
from dataclasses import dataclass
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import torch
from .episodes import split_episode

ROOT = Path(__file__).resolve().parents[3]
KS = [0, 1, 2, 4]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_profiles():
    raw = pd.read_csv(ROOT / 'data/processed/full16_v1/primary_full16.csv', low_memory=False)
    targets = raw[raw.target_log_dv.notna()].copy()
    assert targets.observation_id.is_unique
    ledger = pd.read_csv(ROOT / 'data/processed/full16_v1/dose_events.csv')
    original = json.loads((ROOT / 'data/provenance/forecast_participant_split_origin_LOCAL_ONLY.json').read_text())
    membership = {}
    for source, parts in original['parts'].items():
        for part, keys in parts.items():
            for drug, profile in keys:
                key = (source, drug, profile)
                assert key not in membership
                membership[key] = part
    profiles, excluded = [], []
    for (drug, profile), group in targets.groupby(['dataset', 'global_subject_id'], sort=True):
        group = group.sort_values(['time', 'observation_id'])
        source, participant = group.source_path.iloc[0], group.participant_id.iloc[0]
        assert group.participant_id.nunique() == 1
        key = (source, drug, profile)
        meta = dict(source_id=source, drug_id=drug, participant_id=participant, profile_id=profile,
                    known_part=membership[key])
        times = np.sort(group.time.unique())
        if len(times) < 5:
            excluded.append({**meta, 'reason': 'fewer_than_five_distinct_times', 'time_blocks': len(times)})
            continue
        episodes = [split_episode(group, k, common_future_after=float(times[3])) for k in KS]
        query = episodes[0].query
        assert all(e.query.observation_id.tolist() == query.observation_id.tolist() for e in episodes)
        events = ledger[(ledger.profile_id == profile) & (ledger.drug == group.drug.iloc[0])]
        assert len(events)
        ev = np.array([[r.time_h, r.amount_umol, {'oral': 0, 'iv_bolus': 1, 'iv_infusion': 2}[r.route], r.duration_h]
                       for r in events.itertuples()], dtype=np.float64)
        profiles.append(dict(**meta, episodes=episodes, query=query, events=ev,
                             route=';'.join(sorted(events.route.unique()))))
    meta = pd.DataFrame([{k: p[k] for k in ['source_id', 'drug_id', 'participant_id', 'profile_id', 'known_part', 'route']}
                         for p in profiles])
    # A participant can span drugs/profiles, but never parts within a source.
    assert meta.groupby(['source_id', 'participant_id']).known_part.nunique().max() == 1
    assert meta.groupby('drug_id').source_id.nunique().max() == 1
    return profiles, meta, pd.DataFrame(excluded)


def indices_for_split(meta, fold):
    sources = sorted(meta.source_id.unique())
    if fold == 'known':
        train = np.flatnonzero(meta.known_part == 'train')
        val = np.flatnonzero(meta.known_part == 'validation')
        test = np.flatnonzero(meta.known_part == 'test')
    else:
        index = int(fold.split('_')[1])
        test_source, val_source = sources[index], sources[(index + 1) % len(sources)]
        train = np.flatnonzero(~meta.source_id.isin([test_source, val_source]))
        val = np.flatnonzero(meta.source_id == val_source)
        test = np.flatnonzero(meta.source_id == test_source)
    sets = [set(meta.iloc[x].participant_id) for x in [train, val, test]]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    if fold != 'known':
        drugs = [set(meta.iloc[x].drug_id) for x in [train, val, test]]
        assert not (drugs[0] & drugs[1] or drugs[0] & drugs[2] or drugs[1] & drugs[2])
    return train, val, test


def representations(meta, fitting_indices, representation, mapping_seed):
    drugs = sorted(meta.drug_id.unique())
    training = sorted(meta.iloc[fitting_indices].drug_id.unique())
    if representation == 'unimol':
        with np.load(ROOT / 'data/processed/full16_v1/embeddings/unimol.npz') as cache:
            vectors = dict(zip(cache['drugs'].astype(str), cache['embeddings'].astype(np.float64)))
        matrix = np.stack([vectors[d] for d in drugs])
    elif representation == 'random':
        # One fixed independent vector per identity, never per row/person.
        matrix = np.random.default_rng(mapping_seed).normal(size=(len(drugs), 512))
    else:
        matrix = np.zeros((len(drugs), 512))
    if representation in ['unimol', 'random']:
        values = matrix[[drugs.index(d) for d in training]]
        mean, sd = values.mean(0), values.std(0).clip(.1)
        matrix = (matrix - mean) / sd
    else:
        mean, sd = np.zeros(512), np.ones(512)
    z = torch.tensor(matrix[[drugs.index(d) for d in meta.drug_id]], dtype=torch.float64)
    ids = torch.tensor([training.index(d) if d in training else -1 for d in meta.drug_id])
    scaler = dict(fitted_drugs=training, representation=representation, mapping_seed=mapping_seed,
                  mean=mean.tolist(), sd=sd.tolist(), minimum_feature_sd=.1)
    return z, ids, scaler


@dataclass
class Batch:
    indices: np.ndarray
    z: torch.Tensor
    ids: torch.Tensor
    events: torch.Tensor
    support_t: torch.Tensor
    support_y: torch.Tensor
    support_mask: torch.Tensor
    query_t: torch.Tensor
    query_y: torch.Tensor
    query_mask: torch.Tensor
    weights: torch.Tensor

    def inputs(self, ks):
        rows = torch.arange(len(self.indices))
        return (self.z, self.events, self.support_t[rows, ks], self.support_y[rows, ks],
                self.support_mask[rows, ks], self.query_t, self.ids)


def make_batch(profiles, meta, indices, z, ids):
    selected = [profiles[i] for i in indices]
    b, e = len(selected), max(len(p['events']) for p in selected)
    q = max(len(p['query']) for p in selected)
    s = max(len(p['episodes'][-1].support) for p in selected)
    ev = np.zeros((b, e, 4)); st = np.zeros((b, 4, s)); sy = st.copy(); sm = st.astype(bool)
    qt = np.zeros((b, q)); qy = qt.copy(); qm = qt.astype(bool)
    weights = qt.copy()
    m = meta.iloc[indices]
    sources = m.source_id.nunique()
    drug_counts = m.groupby('source_id').drug_id.nunique()
    people_counts = m.groupby(['source_id', 'drug_id']).participant_id.nunique()
    row_counts = {}
    for p in selected:
        key = (p['source_id'], p['drug_id'], p['participant_id'])
        row_counts[key] = row_counts.get(key, 0) + len(p['query'])
    for i, p in enumerate(selected):
        ev[i, :len(p['events'])] = p['events']
        for j, ep in enumerate(p['episodes']):
            n = len(ep.support)
            st[i, j, :n] = ep.support.time
            sy[i, j, :n] = ep.support.target_log_dv
            sm[i, j, :n] = True
        n = len(p['query'])
        qt[i, :n] = p['query'].time; qy[i, :n] = p['query'].target_log_dv; qm[i, :n] = True
        qt[i, n:] = float(p['query'].time.iloc[-1])
        key = (p['source_id'], p['drug_id'], p['participant_id'])
        weights[i, :n] = 1 / (sources * drug_counts[key[0]] * people_counts[key[:2]] * row_counts[key])
    assert abs(weights.sum() - 1) < 1e-10
    tensor = lambda x: torch.tensor(x, dtype=torch.bool if x.dtype == bool else torch.float64)
    return Batch(np.asarray(indices), z[indices], ids[indices], *map(tensor, [ev, st, sy, sm, qt, qy, qm, weights]))


def score(mean, batch, meta):
    sums = {}
    for row, index in enumerate(batch.indices):
        m = meta.iloc[index]
        sq = (mean[row][batch.query_mask[row]] - batch.query_y[row][batch.query_mask[row]]).square()
        key = (m.source_id, m.drug_id, m.participant_id)
        old = sums.get(key, (0., 0))
        sums[key] = (old[0] + float(sq.sum()), old[1] + len(sq))
    frame = pd.DataFrame([dict(source=s, drug=d, participant=p, mse=v / n) for (s, d, p), (v, n) in sums.items()])
    return float(frame.groupby(['source', 'drug']).mse.mean().pow(.5).groupby('source').mean().mean())
