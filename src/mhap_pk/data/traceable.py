"""Source-traceable cohort view with immutable pre-registered partitions."""
import json
import numpy as np
from .core import ROOT,load_profiles as legacy_load
DATASET_ID='source_traceable_20260925'
SPLIT_ID='source_traceable_20260925'
BASE=ROOT/'data/splits'/SPLIT_ID

def load_profiles():
 ps,meta,excluded=legacy_load()
 reference=json.loads((ROOT/'data/metadata/provenance_groups_20260925/source_traceable/dataset_reference.json').read_text())
 drugs=set(reference['drug_ids']);keep=[i for i,p in enumerate(ps) if p['drug_id'] in drugs]
 return [ps[i] for i in keep],meta.iloc[keep].reset_index(drop=True),excluded[excluded.drug_id.isin(drugs)].reset_index(drop=True)

def indices_for_split(meta,fold):
 spec=json.loads((BASE/'partitions.json').read_text());keys=meta[['source_id','drug_id','participant_id','profile_id']].astype(str).values.tolist();assert keys==spec['profile_keys']
 return tuple(np.array(spec['folds'][fold][role],dtype=int) for role in ['train','validation','test'])
