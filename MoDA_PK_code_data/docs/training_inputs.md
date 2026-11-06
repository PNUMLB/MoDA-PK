# Training tensor payload

`torch.load(..., weights_only=True)`로 읽을 수 있는 dict/list/Tensor만 사용합니다.

```text
{
  "n_drugs": 11,
  "population": float64 Tensor[5],  # training-only calibrated log parameters
  "training": batch,
  "validation": batch,
  "validation_metadata": {
      "source_id": list[str], "drug_id": list[str], "participant_id": list[str]
  }
}
```

각 batch는 다음 key를 가집니다. B는 프로파일 수, E는 event 수 최대값, S는 support record 최대값, T는 query record 최대값입니다.

| Key | dtype/shape |
|---|---|
| molecular | float64 [B,512], training-drug-only standardized frozen vectors |
| ids | int64 [B], shared training drug mapping |
| events | float64 [B,E,4]: time_h, amount_umol, route(0/1/2), duration_h |
| support_t | float64 [B,4,S] |
| support_y | float64 [B,4,S], natural log concentration |
| support_mask | bool [B,4,S] |
| query_t | float64 [B,T] |
| query_y | float64 [B,T] |
| query_mask | bool [B,T] |
| weights | float64 [B,T], source/drug/participant/query hierarchical weights |

Support slot 0/1/2/3 corresponds to **k=0/1/2/4**, not 0/1/2/3. Queries for both training and validation start strictly after the fourth observed block. Padded query times repeat the last valid future time so every query tensor entry passes the model's time check; masked/padded weights are zero.

Training weights sum to 1. Each valid query has weight `1/(n_sources * n_drugs_in_source * n_people_in_source_drug * n_query_rows_for_person_across_profiles)`. Metadata order matches batch rows. Validation excludes training participants; linked profiles move as a group.

The fitter validates only basic tensor properties. It cannot infer whether source-specific units, participant links, split membership, query masking, train-only scaling or preprocessing are correct; preparing and auditing those inputs remains required. The input builder for the complete historical mixed cohort is not bundled as an automatic downloader.

The fitting loop uses 120 full-batch AdamW updates and original regularization, selects using validation, and stores the selected checkpoint plus logs under the ignored `outputs/` path. No test data should appear in this payload.
