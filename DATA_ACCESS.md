# Data access

This code repository does not distribute clinical records, processed participant-level tables, participant split assignments, frozen embeddings, checkpoints, or individual predictions.

Public source links are listed in `docs/DATASET_SOURCES.csv`. Public availability and successful numeric replay are not evidence that every combined or transformed dataset may be redistributed. Redistribution permission has not been verified for this release; this status does not mean all sources prohibit redistribution.

Before adding a source, record the dataset-specific license/access terms, permitted redistribution and transformation, required attribution, and any participant-data restrictions. Check source datasets and pretrained features separately from code licenses. Do not infer data permission from a package's software license.

`environment/required_inputs.json` records expected relative paths, sizes and hashes. Restore authorized files locally at those paths and run `python scripts/check_inputs.py`. Missing data are explicit dependencies; the package does not fabricate replacement clinical data or silently regenerate the original split.

The current repository is code-only and not a self-contained data reproduction bundle. Existing authorized working copies remain in the original research workspace. The source links are a starting point for access, not a universal automatic download pipeline.
