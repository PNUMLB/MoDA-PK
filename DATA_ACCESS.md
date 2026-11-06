# Data access and redistribution

The tracked `data/` directory contains drug-, seed-, or cohort-level aggregate results. Tables containing participant IDs, profile IDs, observation IDs, individual sampling times, observed concentrations, or individual predictions are excluded from Git.

`local_data/` is reserved for separately authorized local records. Its presence in a researcher's workspace does not establish public redistribution permission. Without those local records, the repository supports aggregate-result reproduction only.

Permission to redistribute every row in the complete mixed cohort has not been established. In particular, the historical audit did not identify an explicit row-level redistribution grant for some Monolix example data. The [historical access policy](docs/original_data_access.md) is preserved for context; it is not a complete legal clearance list for all eleven drugs.

Training requires authorized original inputs, the original split and documented participant links, unit harmonization, dose events, frozen molecular representations, training-only scaling, and population calibration. Supplementary Table S1 of the manuscript documents source publications and access links. This repository does not provide a universal automated downloader.
