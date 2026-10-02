# Repository scope

Included: model/solver implementations, training/evaluation and experiment scripts, analysis/figure generation code, configs, tests, environment requirements, input hashes, source links, and model/evaluation documentation.

Excluded from Git: manuscript and PDF/TeX assets, numerical analysis results and figures, run status bundles in experiments/, clinical data and processed rows, checkpoints, embeddings, query predictions, logs, downloaded third-party sources, caches, and earlier packaging manifests.

Analysis code is retained; analysis outputs are excluded. Experiment configuration lives in configs/ and registration/runner scripts. THIRD_PARTY_NOTICES.md remains for attribution, while third_party/ is ignored.

Ignored files remain locally available and in the separate paper-material folder. Git ignores do not remove files already tracked: inspect `git ls-files` if the policy is changed after a commit. At the time of this change no files were tracked and no commit or push had been made.
