# Third-party notices

The included `mamba_observer.py` implementation follows the Mamba-1 recurrence and initialization in `state-spaces/mamba`.

- Upstream: https://github.com/state-spaces/mamba
- Authors: Tri Dao and Albert Gu, Copyright 2023.
- License: Apache License 2.0; a copy is retained at `docs/licenses/MAMBA_APACHE_2_0.txt`.
- Project adaptations: differentiable functional state updates, masking and profile reset, a 43-to-32 input projection, and float64 recurrence.
- The reported FULL model uses a **BiGRU**, not Mamba. This module is retained as an import dependency of the existing observer controls.

Clinical datasets and Uni-Mol pretrained representations have separate upstream terms. Original row-level data and the frozen Uni-Mol encoder/cache are excluded from this Git release. No repository-wide license or additional redistribution rights are granted by this notice.
