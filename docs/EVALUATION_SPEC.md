# Evaluation specification

The registered tracks are K (known drug/new participant), DS (joint unseen drug–source), and XS (known drug/unseen original study). Existing data currently support K and the source-holdout development track; XS and external_locked remain unavailable until independent studies are registered.

Legacy/primary episodes use first-k time blocks with a common query after t4. The additional common-origin protocol fixes the cutoff at t4 and uses `{t4}`, `{t2,t4}`, and `{t1,t2,t3,t4}`. Future concentrations are never used to select rank, regularization, or stopping.
