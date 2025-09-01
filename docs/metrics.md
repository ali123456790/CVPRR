# Metrics
- **SR** (Success Rate): fraction of episodes reaching goal.
- **SPL**: success weighted by inverse path length (0..1).
- **NE** (Navigation Error): meters from goal when stopped.
- **GA** (Grounding Accuracy): percent of steps where chosen region IoU ≥ τ with referent.

Report per-dataset and per-category (attributes vs relations). Keep seeds fixed and
publish the split file & scripts for RxR-FG.
