# E8 ablations

STAR-Edge ablations are future work and are not runnable. The current comparison is the four training modes in `configs/proposed/e7_ft_rcl_atadd_v1.yaml`:

- M0: clean cross-entropy
- M1: degraded-view cross-entropy
- M2: prediction coherence only
- M3: full FT-RCL

Run them with `scripts/run_e7_reworked_seeds.py`.
