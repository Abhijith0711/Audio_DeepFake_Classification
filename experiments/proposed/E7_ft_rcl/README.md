# E7 FT-RCL

One config, four modes, full `reworked_subset` (no data fraction). Checkpoint selection is development macro-F1. Seeds match the reworked notebooks: seed1=42, seed2=1337, seed3=2024.

Train every mode and seed, then write the mean and variance table:

```bash
python scripts/run_e7_reworked_seeds.py
```

One mode and one seed:

```bash
python -m src.training.train \
  --config configs/proposed/e7_ft_rcl_atadd_v1.yaml \
  --mode M3 \
  --seed 42 \
  --experiment-id E7_M3_reworked_seed1 \
  --eval-splits dev,eval
```

Smoke (one epoch, 32 clips):

```bash
python -m src.training.train \
  --config configs/proposed/e7_ft_rcl_atadd_v1.yaml \
  --mode M3 \
  --experiment-id E7_M3_smoke \
  --epochs 1 \
  --max-samples 32
```
