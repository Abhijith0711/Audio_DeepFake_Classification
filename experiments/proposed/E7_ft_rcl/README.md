# E7 FT-RCL

Train (full):

```bash
python -m src.training.train \
  --config configs/proposed/e7_ft_rcl_atadd_v1.yaml \
  --mode M3 \
  --experiment-id E7_M3_ft_rcl_s42
```

Smoke:

```bash
python -m src.training.train \
  --config configs/proposed/e7_ft_rcl_atadd_v1.yaml \
  --mode M3 \
  --experiment-id E7_M3_smoke \
  --epochs 1 \
  --max-samples 32
```
