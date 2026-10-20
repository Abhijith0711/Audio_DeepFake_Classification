# E5 degradation sweep

Build conditioned eval matrix:

```bash
python -m src.robustness.build_condition_matrix --split eval --out data/manifests/conditioned
```

Requires `data/raw/AT_ADD` and optional `ffmpeg` for MP3 conditions.
