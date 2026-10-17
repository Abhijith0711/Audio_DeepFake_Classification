# Master Implementation Document  
## Audio Deepfake Classification — Modular Training, Robustness, and Evaluation

**Document purpose:** Single reference for supervisors and reviewers describing what the repository implements today: architecture, algorithms, file layout, commands, artifacts, and quality gates.  
**Scope:** AT-ADD Track 2 pipeline at 24 kHz (E1 v4 baseline) plus robustness-aware training (FT-RCL and optional STAR-Edge). ASVspoof notebooks and baselines remain in the repository but are outside this modular training path unless extended similarly.  
**Status:** Implementation complete for Phases 0–2 and evaluation skeleton (Phase 4); edge compression (separate design folder under `docs/edge/`) is specified but not coded in `src/deployment/`.

---

## 1. Executive summary

The project detects **real vs. synthetic (fake) audio** using a compact **MobileNetV3-Small** classifier on **log-mel spectrograms**. Earlier work lived mainly in Jupyter notebooks. The current implementation **extracts that logic into importable Python modules** under `src/` so that training, degradation, and evaluation are **reproducible from the command line** with frozen configs and saved artifacts.

Two layers of capability exist:

| Layer | Purpose |
|--------|---------|
| **Baseline (E1 v4)** | Clean-audio training and eval; must match published E1 v4 metrics within a fixed tolerance on the frozen eval split. |
| **Robustness-aware training (E7)** | Same backbone and preprocessing, but training uses **paired clean/degraded views**, **consistency losses** on predictions and frequency-band features, **curriculum over degradation severity**, and optional **ΔM-aware checkpoint selection** on the development split. |

Design principle: **degradation is applied in the waveform domain**; mel extraction and global normalization stay **identical** to the baseline so comparisons are fair.

---

## 2. Scientific and engineering objectives

1. **Reproducibility:** One config file + one command reproduces a run; all hyperparameters and metrics are written to disk.  
2. **No eval leakage:** Development set for early stopping and model selection only; **evaluation split is frozen** for reported numbers.  
3. **Baseline fidelity:** Modular code must not change E1 v4 architecture or preprocessing math relative to the reference notebook.  
4. **Channel robustness:** Measure performance drop under codec, resampling, noise, and telephony-like band-limiting; optimize **ΔM** (clean metric minus conditioned metric) alongside clean accuracy.  
5. **Edge compatibility:** Keep parameter count near **~1.52M**; robustness is a **training-time** intervention compatible with later compression/quantization work.

---

## 3. High-level system architecture

```text
                    ┌─────────────────────────────────────────┐
                    │           YAML experiment config           │
                    │  (paths, audio, mel, training, FT-RCL)   │
                    └────────────────────┬────────────────────┘
                                         │
         ┌───────────────────────────────┼───────────────────────────────┐
         ▼                               ▼                               ▼
  ┌──────────────┐              ┌──────────────┐              ┌──────────────┐
  │  Manifests   │              │ Preprocess   │              │  Robustness  │
  │  CSV splits  │──waveform──► │ 24 kHz, crop │◄──optional──│  degrade()   │
  └──────────────┘              │ GPULogMel    │   paired     └──────────────┘
                                └──────┬───────┘
                                       │ log-mel (1×80×T)
                                       ▼
                                ┌──────────────┐
                                │ MobileNetV3  │
                                │ + (optional) │
                                │ SubbandPool  │
                                └──────┬───────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
             Training loop                          Evaluation harness
         (CE, FT-RCL, STAR-Edge)              (F1, AUROC, EER, ΔM, plots)
                    │
                    ▼
         experiments/{id}/ + results/{figures,logs,tables}/
```

**Data flow (training, robustness mode):**

1. Load clip from manifest → fixed-length waveform (random crop if train, center if eval).  
2. Optionally build degraded waveform with `degrade(...)`.  
3. GPU (or CPU) STFT + mel + log + **global** mean/std normalization.  
4. Forward pass(es) through shared-weight network; combined loss; backward.  
5. Each epoch: dev metrics (+ optional robustness proxy); save checkpoints and logs.

---

## 4. Repository layout (implementation-relevant)

```text
Audio_DeepFake_Classification/
├── requirements.txt                 # Pinned major versions (torch, librosa, …)
├── configs/
│   ├── baselines/e1_atadd_v4.yaml
│   └── proposed/
│       ├── e7_ft_rcl_atadd_v1.yaml
│       └── e7_star_edge_atadd_v1.yaml
├── data/
│   └── manifests/                   # train/dev/eval CSVs (AT-ADD)
├── src/
│   ├── preprocessing/preprocess_atadd_v4.py   # CANONICAL mel frontend (do not fork)
│   ├── models/                      # MobileNet, SubbandPool, FT-RCL wrapper, STAR-Edge helpers
│   ├── datasets/                    # Manifest loaders, paired robustness dataset
│   ├── robustness/                  # Conditions, degrade API, matrix builder
│   ├── training/                    # train CLI, trainer, losses, curriculum
│   └── evaluation/                  # metrics, evaluate CLI, ΔM, plotting
├── experiments/                     # Per-run artifacts (config, logs, checkpoints, eval JSON/CSV)
├── results/
│   ├── figures/                     # PNG curves and heatmaps
│   ├── tables/                      # e.g. ΔM CSV
│   └── logs/                        # stdout capture, run_meta.json
├── notebooks/                       # Historical baselines (E1 v4 source of truth for extraction)
└── docs/
    ├── MASTER_IMPLEMENTATION.md     # This document
    ├── proposed/                    # Algorithm specs (FT-RCL, experiments)
    └── edge/                        # Future edge/compression specs (not implemented in src/)
```

Legacy baseline metrics and configs remain under `experiments/baselines/E1_v4/` for regression comparison.

---

## 5. Datasets and manifests

### 5.1 AT-ADD Track 2 (primary for this implementation)

| Split | Manifest path | Typical size (reference E1 v4) |
|--------|----------------|--------------------------------|
| Train | `data/manifests/train_manifest.csv` | 46,000 |
| Dev | `data/manifests/dev_manifest.csv` | 36,536 |
| Eval | `data/manifests/eval_manifest.csv` | 44,289 |

**Required CSV columns:** `name`, `label`, `split`. Labels are `real` or `fake` (mapped to 0/1). Optional columns used in analysis: `type` (speech, singing, music, sound), `generator`, `file_id`.

**Audio path convention:**

```text
data/raw/AT_ADD/{split}/{name}
```

If manifests exist but raw audio is missing, loaders **fail with an explicit message** pointing to `src/datasets/create_btp_dataset.py` or `src/datasets/create_atadd_share.py`.

### 5.2 ASVspoof

Manifests under `data/manifests/ASVSpoof/` and preparation via `src/preprocessing/prepare_asvspoof_v1.py` are supported at dataset level; the new **modular train/eval entrypoints** in this document focus on AT-ADD E1 v4 / E7 unless extended with parallel configs.

### 5.3 Split discipline

| Split | Allowed use |
|--------|-------------|
| Train | Parameter updates |
| Dev | Early stopping, checkpoint selection, hyperparameter-free monitoring |
| Eval | **Final reported metrics only** — no tuning |

---

## 6. Canonical preprocessing (E1 v4)

**Single module:** `src/preprocessing/preprocess_atadd_v4.py`  
**Version tag:** `v4.0_24khz_repeat_pad`

### 6.1 Audio loading and segmentation

| Parameter | Value |
|-----------|--------|
| Sample rate | 24,000 Hz |
| Segment duration | 4.0 s → 96,000 samples |
| Long clips | Random crop (train) or center crop (eval) |
| Short clips | **Repeat-pad** (tile waveform) — not zero-pad |

Resampling uses librosa when file native rate ≠ 24 kHz. NaNs/Infs are zeroed after load.

### 6.2 Log-mel spectrogram (GPU-batched in training)

| Parameter | Value |
|-----------|--------|
| n_fft | 1024 |
| win_length | 600 |
| hop_length | 240 |
| n_mels | 80 |
| Window | Hann |
| Power | \|STFT\|² |
| Log | log(power + 1e-10) |
| Mel norm | Slaney, HTK=False |

Implementation class: `GPULogMel` — mel filterbank and window registered as buffers; STFT via `torch.stft` on batches of waveforms.

### 6.3 Global normalization

Mean and standard deviation are computed over **all mel bins and time frames** on the **training set only**, then applied at train and eval time:

```text
normalized = (log_mel - mean) / std
```

Reference statistics (frozen for regression):  
`experiments/baselines/E1_v4/E1_ATADD_logmel_stats_v4.json`  
Example: mean ≈ -9.191, std ≈ 6.851 (full train, seed 42).

**Rule for robustness experiments:** Never recompute or per-condition mel stats; degradations happen **before** mel extraction.

---

## 7. Baseline model (E1 v4)

### 7.1 Architecture

- **Backbone:** torchvision `mobenet_v3_small`, ImageNet weights **disabled** (train from scratch).  
- **Input:** 1-channel log-mel treated as a single-channel image (height ≈ mel bands, width ≈ time).  
- **Modification:** Replace first conv 3→1 channel (same kernel/stride/padding as RGB stem).  
- **Head:** Binary classifier, 2 logits (real vs fake).  
- **Parameters:** **1,519,618** trainable (matches E1 v4 config JSON).

**Module:** `src/models/mobilenet_v3_audio.py` → class `MobileNetV3SmallAudio`.

### 7.2 Training recipe (baseline)

| Setting | Value |
|---------|--------|
| Loss | Class-weighted cross-entropy |
| Optimizer | AdamW, lr=1e-3, weight_decay=1e-4 |
| Batch size | 32 |
| Epochs | 20 (early stop patience 5 on dev macro-F1) |
| LR schedule | ReduceLROnPlateau on dev macro-F1, factor 0.5, patience 2 |
| Seed | 42 (Python, NumPy, PyTorch, CUDA, DataLoader generator) |
| DataLoader | num_workers=4, prefetch_factor=2, pin_memory, persistent_workers |
| AMP | FP16 autocast + GradScaler on CUDA |

Class weights: inverse frequency style  
`weight_c = N / (2 * count_c)` for c ∈ {real, fake}.

### 7.3 Reference eval performance (regression gate)

File: `experiments/baselines/E1_v4/E1_ATADD_eval_metrics_v4.json`

| Metric | Reference value |
|--------|-----------------|
| Macro-F1 | 0.677588 |
| Accuracy | 0.757005 |
| AUROC | 0.875002 |
| EER | 0.199250 |
| EER threshold | 0.132739 |

**Acceptance:** Reproduction run in **clean-only mode (M0)** must be within **±0.005 absolute Macro-F1** on the **eval** split when using the same splits, stats, and recipe.

### 7.4 Extraction from notebook

Source notebook: `notebooks/ATADD_E1_MobileNetV3Small_Baseline_v4_24kHz_RepeatPad.ipynb`  
Extracted without intentional math changes: model, dataset path logic, mel pipeline, metrics (EER via ROC crossing), training loop behavior, checkpoint contents (model + optimizer + scheduler + mel stats in config blob).

---

## 8. Modular software components

### 8.1 Datasets

| Module | Role |
|--------|------|
| `src/datasets/atadd_manifest_dataset.py` | `ATADDManifestDataset`, manifest loading, path resolution, `make_dataloader` with notebook-identical loader options |
| `src/datasets/paired_robustness_dataset.py` | Returns `(clean_wave, degraded_wave, label, type, metadata)`; shared crop index; samples degradation from a severity-filtered pool |

Collate functions stack tensors and pass metadata as lists for gating and logging.

### 8.2 Models (robustness extensions)

| Module | Role |
|--------|------|
| `subband_pool.py` | Splits last spatial feature map into **K=8** horizontal bands; mean-pool each band → **8-D signature**, L2-normalized |
| `ft_rcl_detector.py` | Wraps MobileNet + SubbandPool; optional channel encoder for STAR-Edge |
| `channel_encoder.py` | Handcrafted waveform stats (RMS, crest, flatness, ZCR) → small MLP → channel code **u** |
| `artifact_projector.py` | Linear projection removing component of **z** along **u** (disentanglement helper) |

For **M0**, only `MobileNetV3SmallAudio` is instantiated. For **M1–M3**, `FTRCLDetector` is used.

### 8.3 Robustness library

**API** (`src/robustness/degrade.py`):

```text
degrade(waveform, sample_rate, condition, mode, source_id)
  → (waveform_degraded, metadata_dict)
```

**Metadata fields (required):** `source_id`, `condition_family`, `condition_name`, `severity`, `seed`, `pipeline_version`.

**Condition catalog** (`conditions.py`), severity-ordered:

| Family | Name / variant | Severity (typical) | Notes |
|--------|----------------|--------------------|--------|
| identity | clean | 0 | No-op |
| resample | down_up_16k | 1 | 24→16→24 kHz |
| mp3_codec | 128 / 64 / 32 kbps | 2–4 | Requires **ffmpeg** on PATH |
| additive_noise | SNR 20 / 10 / 5 dB | 2–4 | Gaussian noise scaled to SNR |
| telephony | bandlimit 300–3400 Hz | 3 | Butterworth bandpass |

If ffmpeg is missing, MP3 conditions log a warning and skip or fall back during matrix build; other degradations use NumPy/SciPy/librosa.

**Condition matrix builder:**

```bash
python -m src.robustness.build_condition_matrix --split eval --out data/manifests/conditioned
```

Writes degraded FLACs and a CSV linking clean path, degraded path, labels, and condition metadata (for offline conditioned eval).

### 8.4 Loss functions

**FT-RCL** (`src/training/losses/ft_rcl.py`):

- **Classification:** CE on clean + β × CE on degraded (β default 0.5).  
- **Prediction coherence (PC):** ‖p_clean − p_deg‖² on fake-class probability.  
- **Frequency-time signature coherence (FTSC):** 1 − cosine(s_clean, s_deg) on subband signatures.  
- **Gating (M3):** w_rob = α_type × confidence × η(severity); confidence = 2×|p_clean − 0.5|; α_type defaults: speech/singing 1.0, music 0.3, sound 0.5.

**STAR-Edge** (`star_edge.py`), optional when enabled in config:

- **STC:** Penalize jumps along a severity ladder in signature and probability space.  
- **ABR:** Asymmetric margin loss on **real** clips so degraded fake-score does not exceed clean by margin m (default 0.05).  
- **CAD:** Auxiliary CE predicting degradation family from channel code **u**.

Combined with FT-RCL terms; weights δ (STC), ρ (ABR) in YAML.

### 8.5 Severity curriculum

Scheduler: `src/training/schedulers/robustness_curriculum.py`

Default milestones (epochs 1–5 / 6–12 / 13–20):

| Epoch range | Max severity σ |
|-------------|----------------|
| 1–5 | 1 (mild) |
| 6–12 | 3 |
| 13+ | 4 (full ladder) |

Paired dataset `set_max_severity()` is updated each epoch so training sees only conditions up to the current cap.

### 8.6 Training modes

| Mode | Description |
|------|-------------|
| **M0 / clean** | Baseline: single clean view, CE only |
| **M1** | Random degraded view; CE on degraded only |
| **M2** | M1 + PC (γ_pc > 0, FTSC off) |
| **M3** | M2 + FTSC + type/confidence gating + severity curriculum |

CLI flag `--mode` accepts `clean`, `M0`, `M1`, `M2`, `M3`.

### 8.7 Checkpoint selection

**Baseline:** Best dev **macro-F1** (same as notebook).

**FT-RCL (configurable):** Score on dev:

```text
J = MacroF1_clean − λ × mean(ΔMacroF1)
```

λ default 0.5. ΔMacroF1 estimated on a **small number of dev batches** (default 5) with on-the-fly degradations for speed; increase `selection.delta_max_batches` for fuller estimates.

Checkpoints store: model state, optimizer, scheduler, mode, mel mean/std, config snapshot.

---

## 9. Evaluation and reporting

### 9.1 Metrics (`src/evaluation/metrics.py`)

Unified implementation (no notebook duplication):

| Metric | Definition |
|--------|------------|
| Macro-F1 | Unweighted mean of F1 on real and fake classes |
| Accuracy | Fraction correct at threshold 0.5 on scores |
| AUROC | ROC AUC using fake-class score |
| EER | Point where FPR ≈ FNR on ROC; threshold at EER reported |

Scores: softmax probability of **fake** class (index 1).

### 9.2 Evaluation CLI

```bash
python -m src.evaluation.evaluate \
  --config configs/baselines/e1_atadd_v4.yaml \
  --split eval \
  --checkpoint experiments/E1_v4_repro/best.pt \
  --experiment-id E1_v4_repro_eval
```

Produces predictions CSV, per-type and per-generator breakdowns (when columns exist), failure cases (top false positives/negatives), ROC and confusion matrix PNGs, and JSON metrics.

**Regression table:** For eval split, prints side-by-side comparison vs `E1_ATADD_eval_metrics_v4.json`.

### 9.3 ΔM analysis (`delta_m.py`)

For each condition and audio type:

```text
ΔM = M_clean − M_conditioned
```

Metrics M: macro-F1, AUROC, EER (per-type subsets and overall). Output CSV default: `results/tables/E7_delta_m_atadd.csv`; heatmap under `results/figures/`.

---

## 10. Training CLI and configs

### 10.1 Train entrypoint

```bash
python -m src.training.train \
  --config configs/baselines/e1_atadd_v4.yaml \
  --mode clean \
  --experiment-id E1_v4_repro
```

| Flag | Purpose |
|------|---------|
| `--epochs` | Override config epoch count |
| `--max-samples` | Truncate manifests for smoke tests |
| `--checkpoint` | Skip training; load weights and run eval only |
| `--eval-splits` | Comma-separated splits to evaluate after train (default `eval`) |

### 10.2 Config files

| File | Use |
|------|-----|
| `configs/baselines/e1_atadd_v4.yaml` | E1 reproduction; `selection.use_j_score: false` |
| `configs/proposed/e7_ft_rcl_atadd_v1.yaml` | FT-RCL; J-score enabled; STAR-Edge off |
| `configs/proposed/e7_star_edge_atadd_v1.yaml` | FT-RCL + STAR-Edge flags on |

Shared keys: dataset manifest paths, audio/spectrogram blocks, training hyperparameters, `logmel_stats_path`, `baseline_metrics_path`, `ft_rcl.*`, `star_edge.*`.

---

## 11. Output contract (every train/eval run)

Under **`experiments/{experiment_id}/`:**

| Artifact | Content |
|----------|---------|
| `config_resolved.yaml` | Full resolved config + git commit hash when available |
| `train_log.jsonl` | One JSON object per epoch (loss, acc, dev metrics, J, lr, mode) |
| `training_history.csv` | Same columns for plotting/pandas |
| `best.pt` / `last.pt` | Weights + optimizer + scheduler + mel stats |
| `eval_metrics_{split}.json` | macro_f1, accuracy, auroc, eer, eer_threshold |
| `eval_predictions_{split}.csv` | filepath, label, score, pred, type, generator |
| `eval_by_type_{split}.csv` | Per-type metrics |
| `eval_by_generator_{split}.csv` | Fake recall by generator when applicable |
| `failure_cases_{split}.csv` | Top false positives and negatives |

Under **`results/figures/{experiment_id}/`:**

- `training_loss_accuracy.png`, `dev_macro_f1_auroc.png`, `dev_eer.png`  
- `roc_{split}.png`, `confusion_matrix_{split}.png`  
- `delta_m_heatmap_type_x_condition.png` when ΔM is run  

Under **`results/logs/{experiment_id}/`:**

- `stdout.txt` (logging FileHandler)  
- `run_meta.json` (timestamp, host, torch version, device, seed)

Matplotlib uses **Agg** backend, **200 DPI**, explicit `plt.close()` after each figure.

---

## 12. Dependencies

`requirements.txt` pins major versions for:

torch, torchvision, librosa, soundfile, pandas, scikit-learn, matplotlib, seaborn, PyYAML, tqdm, numpy, scipy.

Recommended: Python virtual environment (`.venv`) and `pip install -r requirements.txt`.

Optional system dependency: **ffmpeg** for MP3 codec degradations.

---

## 13. Testing and quality assurance

| Test | Location | What it verifies |
|------|----------|------------------|
| Loss forward passes | `tests/test_losses.py` | FT-RCL and STAR-Edge losses on random tensors (batch=4) |
| Manifest smoke | same file | Skips if data absent |
| Manual regression | Eval CLI vs E1 JSON | Macro-F1 within ±0.005 on eval |
| Smoke train | `--epochs 1 --max-samples 32` | End-to-end wiring when raw audio present |

Unit tests run on **CPU** without datasets. Integration tests require `data/raw/AT_ADD/`.

---

## 14. Reproduction checklist (supervisor-facing)

1. Install dependencies from `requirements.txt`.  
2. Ensure manifests exist under `data/manifests/`.  
3. Populate raw audio under `data/raw/AT_ADD/{train,dev,eval}/`.  
4. Confirm cached mel stats at `experiments/baselines/E1_v4/E1_ATADD_logmel_stats_v4.json`.  
5. Run baseline training (M0) or evaluate an existing `best.pt`.  
6. Compare eval JSON to reference E1 v4 metrics.  
7. For robustness: install ffmpeg (optional), run condition matrix builder, train E7 with `--mode M3`, run `delta_m.py`.  

**Full E7 example:**

```bash
python -m src.training.train \
  --config configs/proposed/e7_ft_rcl_atadd_v1.yaml \
  --mode M3 \
  --experiment-id E7_M3_ft_rcl_s42

python -m src.evaluation.delta_m \
  --checkpoint experiments/E7_M3_ft_rcl_s42/best.pt \
  --config configs/proposed/e7_ft_rcl_atadd_v1.yaml \
  --splits eval \
  --out results/tables/E7_delta_m_atadd.csv
```

---

## 15. What was intentionally not changed

To keep scientific claims valid:

- E1 **architecture width/depth** and ~1.52M parameter count  
- Train/dev/eval **manifests and splits**  
- Mel **hyperparameters** and v4 repeat-pad frontend  
- Replacing MobileNet with SSL/wav2vec in this track  
- Tuning on the **eval** split  

Notebook baselines (E2 GRU/LSTM, ASV paths) remain for historical comparison but are not required to run the modular E1/E7 path.

---

## 16. Known limitations and follow-on work

| Topic | Current state |
|--------|----------------|
| Raw AT-ADD audio | Must be present locally; not stored in git |
| MP3 degradations | Need ffmpeg; otherwise ladder is partial |
| Dev Δ for J-score | Subsampled batches by default; configurable |
| STAR-Edge CAD | Channel auxiliary + losses wired; full logits-from-artifact-subspace is simplified |
| Conditioned eval at scale | Matrix builder pre-generates eval degradations; train uses online degrade |
| Edge deployment | Specified under `docs/edge/` (compression, quantization, latency tables); **not** implemented in `src/deployment/` yet |
| E8 ablation grid | Documented under `experiments/proposed/E8_ablations/`; run by toggling YAML flags |

---

## 17. Relation to design specifications

Detailed algorithm prose and ablation plans live in:

- `docs/proposed/00_PIPELINE_PREREQUISITES.md` — interfaces and gates  
- `docs/proposed/01_NOVEL_METHOD_FT-RCL_SPEC.md` — FT-RCL and STAR-Edge math  
- `docs/proposed/02_EXPERIMENT_AND_ABLATION_PLAN.md` — experiment IDs  

**This master document** describes the **actual code paths and artifacts** that implement those specs; where the code is simplified (e.g. approximate dev Δ, partial CAD), Section 16 applies.

---

## 18. Glossary

| Term | Meaning |
|------|---------|
| **AT-ADD** | Multi-domain audio deepfake detection benchmark (Track 2) |
| **E1 v4** | MobileNetV3-Small baseline at 24 kHz with repeat-pad |
| **E7** | Proposed robustness-aware training experiment family |
| **FT-RCL** | Frequency-Time Robustness Coherence Learning |
| **STAR-Edge** | Extension: severity trajectory, channel–artifact disentanglement, asymmetric bona-fide robustness |
| **ΔM** | Drop in metric M under channel degradation vs clean |
| **EER** | Equal error rate |
| **GPULogMel** | Torch module for batched STFT/mel/log on GPU |

---

*End of master implementation document.*
