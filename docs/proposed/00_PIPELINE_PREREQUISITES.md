# Pipeline Prerequisites Before Robustness-Aware Training (Nikkath Track)

**Status:** Design document (no implementation)  
**Audience:** Full team (Abhijith, Bhuvan, Joshi, Nikkath) + guide review  
**Purpose:** Define what must exist *before* Phase D (proposed detector) so experiments are reproducible, leakage-safe, and defensible in viva.

---

## 1. Scope

This document lists **missing or incomplete** items from Phases A–B that block a credible robustness pipeline (Phases C–D). It does **not** implement them; it specifies acceptance criteria and handoff contracts.

Your execution plan (Section 3, 14, 15) assumes a **single integrated repo**. Current state: strong **notebook-centric baselines** + **dataset scripts**, but no shared `training/`, `robustness/`, or `evaluation/` modules.

---

## 2. Prerequisite checklist (gate for Phase D)

| ID | Owner | Deliverable | Acceptance criteria | Current state |
|----|--------|-------------|---------------------|---------------|
| P1 | Abhijith | **Frozen canonical baselines** | One official checkpoint + config per dataset: `ATADD_E1_v4`, optional `ASV_E1_v2`; eval Macro-F1/AUROC/EER match saved JSON within tolerance | Partial: metrics exist; checkpoints gitignored |
| P2 | Abhijith | **Shared model + metrics module** | `MobileNetV3SmallDetector` (E1) importable; EER/AUROC/Macro-F1 single implementation | Missing: duplicated in notebooks |
| P3 | Abhijith | **Train/eval entrypoint** | `python -m src.training.train --config experiments/...yaml` reproduces E1 v4 eval within ±0.5% Macro-F1 | Missing |
| P4 | Team | **Environment lock** | `requirements.txt` or `environment.yml` with torch, librosa, versions pinned | Missing |
| P5 | Bhuvan | **Degradation library** | `src/robustness/apply.py`: deterministic + stochastic modes; returns `(audio_out, metadata)` | Missing |
| P6 | Bhuvan | **Condition matrix + manifests** | CSV/JSON listing `pair_id, clean_path, condition_id, severity, degraded_path`; traceable to source clip | Missing |
| P7 | Bhuvan | **Paired-view dataset API** | Given `pair_id`, yields `(x_clean, x_degraded, label, meta)` with identical label and crop policy | Missing |
| P8 | Nikkath | **Evaluation harness** | Clean eval + conditioned eval on **same** utterances; computes ΔM for Macro-F1, AUROC, EER | Missing |
| P9 | Team | **Experiment registry** | Naming: `model_condition_split_seed`; table of “official” runs for final report | Partial |
| P10 | Team | **Manifests in repo or DVC** | SHA256 + generation script documented; new member can rebuild splits | Partial: scripts only |

**Phase D start rule:** P1, P2, P5, P6, P7, P8 must be **Done**; P3 strongly recommended before large γ-sweeps.

---

## 3. Standardized interfaces (contracts)

### 3.1 Preprocessing (existing — do not fork)

- **Module:** `src/preprocessing/preprocess_atadd_v4.py`
- **Fixed for robustness experiments:** 24 kHz, 4.0 s, repeat-pad, GPU log-mel (80 bins), global mean/std from **training set only**
- **Rule:** Degradation applies to **waveform** before mel; never re-tune mel stats per condition

### 3.2 Degradation API (Bhuvan → all)

```text
degrade(
  waveform: np.ndarray,
  sample_rate: int,
  condition: ConditionSpec,   # family, codec, bitrate, snr, seed
  mode: "deterministic" | "random",
) -> (waveform_degraded, ConditionMetadata)
```

**Required metadata fields:** `source_id`, `condition_family`, `condition_name`, `severity`, `seed`, `pipeline_version`.

**Condition families (minimum for E5–E7):** MP3/AAC/Opus bitrate ladder, resample chain, telephony band-limit, additive noise SNR ladder, single-step transcoding, optional reverb.

### 3.3 Paired batch (Nikkath training)

For each training step, sample clip `u`:

1. Load clean waveform `w`
2. `w_deg = degrade(w, condition ~ π_train)`
3. Optional: shared random crop start index for both (same time segment)
4. Forward: `f(x(w))`, `f(x(w_deg))` where `x` is mel frontend

### 3.4 Evaluation harness

| Split | Use |
|-------|-----|
| Train | Training only |
| Dev | Early stopping **only**; report but do not use as primary claim |
| Eval (frozen) | **All paper/challenge claims**; never used for hyperparameter tuning |

**Robustness eval sets:**

- `eval_clean` — existing eval manifest  
- `eval_conditioned_{c}` — same files as eval, pre-generated degraded copies (deterministic seeds)  
- Report **ΔM = M_clean − M_conditioned** per condition and macro-average

---

## 4. Baseline alignment fixes (Abhijith — before comparing E7)

1. **Frontend parity:** All ASV comparisons at 24 kHz + repeat-pad if comparing to AT-ADD v4 story.  
2. **Model selection metric:** Document that dev Macro-F1 ≫ eval Macro-F1 on AT-ADD (generator-disjoint dev still “easier”); for robustness work, consider **eval-clean Macro-F1** or **mean ΔM** for checkpoint selection alongside dev.  
3. **E3 SSL reference (plan):** Can remain Abhijith-owned; Nikkath method must beat **E1 v4 compact** under same conditioned eval, not only SSL.

---

## 5. Repository layout (target — align with plan Section 14)

```text
src/
  models/           # MobileNetV3 E1 (+ optional E2 head)
  datasets/         # manifest loaders + PairedRobustnessDataset
  preprocessing/    # existing v4
  robustness/       # Bhuvan: conditions, degrade(), matrix builder
  training/         # train loop, losses (baseline + proposed)
  evaluation/       # metrics, ΔM, tables, figure scripts
configs/
  baselines/
  robustness/
  proposed/
experiments/
  baselines/        # existing
  robustness/       # E5, E6 artifacts
  proposed/         # E7, E8 artifacts
docs/
  proposed/         # this folder
```

---

## 6. Minimum artifacts for Nikkath individual package (Section 13)

Before viva, Nikkath folder must contain:

1. `src/training/losses/proposed_*.py` (or equivalent) — **attributable module**  
2. Config: `experiments/proposed/E7_ft_rcl_*.yaml`  
3. Table: baseline CE vs proposed, clean + conditioned, ΔM  
4. Ablation: remove spectral coherence term; γ sweep; severity curriculum on/off  
5. One **failure analysis**: e.g. music/sound where robustness does not recover EER  

---

## 7. Suggested team timeline (2-week gates)

| Week | Milestone |
|------|-----------|
| W1 | P4, P2, P3 — refactor E1 v4 into train script |
| W2 | P5, P6 — condition matrix on **dev subset** first, then full eval cache |
| W3 | P7, P8 — paired loader + ΔM tables on **baseline CE** (E5) |
| W4+ | Phase D — proposed loss (FT-RCL spec in `01_NOVEL_METHOD_FT-RCL_SPEC.md`) |

---

## 8. Sign-off template

| Gate | Owner | Date | Signed |
|------|-------|------|--------|
| P1 Frozen E1 v4 weights available to team | Abhijith | | |
| P5 Degradation API v0.1 | Bhuvan | | |
| P6 Eval conditioned manifests | Bhuvan | | |
| P8 ΔM harness on baseline | Nikkath | | |
| Phase D training start | Nikkath | | |
