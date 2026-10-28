# Proposed Method: FT-RCL (Frequency-Time Robustness Coherence Learning)

**Full name:** Frequency-Time Robustness Coherence Learning for Compression-Robust Edge Audio Deepfake Detection  
**Contributor track:** Nikkath (Phase D — E7/E8)  
**Backbone:** MobileNetV3-Small + existing `preprocess_atadd_v4` (E1 v4)  
**Status:** Algorithm specification for implementation and thesis — **not coded in this document**

---

## 1. Problem statement

Lightweight detectors (E1 v4) reach usable performance on **clean**, generator-disjoint AT-ADD eval (~20% EER speech/singing) but:

1. **Real channels** (codec, telephony, noise, transcoding) shift mel statistics and collapse margins — plan metric **ΔM** not yet measured in-repo.  
2. **Naive consistency** (`L_cls + γ ||p_clean − p_deg||²`) can **hurt** music/environment domains where the clean model is already uncertain (your E1 v4 music EER ~46%).  
3. Literature pushes **heavy SSL/ALLM** (AT-ADD challenge winners, dual-domain fusion) — incompatible with **edge** constraints without Joshi’s compression stage.

**Goal:** Improve **ΔMacro-F1 / ΔAUROC / ΔEER** under Bhuvan’s condition matrix while **not regressing** clean eval vs E1 v4, using **≤5% extra parameters** and **≤15% training overhead** vs E1.

---

## 2. Novelty claims (defence-ready)

| # | Claim | vs prior art in `btp/` folder |
|---|--------|-------------------------------|
| N1 | **Dual coherence:** classification consistency **plus** band-wise spectral **signature** consistency on intermediate features | Differs from label-only consistency (plan §5.2); lighter than FT-GRPO (2601.02983 — ALLM CoT, not edge CNN) |
| N2 | **Type-adaptive robustness gating:** robustness loss scaled by audio **type** (speech/singing/music/sound) and **clean confidence** | Addresses all-type heterogeneity (2608.23437 AT-ADD; 2406.08052 general audio) ignored by speech-only robustness |
| N3 | **Severity curriculum:** pair views sampled along ordered degradation severity; coherence weight ramps with severity | Stronger than i.i.d. random augment (P2V 3770855; ADD/Vicomtech codec tracks) |
| N4 | **ΔM-aware validation:** checkpoint selection uses **robustness score** `J = M_clean − λ·ΔM_avg` on dev (not dev accuracy alone) | Directly ties to project objectives §2.3; mitigates dev≫eval optimism documented in README |

**Working title for thesis chapter:** *“FT-RCL: Teaching compact detectors to agree on decisions and frequency-time artifact structure under channel degradation.”*

---

## 3. Architecture (unchanged backbone + optional thin head)

```text
                    ┌─────────────────────────────────────┐
  x_clean ──► Mel ──► MobileNetV3 ──► GAP ──► z_clean ──► logits_clean ──► p_clean
                    │                    │
                    │                    └──► SubbandPool ──► s_clean  (K-dim signature)
                    │
  x_deg   ──► Mel ──► MobileNetV3 (shared weights) ──► z_deg, s_deg, p_deg
```

- **Shared weights** for clean and degraded (standard paired-view setup).  
- **SubbandPool (new, lightweight):** partition 80 mel bins into **K=8** bands; for each band compute mean activation over time on the **last spatial feature map** (before GAP). Stack → vector `s ∈ ℝ^K`, L2-normalize.  
- **Parameter overhead:** only pooling indices + optional 8→8 linear (can start with fixed bands only → **0 extra params**).

Optional extension (E8 ablation): tiny **FiLM** scale/shift from `type_id` on band features only (+~2K params).

---

## 4. Loss functions

Let `y` be binary label (real/fake), `p` = softmax probability of fake, `ℓ_cls` = class-weighted cross-entropy (same as E1 v4).

### 4.1 Prediction coherence (PC)

\[
\mathcal{L}_{PC} = \big\| p_{\text{clean}} - p_{\text{deg}} \big\|_2^2
\]

Use **probability** space, not logits (more stable under strong compression).

### 4.2 Frequency-time signature coherence (FTSC) — **core novelty**

Define normalized subband signatures \(s_{\text{clean}}, s_{\text{deg}} \in \mathbb{R}^K\).

\[
\mathcal{L}_{FTSC} = 1 - \cos(s_{\text{clean}}, s_{\text{deg}}) = 1 - \frac{s_{\text{clean}}^\top s_{\text{deg}}}{\|s_{\text{clean}}\|\|s_{\text{deg}}\|}
\]

**Rationale:** Codecs and telephony primarily **warp energy across frequency** while deepfake artifacts (vocoder ripples, phase inconsistencies) often leave **relative band structure** partially intact in fake clips; real clips may shift differently under noise. Forcing **cosine alignment** of band summaries under degradation encourages **artifact-relevant** invariance rather than raw mel identity (which is impossible after MP3).

**Optional temporal term (TSC):** on GRU/LSTM variant only — align last hidden states `h_clean`, `h_deg` with same cosine loss (ablation: E2 + FT-RCL).

### 4.3 Type- and confidence-gated robustness

Let `τ_type ∈ {speech, singing, music, sound}` with learnable or fixed weights `α_τ` (init: speech=1.0, singing=1.0, music=0.3, sound=0.5 from E1 failure analysis).

Let `c = |p_{\text{clean}} - 0.5| × 2` (clean confidence in [0,1]).

\[
w_{\text{rob}} = \alpha_{\tau} \cdot c \cdot \eta(severity)
\]

where `η(severity)` ramps from 0 at clean to 1 at max severity in curriculum (Section 5).

### 4.4 Total loss

\[
\mathcal{L} = \mathcal{L}_{cls}^{\text{clean}} + \beta \mathcal{L}_{cls}^{\text{deg}} + w_{\text{rob}} \left( \gamma_1 \mathcal{L}_{PC} + \gamma_2 \mathcal{L}_{FTSC} \right)
\]

**Defaults for first run:** `β=0.5`, `γ_1=1.0`, `γ_2=0.5`.  
**Ablations (E8):** remove FTSC; remove gating; fix `w_rob=1`; swap PC for KL.

**Distillation (optional, after E7 stable — Joshi/Nikkath boundary):** add `γ_3 KL(p_student || p_teacher)` where teacher is frozen E1 v4 or SSL reference on **clean only** — do not distill degraded views until baseline robustness proven.

---

## 5. Training protocol

### 5.1 Severity curriculum (pairs with Bhuvan’s matrix)

Order conditions by approximate severity `σ` (e.g. clean=0, resample mild=1, MP3 128k=2, …, MP3 32k + noise = max).

- **Epochs 1–5:** sample degraded views with `σ ≤ σ_1` (mild only)  
- **Epochs 6–12:** `σ ≤ σ_2`  
- **Epochs 13–20:** full matrix  

Within batch, **same clip** `u`, same crop offset, two conditions max (clean + one degraded) to limit compute.

### 5.2 Mixed training modes (E7 experiments)

| Mode | Description |
|------|-------------|
| M0 | Clean only (E1 baseline) |
| M1 | Random degraded, CE only on degraded |
| M2 | M1 + PC (standard consistency) |
| M3 | **FT-RCL full** (PC + FTSC + gating + curriculum) |

### 5.3 Checkpoint selection

On dev (not eval):

\[
J = \text{MacroF1}_{\text{clean}} - \lambda \cdot \overline{\Delta \text{MacroF1}}
\]

with `λ=0.5` initially; compare to E1’s **dev Macro-F1 only** selection in ablation table (shows why ΔM-aware selection matters).

---

## 6. Evaluation protocol (E7/E8 + E13)

**Primary (frozen eval):**

- Macro-F1, AUROC, EER on `eval_clean`  
- Same metrics on each `eval_conditioned_c`  
- **ΔM** per condition; **macro-Δ** across conditions  
- Breakdown by **type** and **generator** (reuse E1 v4 CSV schema)

**Secondary:**

- Cross-condition: train robustness on {MP3,AAC}, test Opus  
- Bona-fide sensitivity: false-positive rate on **real** only under degradation  

**Success criteria (internal, adjust after E5 baseline):**

- Macro-ΔF1 improved by **≥15% relative** vs M2 (PC-only) at similar clean Macro-F1 (±0.01)  
- Music/sound: no clean regression **>0.02** Macro-F1 vs E1 v4  

---

## 7. Expected failure modes (honest analysis for thesis)

1. **Music generators (SSM-TTM, ACE-Step):** channel robustness cannot fix generator blind spots — report separately.  
2. **Strong low-pass telephony:** FTSC may over-penalize; gating by severity should down-weight.  
3. **Compute:** two forwards per step — Joshi stage still needed for edge; FT-RCL is **training-time** robustness.  

---

## 8. Relation to reference papers (one-line positioning)

| Paper | Relevance | FT-RCL distinction |
|-------|-----------|-------------------|
| AT-ADD benchmark (2608.23437) | All-type + perturbations in Track 1 | Compact CNN + explicit FT coherence, not SSL fusion |
| FT-GRPO (2601.02983) | Frequency-time reasoning for ADD | No LLM; operational band signatures on CNN features |
| P2V (3770855.3817542) | Noisy real-world eval | Train-time coherence + ΔM metric, not new dataset |
| SLS / XLS-R (3664647.3681345) | SSL layer sensitivity | Subband pooling on **mel-CNN** maps, edge-feasible |
| Vicomtech ADD Wav2Vec2 | Codec/telephony experience | Student stays MobileNetV3 for Joshi compression path |
| TRACE (Khan, CVPRW 2026) | Embedding trajectory | Training-time trajectory across **severity**, not training-free partial |
| Sensors 2025 survey | Robustness gap | Implements plan’s consistency + measurable ΔM |
| VoiceRadar (10590871) | Physics-informed features | Optional future: add micro-freq branch as E8+ extension only |

---

## 9. Implementation mapping (when coding starts)

| Component | Path (planned) |
|-----------|----------------|
| SubbandPool + signatures | `src/models/subband_pool.py` |
| FT-RCL loss | `src/training/losses/ft_rcl.py` |
| Gating + curriculum | `src/training/schedulers/robustness_curriculum.py` |
| Config | `configs/proposed/e7_ft_rcl_atadd_v1.yaml` |
| Run script | `experiments/proposed/E7/README.md` + train command |

---

## 10. One-slide pitch for guide

> We keep the E1 v4 edge backbone. Under Bhuvan’s degradations, we train the network so that **fake/real decisions** and **mel-band artifact summaries** stay coherent— but only when the model is already confident and for audio types where robustness helps. Checkpoint selection for M0–M3 is **development macro-F1**, with EER logged every epoch. Ablations isolate frequency-time coherence vs standard prediction consistency.

---

## 11. Stronger novelty variant — **STAR-Edge** (future work)

**Status:** STAR-Edge is not in the training code. The loss module, channel encoder, artifact projector, and `e7_star_edge_atadd_v1.yaml` were removed. The runnable modes are M0, M1, M2, and M3. The rest of this section is a design note for later work.

**Name:** **S**everity-**T**rajectory **A**rtifact **R**obustness for **Edge** detection  
**Relation to FT-RCL:** STAR-Edge **includes** FT-RCL (PC + FTSC + gating + curriculum + ΔM selection) and adds three mechanisms that are harder to dismiss as “standard consistency.”

### 11.1 Why FT-RCL alone may feel incremental

Examiners can say: *“Prediction consistency + augmentation is well known.”*  
STAR-Edge adds **structure** (trajectory + disentanglement + asymmetric real-audio constraint) that maps to clear hypotheses and ablations.

### 11.2 New mechanism A — Severity trajectory coherence (STC)

**Idea (TRACE-inspired, but training-time and edge-scale):**  
One clean/degraded **pair** is weak evidence. For the **same clip** and **same crop**, build a **ladder** of views:

\[
x_0 \ (\text{clean}),\ x_1\ (\sigma_1),\ x_2\ (\sigma_2),\ \ldots,\ x_T\ (\sigma_T)
\]

Compute signatures \(s_t\) and fake probabilities \(p_t\).

**Loss — smooth artifact path, not identical mel:**

\[
\mathcal{L}_{STC} = \sum_{t=1}^{T-1} \left(1 - \cos(s_t, s_{t+1})\right) + \mu \sum_{t=1}^{T-1} (p_t - p_{t+1})^2
\]

Optional **curvature penalty** (stronger novelty, one ablation):

\[
\mathcal{L}_{curv} = \sum_{t=1}^{T-2} \| (s_{t+1} - s_t) - (s_t - s_{t-1}) \|_2^2
\]

**Hypothesis:** Codecs change features **gradually** with severity; deepfake artifact structure should trace a ** smoother, more predictable path** in band-signature space than naive consistency from clean→worst only.

**Inference cost:** **Zero** — only used during training (2–4 forwards per clip on a subset of batches).

**Distinction from TRACE (Khan, CVPRW 2026):** they analyze **embedding trajectories for partial spoofing at test time without training**; you **learn** trajectory regularity for **full-clip binary** detection under **Bhuvan’s channel matrix**.

### 11.3 New mechanism B — Channel–artifact disentanglement (CAD)

**Problem:** Coherence on full CNN features can force the model to **preserve codec fingerprints** instead of **spoof cues**.

**Idea:** Split representation into:

- **Channel code** \(u\) — “how damaged is the transmission?” (auxiliary, cheap)  
- **Artifact code** \(v\) — “real vs fake?” (primary)

**Channel encoder (no extra heavy model):**  
Handcrafted stats on waveform (band energy ratios, spectral flatness, crest factor) → tiny MLP (~1–3K params) → \(u \in \mathbb{R}^{d_c}\).

**Backbone feature** \(z\) from MobileNet (pre-GAP). **Artifact projection:**

\[
v = z - \mathrm{proj}(z \mid u) = z - u (u^\top u)^{-1} u^\top z
\]

(use detached \(u\) for projection during warm-up epochs 1–3, then joint).

**Train:**

- Auxiliary: predict **degradation family** (or severity bin) from \(u\) with CE — \( \mathcal{L}_{chan}\)  
- Authenticity: CE on fake/real from **SubbandPool(\(v\))** only — \( \mathcal{L}_{cls}\)  
- FT-RCL coherence applies to **\(s(v)\)** not raw \(s(z)\)

**Hypothesis:** Robustness losses act on **artifact subspace orthogonal to channel**, reducing “learn MP3 artifact = fake” shortcuts.

**Distinction:** Not multimodal AV; not SSL — **explicit linear disentanglement** on edge CNN.

### 11.4 New mechanism C — Asymmetric bona-fide robustness (ABR)

Bhuvan’s plan asks whether transforms **increase false positives on real audio**.

For **real** clips (\(y=0\)), add:

\[
\mathcal{L}_{ABR} = \max(0,\ p_{\text{deg}} - p_{\text{clean}} + m)
\]

margin \(m \approx 0.05\). **Do not apply symmetrically on fake** (fakes may legitimately shift probability under noise).

**Hypothesis:** Improves **real-only FPR** under degradation without forcing fake clips to match clean probability exactly.

### 11.5 STAR-Edge total loss

\[
\mathcal{L} = \mathcal{L}_{cls} + \mathcal{L}_{chan} + w_{\text{rob}}(\gamma_1 \mathcal{L}_{PC} + \gamma_2 \mathcal{L}_{FTSC}) + \delta \mathcal{L}_{STC} + \rho \mathcal{L}_{ABR}
\]

(+ optional \(\mathcal{L}_{curv}\)). Keep **gating** \(w_{\text{rob}}\) from §4.3.

**Suggested defaults:** \(\delta=0.3\), \(\rho=0.2\), trajectory length \(T=3\) per step (clean + mild + strong).

### 11.6 Updated novelty claims (use in thesis if STAR-Edge is adopted)

| # | Claim |
|---|--------|
| N5 | **Severity trajectory coherence** — multi-step degradation path regularization (not single pair) |
| N6 | **Channel–artifact disentanglement** — robustness on projected artifact subspace + channel auxiliary task |
| N7 | **Asymmetric bona-fide robustness** — explicit real-audio FPR control under degradation |
| N1–N4 | Retained from FT-RCL |

### 11.7 Extra ablations (E8-bis)

| Run | What it tests |
|-----|----------------|
| S0 | FT-RCL (M3) baseline |
| S1 | S0 + STC only |
| S2 | S0 + CAD only |
| S3 | S0 + ABR only |
| S4 | **STAR-Edge full** (S0 + STC + CAD + ABR) |
| S5 | S4 without \(\mathcal{L}_{curv}\) |

**Defence line:** *“Each block addresses a different failure mode: pair consistency (FT-RCL), path stability (STC), shortcut removal (CAD), real-user false alarms (ABR).”*

### 11.8 Cost / scope

| Item | Overhead vs E1 |
|------|----------------|
| FT-RCL | ~2× forward / step |
| STAR-Edge (T=3 ladder on 50% batches) | ~2.5× average forward / step |
| CAD MLP | +~2K params |

Still compatible with Joshi INT8 stage on **same** MobileNet backbone.

### 11.9 Honest ceiling

STAR-Edge is **more novel** than FT-RCL for an honours viva, but it is still **not** claiming a new foundation model or benchmark. Success = **ablations + ΔM gains + real-only FPR story** on AT-ADD frozen eval.

### 11.10 Implementation add-ons

Not implemented. Do not add these paths back unless STAR-Edge is revived as a separate experiment.
