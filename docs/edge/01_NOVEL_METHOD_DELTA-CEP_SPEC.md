# Proposed Method: Δ-CEP (Δ-aware Compression & Edge Packaging)

**Full name:** Δ-aware Compression and Edge Packaging for Robust Lightweight Audio Deepfake Detectors  
**Owner:** Joshi — Phase E–F (E9–E12)  
**Student backbone:** MobileNetV3-Small (+ optional STAR-Edge heads: subband pool, channel MLP)  
**Status:** Specification only — not implemented

---

## 1. Problem

Standard compression pipelines optimize **clean** validation accuracy. For this project, the product requirement is different:

> A edge model must stay useful **after real transmission**, not only on clean eval.

If Joshi only reports “INT8 saved 4× size, clean F1 dropped 0.01,” that matches generic ML engineering — **weak honours novelty**.

**Δ-CEP** optimizes and **selects** compressed models using **clean performance + robustness degradation (ΔM) + latency + size** together, and uses **compression techniques calibrated on degraded audio**, not clean-only.

---

## 2. Novelty claims (defence-ready)

| # | Claim | vs typical “we used PyTorch quantize” |
|---|--------|----------------------------------------|
| C1 | **Δ-Pareto selection:** choose operating point by multi-objective score including **macro-ΔM**, not clean F1 alone | Links compression to project research question |
| C2 | **Worst-condition sensitivity pruning (WCSP):** structured prune using importance on **worst channel from E5**, not generic L1-norm | Targets robustness bottleneck |
| C3 | **Mixed-reality INT8 calibration:** PTQ/QAT calibration batches = **50% clean mel + 50% conditioned mel** (same clips) | Reduces INT8 collapse under codec noise |
| C4 | **Robustness-preserving distillation (RPD):** teacher is STAR-Edge; student matches **clean + degraded** logits (paired batches from Nikkath pipeline) | Distillation aligned with FT-RCL training objective |
| C5 | **Split-precision packaging (optional):** MobileNet INT8 + tiny channel/artifact heads FP16 when STAR-Edge CAD is used | Edge-realistic hybrid deployment |

**Honest limit:** Δ-CEP is **novel as an integrated recipe** for **robust** ADD on **edge CNNs**; individual techniques (KD, pruning, PTQ) are known.

---

## 3. Pipeline overview (four stages)

```text
Teacher (STAR-Edge or E1 v4, FP32)
        │
        ▼
┌───────────────────┐
│ E9  RPD distill   │  Student: same width or MobileNetV3-Width-0.75
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ E10 WCSP prune    │  Structured (depthwise channels / blocks)
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ E11 Mixed INT8    │  PTQ primary; QAT if ΔM regresses > threshold
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ E12 Δ-Pareto pick │  ONNX/ORT (+ optional TFLite) + benchmark
└───────────────────┘
```

---

## 4. Stage E9 — Robustness-preserving distillation (RPD)

### 4.1 Teacher / student

| Role | Model |
|------|--------|
| Teacher (frozen) | Best **STAR-Edge** checkpoint (preferred) or **E1 v4** |
| Student | Same architecture **or** `MobileNetV3-Small-0.75×` width (document param count) |

### 4.2 Batch structure (requires Nikkath + Bhuvan)

For each clip: `(x_clean, x_deg, y)` — same as robust training.

Teacher outputs: `p_T^clean`, `p_T^deg` (fake probabilities).

Student: `p_S^clean`, `p_S^deg`.

### 4.3 RPD loss

\[
\mathcal{L}_{RPD} = \alpha \ CE(y, p_S^{clean}) + \beta \ CE(y, p_S^{deg}) + \tau \ KL(p_T \| p_S)\ \text{(both views)}
\]

With temperature \(T_{kd}\) on logits for KL terms.

**Default:** \(\alpha=0.5, \beta=0.5, \tau=0.3\).

**Distinction from generic KD:** \(\beta>0\) and KL on **degraded view** explicitly — student must mimic teacher **under channel stress**.

**Optional (if STAR-Edge):** match artifact signature cosine teacher-student on degraded view only (light weight 0.1) — ties to Nikkath’s FTSC without full STAR-Edge retrain.

---

## 5. Stage E10 — Worst-condition sensitivity pruning (WCSP)

### 5.1 Motivation

Channels that matter on **clean** mel may be redundant under INT8; channels that matter on **worst ΔM condition** must be preserved.

### 5.2 Procedure

1. From E5/E3 table, pick condition \(c^\*\) = argmax **ΔMacro-F1** on **teacher** (e.g. MP3-32k + noise).  
2. For each prunable unit (e.g. depthwise block output channel), score:

\[
\text{importance}_j = \mathbb{E}_{batch} \left| \frac{\partial \mathcal{L}_{cls}}{\partial h_j} \right| \Big|_{x \sim \mathcal{D}_{c^\*}}
\]

Run scoring on **conditioned eval subset** (not train — use dev conditioned to avoid overfit; document choice).

3. **Structured prune** lowest 10% → 20% → 30% (ladder); fine-tune 3–5 epochs each with RPD loss.

### 5.3 Ablation (Joshi E10)

| Run | Pruning criterion |
|-----|-------------------|
| P0 | No prune |
| P1 | Magnitude-only (L1 on weights) — **generic baseline** |
| P2 | **WCSP** (worst-condition gradients) |

**Success:** P2 same or better **macro-ΔM** than P1 at equal param count.

---

## 6. Stage E11 — Mixed-reality INT8 calibration (MR-Calib)

### 6.1 PTQ (primary path)

- Framework: PyTorch `quantize` or ONNX Runtime static quantization (document one).  
- **Calibration set:** N=512–1024 mel tensors:

  - 50% from **clean** train/dev crop  
  - 50% from **Bhuvan-conditioned** pairs (same clips, deterministic seeds)

- **Per-channel** weights where supported (Conv layers).

### 6.2 QAT (fallback)

If PTQ causes **macro-ΔM** regression > 15% relative vs FP32 student:

- 5–10 epoch QAT with same RPD loss, fake-quant modules.

### 6.3 Split-precision (SAPD) — if STAR-Edge deployed

| Module | Precision |
|--------|-----------|
| MobileNetV3 trunk | INT8 |
| SubbandPool + classifier | FP16 |
| Channel MLP (CAD) | FP16 |

**Rationale:** Tiny heads suffer most from aggressive INT8; trunk dominates FLOPs.

---

## 7. Stage E12 — Δ-Pareto operating point selection

### 7.1 Metrics vector

For each candidate \(k\):

\[
\mathbf{m}_k = (\text{F1}_{clean},\ \text{macro-ΔF1},\ \text{size}_{MB},\ \text{latency}_{ms},\ \text{params},\ \text{peakRAM}_{MB})
\]

### 7.2 Scalar score (for ranking, not replacing full Pareto plot)

\[
S_k = \text{F1}_{clean} - \lambda_\Delta \cdot \overline{\Delta \text{F1}} - \lambda_L \cdot \frac{L_k}{L_{baseline}} - \lambda_S \cdot \frac{\text{size}_k}{\text{size}_{baseline}}
\]

Normalize terms to [0,1] scale before weighting. **Defaults:** \(\lambda_\Delta=0.4, \lambda_L=0.2, \lambda_S=0.1\).

**Novelty:** Explicit **λ_Δ** ties Joshi’s work to Nikkath/Bhuvan robustness — examiners see one story.

### 7.3 Final edge candidate

Pick **non-dominated** points on (latency, macro-ΔF1) Pareto front; report **one “deployment” point** approved with guide (e.g. max latency 50 ms on target device).

---

## 8. Evaluation (same frozen protocol as team)

| Test | Required |
|------|----------|
| AT-ADD eval **clean** | Yes |
| AT-ADD eval **each major condition** | Yes — compute ΔM vs FP32 teacher |
| ASV eval (optional appendix) | Speech-only compression stress test |
| Real-only FPR under degradation | Yes — show ABR/robust teacher benefit survives INT8 |

**Compare against:**

- FP32 E1 v4  
- FP32 STAR-Edge  
- INT8 from **clean-only** calibration (ablation C3 off)

---

## 9. Expected results narrative

1. **Size/latency:** 3–8× smaller, 2–5× faster (device-dependent — measure, don’t invent).  
2. **Clean F1:** small drop acceptable.  
3. **ΔM:** Δ-CEP should **beat** clean-calibrated INT8 on macro-ΔF1 at same bit-width.  
4. **Failure case:** music domain may still dominate error — compression won’t fix; report separately.

---

## 10. Relation to Nikkath STAR-Edge

| Nikkath | Joshi |
|---------|--------|
| Learns robust **representations** | Preserves robustness **under bit-width + prune** |
| STAR-Edge losses at train | RPD transfers STAR behavior to smaller/INT8 student |
| ΔM-aware checkpoint | Δ-Pareto **deployment** checkpoint |

**Team slide:** *Train robust (Nikkath) → Deploy small (Joshi) → Measure on Bhuvan’s matrix.*

---

## 11. Implementation file map (future)

| File | Role |
|------|------|
| `src/compression/distill_robust.py` | E9 RPD |
| `src/compression/prune_wcsp.py` | E10 |
| `src/deployment/quantize_mrcalib.py` | E11 |
| `src/deployment/pareto_select.py` | E12 |
| `src/deployment/benchmark_edge.py` | Latency/RAM |
| `configs/edge/e9_rpd_star_edge.yaml` | |

---

## 12. One-slide pitch (Joshi defence)

> Generic quantization minimizes clean accuracy loss. **Δ-CEP** distills a robust STAR-Edge teacher on **both clean and degraded** views, prunes using **worst-case channel sensitivity**, calibrates INT8 on **mixed clean/degraded** mel batches, and selects the shipped model on a **Δ-aware Pareto** frontier — so edge deployment is measured on **robustness**, not lab-clean scores only.
