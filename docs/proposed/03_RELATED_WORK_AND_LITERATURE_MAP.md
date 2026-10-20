# Literature Map (`btp/` references → project use)

Use this when writing Related Work, PPT slide 2–3, and viva “how is your work different?”

---

## 1. Papers grouped by theme

### A. Surveys & foundations

| File | Citation theme | Use in thesis |
|------|----------------|---------------|
| `Audio_Deepfake_Approaches.pdf` | Classic detectors, EER landscape | Background §2 |
| `sensors-25-01989-v2.pdf` | 2025 survey: robustness, datasets, frontends | Gap: real-world channel + scalability |
| `s10791-026-10077-1 (1).pdf` | SLR: noise, codecs, generalization limits | Motivation for ΔM |
| `Engineering Reports - 2025 - Shaaban - ...` | DL detection overview | Baseline CNN justification |

### B. General / all-type audio deepfakes

| File | Theme | Contrast with FT-RCL |
|------|--------|----------------------|
| `2406.08052v1.pdf` (FakeSound) | General audio deepfakes, localization | You do binary all-type via AT-ADD T2 |
| `2608.23437v1.pdf` (AT-ADD) | Benchmark, perturbations, SSL winners | You: **lightweight** + coherence training |
| `2608.29021v1.pdf` | Dual SSL fusion (EAT + XLS-R) | You: no 300M SSL at inference |
| `2601.02983v1.pdf` (FT-GRPO) | ALLM frequency-time CoT | You: FT structure on **CNN bands**, edge |

### C. Robustness, codec, telephony

| File | Theme | Contrast |
|------|--------|----------|
| `The_Vicomtech_...Wav2vec2...ADD_Challenge.pdf` | ADD tracks, low-quality audio | Align condition families with their setup |
| `3770855.3817542.pdf` (P2V) | Perturbed real-world voices dataset | You train coherence, not new dataset |
| `2024.emnlp-main.286.pdf` | (check: likely robustness/NLP-audio) | Cite if codec/text-audio |
| `3658644.3670285.pdf` (SafeEar) | Content-free detection | Different threat model |
| `jimaging-09-00122.pdf` | MP3/compression angles | Support codec ladder design |

### D. SSL / layer sensitivity

| File | Theme | Contrast |
|------|--------|----------|
| `3664647.3681345.pdf` (SLS) | XLS-R layer selection | FT-RCL: fixed mel subbands, cheap |
| `3736765.pdf` (SONAR) | Benchmark many TTS | Optional generalization cite |

### E. Multimodal (do not blur team scope)

| File | Note |
|------|------|
| `Kim_X-AVDT_...`, `AVoiD-DF_...`, `Joint_Audio-Visual_...` | Cite as **out of scope** (audio-only edge) unless future work |

### F. Novel detection ideas (inspiration only)

| File | Idea | FT-RCL borrow |
|------|------|----------------|
| `khan_TRACE_...CVPRW_2026.pdf` | Embedding trajectory | Severity trajectory in ablation/extension |
| `10590871.pdf` (VoiceRadar) | Physics micro-frequencies | Future branch, not v1 |
| `1-s2.0-S0957417424008078-main.pdf` | Adversarial attacks | Limitation paragraph |

---

## 2. Suggested Related Work narrative (3 paragraphs)

**Paragraph 1 — Detection & datasets:** Progress from ASVspoof/ADD to all-type benchmarks (AT-ADD, FakeSound). Emphasize generator-disjoint eval and your AT-ADD E1 v4 baseline numbers.

**Paragraph 2 — Robustness gap:** Surveys note codec, telephony, noise sensitivity; challenge systems use heavy SSL + fusion. Gap: **compact models** with explicit **train-time** robustness and **ΔM** reporting on all-type data.

**Paragraph 3 — Our contribution:** FT-RCL adds **frequency-time signature coherence** and **type/confidence gating** on MobileNetV3-Small, with severity curriculum and ΔM-aware model selection — implementable on edge unlike dual SSL or ALLM approaches.

---

## 3. BibTeX task list

For each PDF you cite in the final report:

1. Export citation from Google Scholar / publisher  
2. Store `docs/references.bib` (create when writing thesis)  
3. Map `cite_key → experiment claim` in a spreadsheet  

---

## 4. What **not** to claim as novelty

- Binary real/fake on AT-ADD (benchmark exists)  
- MobileNetV3 baseline (E1 done by Abhijith)  
- Raw augmentation without coherence (standard industry practice)  
- Beating challenge winner 96% F1 (different compute and SSL)  

**Do claim:** FTSC + gating + curriculum + ΔM checkpointing as an **integrated** edge-oriented training recipe with ablations on **your** frozen splits.
