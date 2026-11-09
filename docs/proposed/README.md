# Proposed Robustness Track — Documentation Index

**Owner:** Nikkath (Phase D) · **Status:** Design only — no training code here yet

| Document | Purpose |
|----------|---------|
| [00_PIPELINE_PREREQUISITES.md](./00_PIPELINE_PREREQUISITES.md) | What the team must finish before your robustness experiments |
| [01_NOVEL_METHOD_FT-RCL_SPEC.md](./01_NOVEL_METHOD_FT-RCL_SPEC.md) | **FT-RCL** algorithm: losses, gating, curriculum, evaluation |
| [02_EXPERIMENT_AND_ABLATION_PLAN.md](./02_EXPERIMENT_AND_ABLATION_PLAN.md) | E5–E8, E13 tables and run IDs |
| [03_RELATED_WORK_AND_LITERATURE_MAP.md](./03_RELATED_WORK_AND_LITERATURE_MAP.md) | `btp/` PDFs mapped to thesis sections |
| [04_DOCUMENT_GENERATION_PROMPTS.md](./04_DOCUMENT_GENERATION_PROMPTS.md) | Copy-paste prompts for TRD, thesis, handoffs |
| [05_MASTER_CURSOR_CODE_GENERATION_PROMPT.md](./05_MASTER_CURSOR_CODE_GENERATION_PROMPT.md) | **Main Cursor agent prompt** — structured code, logs, tables, figures |

## Proposed method names

- **FT-RCL** — Frequency-Time Robustness Coherence Learning (baseline proposed method).  
- **STAR-Edge** — future work only. It is not in the training code. The design note is §11 of `01_NOVEL_METHOD_FT-RCL_SPEC.md`.

The runnable methods are M0, M1, M2, and M3 on the E1 v4 MobileNetV3-Small backbone.

## Next step

1. Share `01_NOVEL_METHOD_FT-RCL_SPEC.md` + Prompt 7 with guide for approval.  
2. Bhuvan/Abhijith close prerequisites P5–P3 (see doc 00).  
3. Run E5 on frozen E1 v4 → then implement using Prompt 10.

## Joshi (compression / edge)

See **[`docs/edge/`](../edge/README.md)** — **Δ-CEP** method and experiments **E9–E12**.
