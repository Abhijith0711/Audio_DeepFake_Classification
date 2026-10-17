# Joshi Track — Model Compression & Edge Deployment (Phases E–F)

**Owner:** Joshi Kshitij Santosh (2023BCS0087)  
**Status:** Design documents only (no implementation yet)  
**Execution plan mapping:** Experiments **E9–E12**, plan Section 8, Phase E–F

| Document | Purpose |
|----------|---------|
| [00_PREREQUISITES_AND_HANDOFFS.md](./00_PREREQUISITES_AND_HANDOFFS.md) | Gates J1–J10, team dependencies |
| [01_NOVEL_METHOD_DELTA-CEP_SPEC.md](./01_NOVEL_METHOD_DELTA-CEP_SPEC.md) | **Δ-CEP** — unique compression + edge method |
| [02_EXPERIMENT_PLAN_E9-E12.md](./02_EXPERIMENT_PLAN_E9-E12.md) | Distillation, pruning, INT8, benchmark matrix |
| [03_EDGE_BENCHMARK_PROTOCOL.md](./03_EDGE_BENCHMARK_PROTOCOL.md) | Latency, RAM, size, FLOPs — reproducible protocol |
| [04_DEFENCE_AND_DELIVERABLES.md](./04_DEFENCE_AND_DELIVERABLES.md) | Individual marks evidence, viva Q&A |

**Proposed method name:** **Δ-CEP** — **Δ**-aware **C**ompression & **E**dge **P**ackaging for robust audio deepfake detectors.

**Integration:** Compresses the **same** MobileNetV3-Small backbone used by Abhijith (E1 v4) and Nikkath (FT-RCL / STAR-Edge), selected by **clean quality + robustness drop (ΔM)**, not clean accuracy alone.
