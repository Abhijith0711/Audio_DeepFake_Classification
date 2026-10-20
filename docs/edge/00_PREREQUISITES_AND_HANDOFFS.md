# Prerequisites & Handoffs — Joshi (Compression / Edge)

**Phase E start rule:** Do not run large compression sweeps until **teacher model** and **ΔM harness** exist.

---

## 1. What Joshi depends on (from teammates)

| ID | From | Deliverable | Why Joshi needs it |
|----|------|-------------|-------------------|
| J1 | Abhijith | Exportable **E1 v4** checkpoint + `torch → ONNX` graph (mel in or wav in — pick one, document) | Baseline “before compression” row |
| J2 | Nikkath | Best **STAR-Edge** (or FT-RCL M3) checkpoint + config | Primary **teacher** for robustness-aware distillation |
| J3 | Nikkath + Bhuvan | **ΔM table** on eval (clean + conditioned), worst condition ID | Worst-condition pruning & calibration |
| J4 | Bhuvan | Frozen **eval_conditioned_*** manifests (deterministic) | Fair before/after compression robustness |
| J5 | Team | Shared **`src/evaluation/metrics.py`** | Same Macro-F1 / AUROC / EER / ΔM |
| J6 | Abhith/Nikkath | **`src/models/`** single MobileNet definition | Pruning/QAT hooks on one architecture |
| J7 | Team | **`requirements.txt`** + export versions (torch, onnx, onnxruntime) | Reproducible edge builds |

**Minimum to start E9:** J1, J5, J6.  
**Minimum for unique Δ-CEP story:** J2, J3, J4.

---

## 2. What Joshi must deliver back to the team

| ID | Deliverable | Consumer |
|----|-------------|----------|
| J-OUT-1 | `checkpoints/edge/` naming convention + best **INT8** candidate | E13 final matrix |
| J-OUT-2 | `results/tables/E12_edge_efficiency.csv` | Report §11.1 |
| J-OUT-3 | `src/deployment/export_onnx.py`, `benchmark_latency.py` | Reproducibility |
| J-OUT-4 | Trade-off figure: size/latency vs Macro-F1 vs **macro-ΔM** | All members’ conclusion slide |

---

## 3. Repository layout (target)

```text
src/
  deployment/
    export_onnx.py
    export_tflite.py          # optional
    quantize_ptq.py           # post-training INT8
    quantize_qat.py           # optional QAT
    benchmark_edge.py
  compression/
    distill_robust.py         # Δ-CEP stage 1
    prune_worst_condition.py  # Δ-CEP stage 2
    pareto_select.py          # Δ-CEP stage 3
experiments/
  edge/
    E9_distill/
    E10_prune/
    E11_int8/
    E12_benchmark/
configs/
  edge/
    e9_delta_cep_teacher_star_edge.yaml
results/
  tables/   # compression comparison
  figures/  # Pareto plots
```

---

## 4. Scope boundaries (viva-safe)

| In scope (Joshi) | Out of scope (other members) |
|------------------|------------------------------|
| Smaller/faster **same-task** binary detector | New detection algorithm (Nikkath) |
| Distillation / pruning / INT8 / export | Bhuvan degradation library |
| Latency, RAM, size, FLOPs protocol | AT-ADD baseline training (Abhijith) |
| **ΔM after compression** | Full SSL teacher training |

**Clarify in defence:** *Audio compression* (MP3) ≠ *model compression* (INT8). Joshi owns **model** compression; Bhuvan owns **audio** channel simulation.

---

## 5. Sign-off template

| Gate | Owner | Done |
|------|-------|------|
| J1 ONNX export of E1 v4 | Abhijith / Joshi | |
| J2 STAR-Edge teacher frozen | Nikkath | |
| J3 Worst-condition ID from E5 | Bhuvan + Nikkath | |
| E9 robust distillation complete | Joshi | |
| E12 edge benchmark complete | Joshi | |
