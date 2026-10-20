# Joshi — Individual Deliverables & Defence

Maps to execution plan **Section 13** (Joshi row) and **Definition of Done** (compression + edge).

---

## 1. Minimum individual package

| Item | Path (planned) |
|------|----------------|
| Attributable code | `src/compression/*`, `src/deployment/*` |
| Reproducible config | `configs/edge/e9_rpd_star_edge.yaml` |
| Quantitative table | `results/tables/E12_edge_efficiency.csv` |
| Pareto / trade-off figure | `results/figures/delta_cep_pareto.png` |
| Written analysis | `docs/edge/analysis_joshi_limitations.md` (create after experiments) |
| Demo | 30 s screen recording: ONNX INT8 infer on wav file + printed fake prob |

---

## 2. Research narrative (examiner-facing)

1. **Problem:** Edge devices need small/fast detectors; naive quantization hurts **already fragile** robustness under MP3/telephony.  
2. **Method:** Δ-CEP — RPD + WCSP + MR-Calib + Δ-Pareto selection.  
3. **Evidence:** E9-T2 vs E9-T1 on ΔM; E11-Q2 vs E11-Q1; E12 latency/size.  
4. **Limitation:** Compression cannot recover music-domain generator failures; may widen EER on worst codec if pruned too aggressively.

---

## 3. Differentiation from teammates

| Member | Joshi must NOT claim |
|--------|----------------------|
| Abhijith | “I built the baseline CNN” |
| Bhuvan | “I built the codec simulator” |
| Nikkath | “I invented STAR-Edge losses” |

| Joshi SHOULD claim |
|--------------------|
| “I preserve Nikkath’s robustness **under** prune/INT8 using Δ-CEP” |
| “I prove **audio compression ≠ model compression** with paired metrics” |

---

## 4. Sample viva Q&A

**Q: Why not quantize the SSL teacher?**  
A: Out of scope and violates edge param budget; we distill into MobileNet student.

**Q: Why worst-condition pruning?**  
A: E5 shows heterogeneous ΔM; clean-sensitive pruning drops channels needed under MP3-32k.

**Q: Is MR-Calib novel?**  
A: Mixed calibration is known in CV; **our** contribution is tying it to **ADD ΔM** and ablation vs clean-only on **same** AT-ADD eval.

**Q: FP16 heads + INT8 trunk?**  
A: SAPD keeps CAD/subband stable; ablation E11-Q4.

---

## 5. Prompt for Joshi’s thesis subsection (copy to LLM)

```text
Write "Model Compression and Edge Deployment" chapter using attached Δ-CEP spec and E9-E12 plan.
Include: distinction audio vs model compression, RPD equations, WCSP algorithm steps, MR-Calib protocol, Δ-Pareto formula, empty results tables, one limitation paragraph on music domain.
800-1200 words, formal tone.
```

---

## 6. Coordination checkpoints with Nikkath

| Checkpoint | Action |
|------------|--------|
| Teacher frozen | Nikkath tags `star_edge_best_v1.pth` + eval JSON |
| Before INT8 | Joshi shares E11-Q1 result; if ΔM bad, Nikkath verifies teacher eval unchanged |
| E13 table | Single row agreed: `delta_cep_final` hyperparameters |
