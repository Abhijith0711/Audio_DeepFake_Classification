# Edge Benchmark Protocol (E12)

**Owner:** Joshi (+ Nikkath for co-benchmark if guide assigns Phase F jointly)  
**Goal:** Numbers in the report are **repeatable** by another student on the same hardware class.

---

## 1. Target platforms (pick ≥2)

| Tier | Example | Role |
|------|---------|------|
| A — Reference | Laptop CPU (no GPU), batch=1 | Everyone can reproduce |
| B — Edge class | Raspberry Pi 4 (4GB) or Pi 5 | Plan “edge-device constraints” |
| C — Optional | Android phone via NNAPI / TFLite | Extra credit if stable |

Document **exact** device model, OS version, power plugged vs battery.

---

## 2. Model artifacts

| Format | Use |
|--------|-----|
| ONNX | Primary interchange |
| ONNX Runtime (`CPUExecutionProvider`) | Tier A/B latency |
| PyTorch FP32 | Teacher parity check |
| INT8 ORT quantized | Δ-CEP candidate |

**Input contract:** Fixed **4.0 s**, **24 kHz**, repeat-pad — mel computed **on device** or **precomputed mel input** — **choose one** and never mix in comparisons.

**Recommended for fair edge story:**

- **On-device:** waveform → mel (if feasible on Pi) → network  
- **If mel on server only:** report “feature extraction latency” separately from “classifier latency”

---

## 3. Latency protocol

- **Warmup:** 50 inferences (discard)  
- **Measure:** 500 inferences, batch=1  
- **Report:** mean, p50, p95, std (ms)  
- **Threading:** ORT `intra_op_num_threads=4` (document if changed)  
- **No GPU** for edge tier B unless explicitly labeled “edge GPU”

---

## 4. Memory protocol

- **Peak RSS** of benchmark process (e.g. `/usr/bin/time -v` on Linux, `ps` sampling)  
- Report **model file size** on disk (MB, 2 decimal)  
- Optional: ORT memory arena off for worst-case estimate (document)

---

## 5. Compute proxy

- **Params** — count from checkpoint  
- **FLOPs** — one forward, input shape `[1, 1, n_mels, n_frames]` documented  
- Tool: `fvcore` or `ptflops` (pin version in requirements)

---

## 6. Quality on edge

Run **identical** eval manifest as server:

- If INT8 logits differ slightly, metrics should be within **0.01 Macro-F1** of server ORT — else debug export.

Report **clean + conditioned** metrics on Tier A at minimum (Tier B may subsample eval for time — document subsample seed).

---

## 7. Required log fields per run

```json
{
  "run_id": "e12_mobilenetv3_delta_cep_int8_s42",
  "device_tier": "A",
  "device_model": "...",
  "onnx_opset": 17,
  "quantization": "int8_mrcalib",
  "latency_ms": {"mean": 0, "p50": 0, "p95": 0},
  "model_size_mb": 0,
  "peak_rss_mb": 0,
  "params": 0,
  "flops_g": 0,
  "macro_f1_clean": 0,
  "macro_delta_f1": 0
}
```

Store under `results/logs/edge/`.

---

## 8. Figures for report

1. **Pareto:** latency vs Macro-F1 (clean), points colored by macro-ΔF1  
2. **Bar:** size reduction vs teacher  
3. **Table:** plan Section 8.4 filled  

---

## 9. Common mistakes to avoid

- Tuning INT8 on **eval** set  
- Comparing GPU training latency to CPU deployment  
- Reporting **accuracy** only on spoof-heavy eval without Macro-F1  
- Omitting **degraded** eval for compressed model
