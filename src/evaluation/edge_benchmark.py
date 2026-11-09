"""Deployment-oriented measurements: size, MACs, latency (including the mel frontend), memory
and INT8 accuracy of a trained checkpoint on the host CPU."""
import argparse
import io
import json
import resource
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.flop_counter import FlopCounterMode

from src.datasets.atadd_manifest_dataset import audio_path_from_row, find_project_root, load_manifest_df, stratified_indices
from src.evaluation.eval_runner import build_model_from_checkpoint
from src.evaluation.metrics import calculate_metrics
from src.preprocessing.preprocess_atadd_v4 import load_audio, repeat_pad_audio
from src.training.frontend import MelFrontend, build_gpu_logmel

NUM_SAMPLES = 96000


def time_it(fn, warmup=10, runs=100):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return {"median_ms": statistics.median(times), "p95_ms": float(np.percentile(times, 95)), "mean_ms": float(np.mean(times))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", default="F_M0")
    parser.add_argument("--config", default="configs/final/gd_base.yaml")
    parser.add_argument("--n-int8", type=int, default=800)
    parser.add_argument("--out", default="results/tables/final")
    args = parser.parse_args()

    root = find_project_root()
    config = yaml.safe_load((root / args.config).read_text())
    device = torch.device("cpu")
    ckpt = torch.load(root / "experiments" / args.id / "best.pt", map_location=device, weights_only=False)
    model, mode = build_model_from_checkpoint(ckpt, device)
    frontend = MelFrontend(build_gpu_logmel(config, device), ckpt["mel_mean"], ckpt["mel_std"], device)

    result = {"id": args.id, "torch": torch.__version__, "threads_default": torch.get_num_threads()}
    result["parameters"] = int(sum(p.numel() for p in model.parameters()))

    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    result["weights_only_fp32_mb"] = buf.getbuffer().nbytes / 1e6
    result["full_checkpoint_mb"] = (root / "experiments" / args.id / "best.pt").stat().st_size / 1e6

    x = torch.randn(1, 1, 80, 401)
    infer = (lambda: model(x)) if mode == "M0" else (lambda: model(x)[0])
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        infer()
    result["macs_per_clip"] = fc.get_total_flops() / 2

    wave = torch.randn(1, NUM_SAMPLES) * 0.1
    model.eval()
    with torch.no_grad():
        for threads in (1, 4):
            torch.set_num_threads(threads)
            result[f"latency_threads{threads}"] = {
                "model_only": time_it(infer),
                "frontend_plus_model": time_it(lambda: model(frontend(wave)) if mode == "M0" else model(frontend(wave))[0]),
                "frontend_only": time_it(lambda: frontend(wave)),
            }
    result["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # bytes on macOS

    # ONNX export and INT8 static quantisation (QDQ) with onnxruntime.
    try:
        import onnxruntime as ort
        from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

        class Wrapper(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, inp):
                out = self.m(inp)
                return out if mode == "M0" else out[0]

        work = root / args.out / "onnx"
        work.mkdir(parents=True, exist_ok=True)
        fp32_path, int8_path = work / "model_fp32.onnx", work / "model_int8.onnx"
        torch.onnx.export(Wrapper(model).eval(), x, str(fp32_path), input_names=["mel"], output_names=["logits"], opset_version=17, dynamo=False)

        df = load_manifest_df(root, config["dataset"]["eval_manifest"], "eval")
        df = df.iloc[stratified_indices(df, args.n_int8, seed=3)].reset_index(drop=True)
        feats, labels = [], []
        for _, row in df.iterrows():
            w = load_audio(audio_path_from_row(row, root, config["dataset"]["audio_root"]), 24000)
            w = w[(len(w) - NUM_SAMPLES) // 2 :][:NUM_SAMPLES] if len(w) >= NUM_SAMPLES else repeat_pad_audio(w, NUM_SAMPLES)
            feats.append(frontend(torch.from_numpy(w[None]).float()).numpy())
            labels.append(int(row["label"] == "fake"))

        class Reader(CalibrationDataReader):
            def __init__(self, data):
                self.it = iter([{"mel": d} for d in data])

            def get_next(self):
                return next(self.it, None)

        quantize_static(str(fp32_path), str(int8_path), Reader(feats[:200]), quant_format=QuantFormat.QDQ, activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)

        def run(path):
            sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            scores = []
            for f in feats:
                lg = sess.run(None, {"mel": f})[0][0]
                e = np.exp(lg - lg.max())
                scores.append(float(e[1] / e.sum()))
            return sess, np.asarray(scores)

        s32, sc32 = run(fp32_path)
        s8, sc8 = run(int8_path)
        sess_in = {"mel": feats[0]}
        result["onnx"] = {
            "fp32_mb": fp32_path.stat().st_size / 1e6,
            "int8_mb": int8_path.stat().st_size / 1e6,
            "fp32_latency": time_it(lambda: s32.run(None, sess_in)),
            "int8_latency": time_it(lambda: s8.run(None, sess_in)),
            "fp32_metrics": calculate_metrics(labels, sc32)[0],
            "int8_metrics": calculate_metrics(labels, sc8)[0],
            "n_eval": len(labels),
        }
    except Exception as exc:  # recorded rather than hidden
        result["onnx_error"] = repr(exc)

    out = root / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / f"edge_{args.id}.json").write_text(json.dumps(result, indent=2, default=float))
    print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
