import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from src.datasets.atadd_manifest_dataset import find_project_root, load_manifest_df
from src.evaluation.metrics import calculate_metrics
from src.evaluation.plotting import save_delta_m_heatmap
from src.evaluation.eval_runner import build_model_from_checkpoint
from src.preprocessing.preprocess_atadd_v4 import load_audio, repeat_pad_audio
from src.robustness.conditions import all_training_conditions
from src.robustness.degrade import degrade, ffmpeg_available
from src.training.frontend import MelFrontend, build_gpu_logmel


def evaluate_waveforms(model, frontend, waveforms, labels, device, mode):
    amp_enabled = device.type == "cuda"
    all_scores = []

    with torch.no_grad():
        for start in range(0, len(waveforms), 32):
            batch_w = torch.tensor(np.stack(waveforms[start : start + 32]), device=device)
            batch_y = labels[start : start + 32]
            x = frontend(batch_w)

            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                if mode == "M0":
                    logits = model(x)
                else:
                    logits, _, _ = model(x)

            scores = torch.softmax(logits.float(), dim=1)[:, 1]
            all_scores.extend(scores.cpu().numpy().tolist())

    metrics, _ = calculate_metrics(labels, all_scores)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--splits", default="eval")
    parser.add_argument("--out", default="results/tables/E7_delta_m_atadd.csv")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    project_root = find_project_root()

    with (project_root / args.config).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        checkpoint = torch.load(project_root / args.checkpoint, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(project_root / args.checkpoint, map_location=device)
    model, mode = build_model_from_checkpoint(checkpoint, device)

    mel_mean = checkpoint.get("mel_mean") or config.get("mel", {}).get("mean")
    mel_std = checkpoint.get("mel_std") or config.get("mel", {}).get("std")

    if mel_mean is None:
        stats_path = project_root / config.get(
            "logmel_stats_path",
            "experiments/baselines/E1_v4/E1_ATADD_logmel_stats_v4.json",
        )
        with stats_path.open("r", encoding="utf-8") as f:
            stats = json.load(f)
        mel_mean, mel_std = stats["mean"], stats["std"]

    frontend = MelFrontend(build_gpu_logmel(config, device), mel_mean, mel_std, device)

    sample_rate = config["audio"]["sample_rate"]
    num_samples = sample_rate * config["audio"]["segment_seconds"]

    rows = []

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        manifest_rel = config["dataset"][f"{split}_manifest"]
        df = load_manifest_df(project_root, manifest_rel, split, max_samples=args.max_samples)

        clean_waveforms = []
        labels = []
        types = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"load_{split}"):
            from src.datasets.atadd_manifest_dataset import audio_path_from_row

            path = audio_path_from_row(row, project_root)

            if not path.exists():
                continue

            waveform = load_audio(path, sample_rate)

            if len(waveform) >= num_samples:
                start = (len(waveform) - num_samples) // 2
                waveform = waveform[start : start + num_samples]
            else:
                waveform = repeat_pad_audio(waveform, num_samples)

            clean_waveforms.append(waveform.astype(np.float32))
            labels.append({"real": 0, "fake": 1}[str(row["label"])])
            types.append(str(row.get("type", "unknown")))

        clean_metrics = evaluate_waveforms(model, frontend, clean_waveforms, labels, device, mode)

        for condition in all_training_conditions():
            if condition.family == "identity":
                continue

            if condition.family == "mp3_codec" and not ffmpeg_available():
                continue

            degraded = []

            for i, waveform in enumerate(clean_waveforms):
                deg, _ = degrade(
                    waveform,
                    sample_rate,
                    condition,
                    mode="deterministic",
                    source_id=f"sample_{i}",
                )
                degraded.append(deg)

            cond_metrics = evaluate_waveforms(model, frontend, degraded, labels, device, mode)

            for audio_type in sorted(set(types)):
                idx = [i for i, t in enumerate(types) if t == audio_type]

                if not idx:
                    continue

                sub_clean = [clean_waveforms[i] for i in idx]
                sub_labels = [labels[i] for i in idx]
                sub_deg = [degraded[i] for i in idx]

                m_clean = evaluate_waveforms(model, frontend, sub_clean, sub_labels, device, mode)
                m_deg = evaluate_waveforms(model, frontend, sub_deg, sub_labels, device, mode)

                rows.append(
                    {
                        "split": split,
                        "condition_family": condition.family,
                        "condition_name": condition.name,
                        "severity": condition.severity,
                        "type": audio_type,
                        "macro_f1_clean": m_clean["macro_f1"],
                        "macro_f1_conditioned": m_deg["macro_f1"],
                        "delta_macro_f1": m_clean["macro_f1"] - m_deg["macro_f1"],
                        "auroc_clean": m_clean["auroc"],
                        "auroc_conditioned": m_deg["auroc"],
                        "delta_auroc": m_clean["auroc"] - m_deg["auroc"],
                        "eer_clean": m_clean["eer"],
                        "eer_conditioned": m_deg["eer"],
                        "delta_eer": m_clean["eer"] - m_deg["eer"],
                        "overall_clean_macro_f1": clean_metrics["macro_f1"],
                        "overall_cond_macro_f1": cond_metrics["macro_f1"],
                    }
                )

    out_path = project_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)

    figure_dir = project_root / "results" / "figures" / Path(args.checkpoint).parent.name
    save_delta_m_heatmap(out_df, figure_dir / "delta_m_heatmap_type_x_condition.png")

    print(f"Wrote {out_path} ({len(out_df)} rows)")


if __name__ == "__main__":
    main()
