import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.datasets.atadd_manifest_dataset import audio_root_from_config, load_manifest_df, make_dataloader
from src.evaluation.metrics import calculate_metrics
from src.evaluation.plotting import save_confusion, save_roc
from src.models.ft_rcl_detector import FTRCLDetector
from src.models.mobilenet_v3_audio import MobileNetV3SmallAudio
from src.training.frontend import MelFrontend, build_gpu_logmel


def build_model_from_checkpoint(checkpoint, device):
    config = checkpoint.get("config", {})
    mode = checkpoint.get("mode", "M0")
    star = config.get("star_edge", {})

    if mode == "M0":
        model = MobileNetV3SmallAudio(num_classes=2)
    else:
        model = FTRCLDetector(num_classes=2, use_cad=bool(star.get("use_cad", False)))

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, mode


def run_split_evaluation(
    config,
    project_root,
    split,
    checkpoint_path,
    experiment_dir,
    figure_dir,
    device,
    max_samples=None,
    sample_fraction=None,
    sample_seed=42,
):
    manifest_key = f"{split}_manifest"
    manifest_rel = config["dataset"].get(f"{split}_manifest") or config["dataset"].get(manifest_key)

    if split == "dev":
        manifest_rel = config["dataset"]["dev_manifest"]
    elif split == "eval":
        manifest_rel = config["dataset"]["eval_manifest"]
    elif split == "train":
        manifest_rel = config["dataset"]["train_manifest"]

    df = load_manifest_df(
        project_root,
        manifest_rel,
        split,
        max_samples=max_samples,
        sample_fraction=sample_fraction,
        sample_seed=sample_seed,
    )

    loader = make_dataloader(
        df,
        project_root,
        batch_size=config["training"]["batch_size"],
        training=False,
        seed=config["training"]["seed"],
        num_workers=config["training"].get("num_workers", 4),
        prefetch_factor=config["training"].get("prefetch_factor", 2),
        pin_memory=config["training"].get("pin_memory", device.type == "cuda"),
        sample_rate=config["audio"]["sample_rate"],
        segment_seconds=config["audio"]["segment_seconds"],
        audio_root_name=audio_root_from_config(config),
    )

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model, mode = build_model_from_checkpoint(checkpoint, device)

    mel_mean = checkpoint.get("mel_mean") or config.get("mel", {}).get("mean")
    mel_std = checkpoint.get("mel_std") or config.get("mel", {}).get("std")

    if mel_mean is None or mel_std is None:
        stats_path = config.get("logmel_stats_path")
        if stats_path:
            with Path(project_root / stats_path).open("r", encoding="utf-8") as f:
                stats = json.load(f)
            mel_mean = stats["mean"]
            mel_std = stats["std"]

    gpu_logmel = build_gpu_logmel(config, device)
    frontend = MelFrontend(gpu_logmel, mel_mean, mel_std, device)

    amp_enabled = device.type == "cuda"

    all_labels = []
    all_scores = []
    rows = []

    with torch.no_grad():
        for waveforms, labels, metas in tqdm(loader, desc=f"eval_{split}"):
            waveforms = waveforms.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            x = frontend(waveforms)

            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                if mode == "M0":
                    logits = model(x)
                else:
                    logits, _, _ = model(x)

            scores = torch.softmax(logits.float(), dim=1)[:, 1]
            preds = (scores >= 0.5).int()

            for i, meta in enumerate(metas):
                rows.append(
                    {
                        "filepath": meta.get("file_id"),
                        "label": int(labels[i].item()),
                        "score": float(scores[i].item()),
                        "pred": int(preds[i].item()),
                        "type": meta.get("type", "unknown"),
                        "generator": meta.get("generator", "-"),
                    }
                )

            all_labels.extend(labels.cpu().numpy().tolist())
            all_scores.extend(scores.cpu().numpy().tolist())

    metrics, predictions = calculate_metrics(all_labels, all_scores)

    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = experiment_dir / f"eval_metrics_{split}.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    pred_df = pd.DataFrame(rows)
    pred_path = experiment_dir / f"eval_predictions_{split}.csv"
    pred_df.to_csv(pred_path, index=False)

    if "type" in pred_df.columns:
        type_rows = []

        for audio_type, group in pred_df.groupby("type"):
            if len(np.unique(group["label"])) < 2:
                continue

            m, _ = calculate_metrics(group["label"].values, group["score"].values)
            type_rows.append({"type": audio_type, **m, "samples": len(group)})

        type_df = pd.DataFrame(type_rows)
        type_path = experiment_dir / f"eval_by_type_{split}.csv"
        type_df.to_csv(type_path, index=False)

    gen_df = pred_df[(pred_df["label"] == 1) & (pred_df["generator"] != "-")]

    if len(gen_df):
        gen_rows = []

        for generator, group in gen_df.groupby("generator"):
            detected = int((group["pred"] == 1).sum())
            gen_rows.append(
                {
                    "generator": generator,
                    "samples": len(group),
                    "fake_detected": detected,
                    "fake_recall": detected / len(group),
                    "mean_fake_score": float(group["score"].mean()),
                }
            )

        gen_path = experiment_dir / f"eval_by_generator_{split}.csv"
        pd.DataFrame(gen_rows).to_csv(gen_path, index=False)

    failures = _failure_cases(pred_df)
    failure_path = experiment_dir / f"failure_cases_{split}.csv"
    failures.to_csv(failure_path, index=False)

    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    save_roc(all_labels, all_scores, figure_dir / f"roc_{split}.png")
    save_confusion(all_labels, predictions, figure_dir / f"confusion_matrix_{split}.png")

    return metrics, {
        "metrics": metrics_path,
        "predictions": pred_path,
        "failures": failure_path,
    }


def _failure_cases(pred_df, top_n=50):
    df = pred_df.copy()
    df["error"] = (df["label"] != df["pred"]).astype(int)
    fp = df[(df["label"] == 0) & (df["pred"] == 1)].sort_values("score", ascending=False).head(top_n)
    fn = df[(df["label"] == 1) & (df["pred"] == 0)].sort_values("score", ascending=True).head(top_n)
    fp["failure_type"] = "false_positive"
    fn["failure_type"] = "false_negative"
    return pd.concat([fp, fn], ignore_index=True)


def compare_to_e1_baseline(metrics, baseline_path):
    baseline_path = Path(baseline_path)

    if not baseline_path.exists():
        return None

    with baseline_path.open("r", encoding="utf-8") as f:
        baseline = json.load(f)

    rows = []

    for key in ["macro_f1", "accuracy", "auroc", "eer"]:
        cur = metrics.get(key)
        base = baseline.get(key)
        delta = cur - base if cur is not None and base is not None else None
        rows.append({"metric": key, "current": cur, "E1_v4": base, "delta": delta})

    return pd.DataFrame(rows)
