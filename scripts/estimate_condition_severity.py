"""Rank training degradations by the drop in a clean model's development AUROC.

This does not change the ordinal severity labels used in training. It writes a
table of clean AUROC minus per-condition AUROC so those labels can be compared
with a measurement.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.atadd_manifest_dataset import (
    ATADDManifestDataset,
    audio_root_from_config,
    find_project_root,
    load_manifest_df,
)
from src.evaluation.eval_runner import build_model_from_checkpoint
from src.evaluation.metrics import calculate_metrics
from src.robustness.conditions import all_training_conditions
from src.robustness.degrade import degrade
from src.training.frontend import MelFrontend, build_gpu_logmel
from src.training.selection import match_length, select_degraded_dev_indices
from src.training.train import load_yaml_config


def _load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _score(model, mode, frontend, waves, device, batch_size):
    scores = []
    amp = device.type == "cuda"
    for start in range(0, len(waves), batch_size):
        batch = torch.stack(waves[start:start + batch_size]).to(device)
        spec = frontend(batch)
        with torch.amp.autocast(device_type=device.type, enabled=amp):
            if mode == "M0":
                logits = model(spec)
            else:
                logits, _, _ = model(spec)
        scores.extend(torch.softmax(logits.float(), dim=1)[:, 1].detach().cpu().tolist())
    return np.asarray(scores, dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/proposed/e7_ft_rcl_atadd_v1.yaml")
    parser.add_argument("--checkpoint", required=True, help="Clean-trained checkpoint, usually an M0 best.pt")
    parser.add_argument("--split", default="dev", choices=["dev", "eval"])
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Stratified subset size. Omit to score the full split.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/tables/condition_severity_auroc_drop.csv")
    args = parser.parse_args()

    root = find_project_root()
    config = load_yaml_config(root / args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = _load_checkpoint(root / args.checkpoint, device)
    model, mode = build_model_from_checkpoint(checkpoint, device)

    manifest_key = f"{args.split}_manifest"
    df = load_manifest_df(root, config["dataset"][manifest_key], args.split)
    if args.max_samples is not None:
        df = df.iloc[select_degraded_dev_indices(df, args.max_samples, args.seed)].reset_index(drop=True)

    dataset = ATADDManifestDataset(
        df,
        root,
        sample_rate=config["audio"]["sample_rate"],
        segment_seconds=config["audio"]["segment_seconds"],
        training=False,
        audio_root_name=audio_root_from_config(config),
    )

    clean_waves = []
    labels = []
    for index in range(len(dataset)):
        wave, label, _meta = dataset[index]
        clean_waves.append(wave if torch.is_tensor(wave) else torch.from_numpy(np.asarray(wave, dtype=np.float32)))
        labels.append(int(label))
    labels = np.asarray(labels)

    mel_mean = checkpoint.get("mel_mean") or config.get("mel", {}).get("mean")
    mel_std = checkpoint.get("mel_std") or config.get("mel", {}).get("std")
    frontend = MelFrontend(build_gpu_logmel(config, device), mel_mean, mel_std, device)
    batch_size = int(config["training"]["batch_size"])

    clean_scores = _score(model, mode, frontend, clean_waves, device, batch_size)
    clean_auroc = calculate_metrics(labels, clean_scores)[0]["auroc"]

    rows = [{
        "condition_family": "identity",
        "condition_name": "clean",
        "assigned_severity": 0,
        "auroc": clean_auroc,
        "auroc_drop": 0.0,
        "n": int(len(labels)),
    }]

    sample_rate = config["audio"]["sample_rate"]
    for condition in all_training_conditions():
        if condition.family == "identity":
            continue
        degraded = []
        for index, wave in enumerate(clean_waves):
            wave_np = wave.numpy()
            out, _meta = degrade(
                wave_np,
                sample_rate,
                condition,
                mode="deterministic",
                source_id=str(dataset.df.iloc[index].get("file_id", index)),
            )
            degraded.append(torch.from_numpy(match_length(out, len(wave_np))))
        scores = _score(model, mode, frontend, degraded, device, batch_size)
        auroc = calculate_metrics(labels, scores)[0]["auroc"]
        rows.append({
            "condition_family": condition.family,
            "condition_name": condition.name,
            "assigned_severity": int(condition.severity),
            "auroc": auroc,
            "auroc_drop": float(clean_auroc - auroc),
            "n": int(len(labels)),
        })

    table = pd.DataFrame(rows).sort_values("auroc_drop", ascending=False)
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    print(table.to_string(index=False))
    print(output)
    print("These drops are not written back into the training severity labels.")


if __name__ == "__main__":
    main()
