import json
import math
from pathlib import Path

import torch

from src.preprocessing.preprocess_atadd_v4 import GPULogMel, compute_logmel_stats_gpu, normalize_log_mel


def build_gpu_logmel(config, device):
    spec = config["spectrogram"]
    return GPULogMel(
        sample_rate=config["audio"]["sample_rate"],
        n_fft=spec["n_fft"],
        win_length=spec["win_length"],
        hop_length=spec["hop_length"],
        n_mels=spec["n_mels"],
        log_eps=spec.get("log_epsilon", 1e-10),
    ).to(device)


def load_or_compute_mel_stats(config, train_loader, gpu_logmel, device, stats_path, max_batches=None):
    stats_path = Path(stats_path)
    version = config.get("input_preprocessing_version", "v4.0_24khz_repeat_pad")

    if stats_path.exists():
        with stats_path.open("r", encoding="utf-8") as f:
            stats = json.load(f)

        compatible = (
            stats.get("version") == version
            and stats.get("sample_rate") == config["audio"]["sample_rate"]
            and stats.get("n_fft") == config["spectrogram"]["n_fft"]
        )

        if compatible:
            return float(stats["mean"]), float(stats["std"]), stats

    batches = max_batches

    if max_batches is not None:
        batches = max(1, math.ceil(max_batches / config["training"]["batch_size"]))

    mean, std = compute_logmel_stats_gpu(train_loader, gpu_logmel, device, batches)

    stats = {
        "version": version,
        "mean": mean,
        "std": std,
        "seed": config["training"]["seed"],
        "train_samples": len(train_loader.dataset),
        "sample_rate": config["audio"]["sample_rate"],
        "segment_seconds": config["audio"]["segment_seconds"],
        "n_fft": config["spectrogram"]["n_fft"],
        "win_length": config["spectrogram"]["win_length"],
        "hop_length": config["spectrogram"]["hop_length"],
        "n_mels": config["spectrogram"]["n_mels"],
        "feature_device": "cuda" if device.type == "cuda" else "cpu",
    }

    stats_path.parent.mkdir(parents=True, exist_ok=True)

    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    return mean, std, stats


class MelFrontend:
    def __init__(self, gpu_logmel, mel_mean, mel_std, device):
        self.gpu_logmel = gpu_logmel
        self.mel_mean = mel_mean
        self.mel_std = mel_std
        self.device = device

    def __call__(self, waveforms):
        waveforms = waveforms.to(self.device, non_blocking=True)
        log_mel = self.gpu_logmel(waveforms)
        log_mel = normalize_log_mel(log_mel, self.mel_mean, self.mel_std)
        return log_mel.unsqueeze(1)
