import hashlib
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Literal, Tuple

import librosa
import numpy as np
from scipy import signal

from src.robustness.conditions import PIPELINE_VERSION, ConditionSpec

logger = logging.getLogger(__name__)

_FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None


def _deterministic_seed(condition: ConditionSpec, source_id: str) -> int:
    """Per-clip, per-condition seed so noise differs between clips but is reproducible."""
    base = condition.seed if condition.seed is not None else 42
    key = f"{base}|{condition.condition_id}|{source_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "little")


def _metadata(source_id: str, condition: ConditionSpec, seed: int) -> dict:
    return {
        "source_id": source_id,
        "condition_family": condition.family,
        "condition_name": condition.name,
        "severity": condition.severity,
        "seed": seed,
        "pipeline_version": PIPELINE_VERSION,
    }


def _apply_resample_down_up(waveform, sample_rate, intermediate_sr):
    down = librosa.resample(waveform, orig_sr=sample_rate, target_sr=intermediate_sr)
    up = librosa.resample(down, orig_sr=intermediate_sr, target_sr=sample_rate)
    return up.astype(np.float32)


def _apply_additive_noise(waveform, snr_db, rng):
    power = np.mean(waveform ** 2) + 1e-12
    noise = rng.standard_normal(waveform.shape).astype(np.float32)
    noise_power = np.mean(noise ** 2) + 1e-12
    target_noise_power = power / (10 ** (snr_db / 10.0))
    noise = noise * np.sqrt(target_noise_power / noise_power)
    return (waveform + noise).astype(np.float32)


def _apply_telephony(waveform, sample_rate, low_hz, high_hz):
    nyq = sample_rate / 2.0
    low = max(low_hz / nyq, 1e-4)
    high = min(high_hz / nyq, 0.999)
    sos = signal.butter(4, [low, high], btype="bandpass", output="sos")
    return signal.sosfilt(sos, waveform).astype(np.float32)


def _apply_mp3_codec(waveform, sample_rate, bitrate_kbps):
    if not _FFMPEG_AVAILABLE:
        raise RuntimeError(
            "ffmpeg is required for the mp3_codec degradation but was not found on PATH"
        )

    import soundfile as sf

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        wav_in = tmp_path / "in.wav"
        mp3_path = tmp_path / "out.mp3"
        wav_out = tmp_path / "decoded.wav"

        sf.write(wav_in, waveform, sample_rate)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(wav_in),
                "-b:a",
                f"{bitrate_kbps}k",
                str(mp3_path),
            ],
            check=True,
        )
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(mp3_path),
                "-ar",
                str(sample_rate),
                str(wav_out),
            ],
            check=True,
        )

        decoded, sr = sf.read(wav_out, dtype="float32")

        if decoded.ndim > 1:
            decoded = decoded.mean(axis=1)

        if sr != sample_rate:
            decoded = librosa.resample(decoded, orig_sr=sr, target_sr=sample_rate)

        if len(decoded) < len(waveform):
            decoded = np.pad(decoded, (0, len(waveform) - len(decoded)))
        elif len(decoded) > len(waveform):
            decoded = decoded[: len(waveform)]

        return decoded.astype(np.float32)


def degrade(
    waveform: np.ndarray,
    sample_rate: int,
    condition: ConditionSpec,
    mode: Literal["deterministic", "random"] = "deterministic",
    source_id: str = "unknown",
) -> Tuple[np.ndarray, dict]:
    waveform = np.asarray(waveform, dtype=np.float32)

    if condition.family == "identity":
        seed = condition.seed if condition.seed is not None else 0
        return waveform.copy(), _metadata(source_id, condition, seed)

    if mode == "deterministic":
        seed = _deterministic_seed(condition, source_id)
    else:
        seed = int(np.random.randint(0, 2 ** 31 - 1))

    rng = np.random.default_rng(seed)
    out = waveform.copy()

    if condition.family == "resample":
        out = _apply_resample_down_up(out, sample_rate, condition.params["intermediate_sr"])
    elif condition.family == "mp3_codec":
        out = _apply_mp3_codec(out, sample_rate, condition.params["bitrate_kbps"])
    elif condition.family == "additive_noise":
        out = _apply_additive_noise(out, condition.params["snr_db"], rng)
    elif condition.family == "telephony":
        out = _apply_telephony(
            out,
            sample_rate,
            condition.params["low_hz"],
            condition.params["high_hz"],
        )
    else:
        raise ValueError(f"Unknown condition family: {condition.family}")

    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return out, _metadata(source_id, condition, seed)


def ffmpeg_available():
    return _FFMPEG_AVAILABLE
