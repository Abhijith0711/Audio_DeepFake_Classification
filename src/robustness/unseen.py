"""Evaluation-only degradations that are not part of the training catalogue.

These test generalisation to codecs and channels never seen during training:
AAC, Opus, G.711 mu-law telephony, synthetic room reverberation and packet loss.
"""
import hashlib
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from src.robustness.degrade import _FFMPEG_AVAILABLE

SAMPLE_RATE = 24000

FFMPEG_CODECS = {
    "aac_32k": (["-c:a", "aac", "-b:a", "32k"], "m4a"),
    "aac_64k": (["-c:a", "aac", "-b:a", "64k"], "m4a"),
    "opus_16k": (["-c:a", "libopus", "-b:a", "16k"], "ogg"),
    "opus_32k": (["-c:a", "libopus", "-b:a", "32k"], "ogg"),
    "mulaw_8k": (["-ar", "8000", "-c:a", "pcm_mulaw"], "wav"),
}

UNSEEN_CONDITIONS = list(FFMPEG_CODECS) + ["reverb_rt60_0.4s", "reverb_rt60_0.8s", "packet_loss_10pct"]


def _rng(name, source_id):
    key = f"{name}|{source_id}".encode("utf-8")
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:4], "little"))


def _ffmpeg_roundtrip(waveform, sample_rate, codec_args, extension):
    if not _FFMPEG_AVAILABLE:
        raise RuntimeError("ffmpeg is required for codec conditions but was not found on PATH.")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src, enc, dec = tmp / "in.wav", tmp / f"enc.{extension}", tmp / "dec.wav"
        sf.write(src, waveform, sample_rate)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), *codec_args, str(enc)], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(enc), "-ar", str(sample_rate), str(dec)], check=True)
        decoded, _ = sf.read(dec, dtype="float32")

    if decoded.ndim > 1:
        decoded = decoded.mean(axis=1)

    if len(decoded) < len(waveform):
        decoded = np.pad(decoded, (0, len(waveform) - len(decoded)))
    return decoded[: len(waveform)].astype(np.float32)


def _reverb(waveform, sample_rate, rt60, rng):
    length = int(rt60 * sample_rate)
    t = np.arange(length) / sample_rate
    rir = rng.standard_normal(length) * np.exp(-6.9078 * t / rt60)  # -60 dB at t = rt60
    rir[0] = 1.0
    out = signal.fftconvolve(waveform, rir / np.linalg.norm(rir), mode="full")[: len(waveform)]
    scale = np.sqrt(np.mean(waveform**2) + 1e-12) / np.sqrt(np.mean(out**2) + 1e-12)
    return (out * scale).astype(np.float32)


def _packet_loss(waveform, sample_rate, loss_rate, rng, frame_ms=20):
    frame = int(sample_rate * frame_ms / 1000)
    out = waveform.copy()

    for start in range(0, len(out) - frame + 1, frame):
        if rng.random() < loss_rate:
            out[start : start + frame] = 0.0

    return out


def apply_unseen(waveform, name, source_id, sample_rate=SAMPLE_RATE):
    waveform = np.asarray(waveform, dtype=np.float32)

    if name in FFMPEG_CODECS:
        args, ext = FFMPEG_CODECS[name]
        return _ffmpeg_roundtrip(waveform, sample_rate, args, ext)

    if name.startswith("reverb_rt60_"):
        return _reverb(waveform, sample_rate, float(name.split("_")[-1].rstrip("s")), _rng(name, source_id))

    if name == "packet_loss_10pct":
        return _packet_loss(waveform, sample_rate, 0.10, _rng(name, source_id))

    raise ValueError(f"Unknown unseen condition: {name}")
