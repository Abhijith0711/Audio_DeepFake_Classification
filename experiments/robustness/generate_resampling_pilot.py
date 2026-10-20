from pathlib import Path
import hashlib

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

PROJECT_ROOT = Path(r"G:\BTP_Git")
SOURCE_ROOT = Path(r"G:\BTP\Data\robustness\codec_subset_dev500_v2")
OUTPUT_ROOT = Path(r"G:\BTP\Data\robustness\resampling_dev500_v1")
MANIFEST_PATH = SOURCE_ROOT / "codec_subset_metadata.csv"

TARGET_SR = 24000
INTERMEDIATE_RATES = [16000, 12000, 8000]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resample_audio(input_path, output_path, intermediate_sr):
    audio, original_sr = sf.read(input_path, dtype="float32",
                                 always_2d=True)
    audio = audio.mean(axis=1)

    if audio.size == 0:
        raise ValueError(f"Empty audio: {input_path}")

    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

    # Simulate lower-rate acquisition, then convert to the model rate.
    low_rate_audio = librosa.resample(
        audio,
        orig_sr=original_sr,
        target_sr=intermediate_sr,
        res_type="soxr_hq",
    )

    restored_audio = librosa.resample(
        low_rate_audio,
        orig_sr=intermediate_sr,
        target_sr=TARGET_SR,
        res_type="soxr_hq",
    )

    peak = float(np.max(np.abs(restored_audio))) if restored_audio.size else 0.0
    if peak > 1.0:
        restored_audio = restored_audio / peak

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, restored_audio, TARGET_SR, subtype="PCM_16")

    check_audio, check_sr = sf.read(output_path, dtype="float32")
    if check_sr != TARGET_SR or check_audio.size == 0:
        raise RuntimeError(f"Invalid output: {output_path}")

    return {
        "original_sr": int(original_sr),
        "intermediate_sr": int(intermediate_sr),
        "output_sr": int(check_sr),
        "duration_seconds": len(check_audio) / check_sr,
        "peak_amplitude": float(np.max(np.abs(check_audio))),
        "sha256": sha256_file(output_path),
    }


def main():
    df = pd.read_csv(MANIFEST_PATH)
    clean = df[df["condition_type"] == "clean"].copy()

    if clean["source_id"].duplicated().any():
        raise ValueError("Duplicate clean source IDs in manifest")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    records = []

    for _, row in tqdm(clean.iterrows(), total=len(clean),
                       desc="Resampling clean sources"):
        source_path = Path(row["audio_path"])
        if not source_path.is_file():
            raise FileNotFoundError(source_path)

        for rate in INTERMEDIATE_RATES:
            out_dir = OUTPUT_ROOT / f"sr_{rate}"
            out_path = out_dir / f'{row["source_id"]}_sr{rate}_to24k.wav'

            info = resample_audio(source_path, out_path, rate)

            records.append({
                "source_id": row["source_id"],
                "source_name": row["source_name"],
                "label": row["label"],
                "split": row["split"],
                "condition_type": "resampling",
                "intermediate_sr": rate,
                "target_sr": TARGET_SR,
                "audio_path": str(out_path),
                **info,
            })

    output_manifest = OUTPUT_ROOT / "resampling_metadata.csv"
    pd.DataFrame(records).to_csv(output_manifest, index=False)

    print(f"\nSources: {len(clean)}")
    print(f"Generated degraded files: {len(records)}")
    print(f"Manifest: {output_manifest}")
    print("\nRecords per sample rate:")
    print(pd.DataFrame(records).groupby("intermediate_sr").size())


if __name__ == "__main__":
    main()
