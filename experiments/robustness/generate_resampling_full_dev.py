from pathlib import Path
import shutil
import time

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

SOURCE_MANIFEST = Path(r"G:\BTP\Data\manifests\dev.csv")
OUTPUT_ROOT = Path(r"G:\BTP\Data\robustness\resampling_full_dev_v1")
TARGET_SR = 24000
RATES = [16000, 12000, 8000]
CHECKPOINT_EVERY = 100


def output_path(row, rate):
    # Use the manifest name, not source_id, to avoid accidental
    # overwriting if source_id is not unique.
    name = Path(str(row["name"])).stem
    return OUTPUT_ROOT / f"sr_{rate}" / f"{name}_sr{rate}_to24k.wav"


def generate_one(source_path, dest, rate):
    audio, original_sr = sf.read(
        str(source_path), dtype="float32", always_2d=True
    )
    audio = audio.mean(axis=1)
    audio = np.nan_to_num(
        audio, nan=0.0, posinf=0.0, neginf=0.0
    )
    if audio.size == 0:
        raise ValueError("Empty source audio")

    low_rate = librosa.resample(
        audio, orig_sr=original_sr, target_sr=rate,
        res_type="soxr_hq"
    )
    restored = librosa.resample(
        low_rate, orig_sr=rate, target_sr=TARGET_SR,
        res_type="soxr_hq"
    )
    restored = np.nan_to_num(
        restored, nan=0.0, posinf=0.0, neginf=0.0
    )

    peak = float(np.max(np.abs(restored))) if restored.size else 0.0
    if peak > 1.0:
        restored = restored / peak

    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_suffix(".tmp.wav")
    sf.write(temp, restored, TARGET_SR, subtype="PCM_16")
    # Replace atomically so an interrupted write isn't mistaken for complete.
    temp.replace(dest)

    info = sf.info(str(dest))
    if info.samplerate != TARGET_SR or info.frames == 0:
        raise RuntimeError(f"Invalid output audio: {dest}")

    return {
        "original_sr": int(original_sr),
        "intermediate_sr": int(rate),
        "target_sr": int(info.samplerate),
        "duration_seconds": float(info.duration),
        "output_bytes": int(dest.stat().st_size),
    }


def main():
    if not SOURCE_MANIFEST.is_file():
        raise FileNotFoundError(SOURCE_MANIFEST)

    df = pd.read_csv(SOURCE_MANIFEST)
    required = {"name", "audio_path", "label", "type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

    if df["name"].astype(str).duplicated().any():
        raise ValueError(
            "'name' is not unique in dev.csv. Stop rather than risk "
            "overwriting outputs; use a unique manifest key."
        )

    absent = ~df["audio_path"].map(lambda p: Path(str(p)).is_file())
    if absent.any():
        raise FileNotFoundError(
            f"{int(absent.sum())} source files are missing."
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_out = OUTPUT_ROOT / "resampling_metadata.csv"
    errors_out = OUTPUT_ROOT / "generation_errors.csv"

    # Estimate storage from the actual source durations before generating.
    print("Scanning source durations for storage estimate...")
    total_duration = 0.0
    for path in tqdm(df["audio_path"].astype(str), desc="Checking audio"):
        total_duration += sf.info(path).duration

    # Mono PCM16 at 24 kHz: approximately 48,000 bytes per second per file.
    estimate_bytes = total_duration * TARGET_SR * 2 * len(RATES)
    free_bytes = shutil.disk_usage(OUTPUT_ROOT.anchor or "G:\\").free

    print(f"\nSource recordings: {len(df):,}")
    print(f"Generated outputs planned: {len(df) * len(RATES):,}")
    print(f"Estimated WAV storage: {estimate_bytes / (1024**3):.2f} GiB")
    print(f"Free disk space: {free_bytes / (1024**3):.2f} GiB")

    if estimate_bytes > free_bytes * 0.80:
        raise RuntimeError(
            "Estimated output exceeds 80% of available free space. "
            "Do not generate yet; use a subset or another drive."
        )

    # Resume safely from the saved manifest and already-valid output files.
    records = []
    errors = []
    if manifest_out.exists():
        try:
            old = pd.read_csv(manifest_out)
            records = old.to_dict(orient="records")
        except Exception:
            raise RuntimeError(
                f"Cannot read existing manifest: {manifest_out}. "
                "Inspect it before restarting."
            )

    completed = {
        (str(r.get("name")), int(r.get("intermediate_sr")))
        for r in records
        if pd.notna(r.get("name")) and pd.notna(r.get("intermediate_sr"))
    }

    started = time.time()
    new_count = 0

    for row in tqdm(
        df.to_dict(orient="records"),
        total=len(df),
        desc="Generating resampling conditions",
    ):
        source_path = Path(str(row["audio_path"]))

        for rate in RATES:
            key = (str(row["name"]), rate)
            dest = output_path(row, rate)

            if key in completed and dest.is_file():
                continue

            try:
                # A valid existing file can be reused after an interrupted run.
                if dest.is_file():
                    info = sf.info(str(dest))
                    if info.samplerate == TARGET_SR and info.frames > 0:
                        details = {
                            "original_sr": int(sf.info(str(source_path)).samplerate),
                            "intermediate_sr": rate,
                            "target_sr": int(info.samplerate),
                            "duration_seconds": float(info.duration),
                            "output_bytes": int(dest.stat().st_size),
                        }
                    else:
                        dest.unlink()
                        details = generate_one(source_path, dest, rate)
                else:
                    details = generate_one(source_path, dest, rate)

                records.append({
                    **row,
                    "source_path": str(source_path),
                    "condition_type": "resampling",
                    "intermediate_sr": rate,
                    "audio_path": str(dest),
                    **details,
                })
                completed.add(key)
                new_count += 1

            except Exception as exc:
                errors.append({
                    "name": row["name"],
                    "audio_path": str(source_path),
                    "intermediate_sr": rate,
                    "error": repr(exc),
                })

            if new_count and new_count % CHECKPOINT_EVERY == 0:
                pd.DataFrame(records).to_csv(manifest_out, index=False)
                pd.DataFrame(errors).to_csv(errors_out, index=False)

    pd.DataFrame(records).to_csv(manifest_out, index=False)
    pd.DataFrame(errors).to_csv(errors_out, index=False)

    result = pd.DataFrame(records)
    print("\nGeneration finished.")
    print("Expected outputs:", len(df) * len(RATES))
    print("Manifest rows:", len(result))
    print("Missing output rows:", len(df) * len(RATES) - len(result))
    print("Generation errors:", len(errors))
    print("Elapsed hours:", round((time.time() - started) / 3600, 2))
    print("Manifest:", manifest_out)
    print("Errors:", errors_out)

    if not result.empty:
        print("\nRows by sample rate:")
        print(result.groupby("intermediate_sr").size().to_string())


if __name__ == "__main__":
    main()
