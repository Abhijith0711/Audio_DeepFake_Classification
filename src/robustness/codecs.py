import shutil
import subprocess
from pathlib import Path
from typing import Optional, Sequence


CODEC_SETTINGS = {
    "mp3": {
        "64": ["-codec:a", "libmp3lame", "-b:a", "64k"],
        "128": ["-codec:a", "libmp3lame", "-b:a", "128k"],
        "192": ["-codec:a", "libmp3lame", "-b:a", "192k"],
    },
    "aac": {
        "32": ["-codec:a", "aac", "-b:a", "32k"],
        "48": ["-codec:a", "aac", "-b:a", "48k"],
        "64": ["-codec:a", "aac", "-b:a", "64k"],
    },
    "opus": {
        "32": ["-codec:a", "libopus", "-b:a", "32k"],
        "64": ["-codec:a", "libopus", "-b:a", "64k"],
        "128": ["-codec:a", "libopus", "-b:a", "128k"],
    },
}

_CODEC_EXTENSIONS = {"mp3": ".mp3", "aac": ".m4a", "opus": ".ogg"}
_RESAMPLE_RATES = (16000, 12000, 8000)
_NOISE_SNRS = (20, 10, 5)


def compress_audio(
    input_path,
    output_path,
    codec,
    bitrate,
    ffmpeg_path=None,
):
    input_path = Path(input_path)
    output_path = Path(output_path)

    codec = str(codec).lower()
    bitrate = str(bitrate)

    if codec not in CODEC_SETTINGS:
        raise ValueError(
            f"Unsupported codec: {codec}. "
            f"Supported codecs: {list(CODEC_SETTINGS)}"
        )

    if bitrate not in CODEC_SETTINGS[codec]:
        raise ValueError(
            f"Unsupported bitrate {bitrate} for {codec}. "
            f"Supported bitrates: {list(CODEC_SETTINGS[codec])}"
        )

    if not input_path.is_file():
        raise FileNotFoundError(f"Input audio not found: {input_path}")

    ffmpeg = str(ffmpeg_path) if ffmpeg_path is not None else shutil.which("ffmpeg")
    if not ffmpeg or not Path(ffmpeg).is_file():
        raise FileNotFoundError(
            "FFmpeg executable not found. Set ffmpeg_path "
            "or add FFmpeg to PATH."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-map",
        "0:a:0",
        "-vn",
        *CODEC_SETTINGS[codec][bitrate],
    ]

    if codec == "mp3":
        command += ["-f", "mp3"]
    elif codec == "aac":
        command += ["-f", "ipod"]
    else:
        command += ["-f", "ogg"]

    command.append(str(output_path))
    result = subprocess.run(command, capture_output=True, text=True, check=False)

    if result.returncode != 0:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"FFmpeg failed for {codec} at {bitrate} kbps.\n"
            f"Command: {' '.join(command)}\n"
            f"Error: {result.stderr}"
        )

    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"FFmpeg did not produce a valid output: {output_path}")

    return output_path


def available_conditions():
    """Return condition-name to (family, parameters, output extension) mappings."""
    conditions = {}
    for codec, bitrates in CODEC_SETTINGS.items():
        for bitrate in bitrates:
            name = f"{codec}_{bitrate}k"
            conditions[name] = (
                f"{codec}_codec",
                {"codec": codec, "bitrate_kbps": int(bitrate)},
                _CODEC_EXTENSIONS[codec],
            )

    for rate in _RESAMPLE_RATES:
        conditions[f"resample_{rate // 1000}k"] = (
            "resample",
            {"intermediate_sr": rate},
            ".wav",
        )

    for snr_db in _NOISE_SNRS:
        conditions[f"snr_{snr_db}db"] = (
            "additive_noise",
            {"snr_db": snr_db},
            ".wav",
        )

    conditions["telephony_300_3400"] = (
        "telephony",
        {"low_hz": 300, "high_hz": 3400},
        ".wav",
    )
    return conditions


def _resolve_source_path(row, manifest_dir, audio_root):
    import pandas as pd

    row_audio_path = row.get("audio_path")
    if row_audio_path is not None and pd.notna(row_audio_path) and str(row_audio_path):
        path = Path(str(row_audio_path))
        if not path.is_absolute():
            path = (audio_root or manifest_dir) / path
        if path.is_file():
            return path.resolve()
        raise FileNotFoundError(f"Source audio not found: {path}")

    name = str(row["name"])
    row_folder = row.get("folder", "")
    folder = (
        str(row_folder)
        if row_folder is not None and pd.notna(row_folder) and str(row_folder)
        else ""
    )
    candidates = []
    if audio_root is not None:
        if folder:
            candidates.append(audio_root / folder / name)
        candidates.append(audio_root / name)
    candidates.append(manifest_dir / name)

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Could not resolve source audio {name!r}; checked: "
        + ", ".join(str(path) for path in candidates)
    )


def _write_waveform(output_path, waveform, sample_rate):
    import soundfile as sf

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f"{output_path.stem}.tmp.wav")
    sf.write(temporary_path, waveform, sample_rate, subtype="PCM_16")
    temporary_path.replace(output_path)

    info = sf.info(str(output_path))
    if info.samplerate != sample_rate or info.frames == 0:
        raise RuntimeError(f"Invalid generated audio file: {output_path}")


def generate_robustness_dataset(
    manifest_path,
    output_root,
    audio_root=None,
    conditions: Optional[Sequence[str]] = None,
    target_sample_rate=24000,
    seed=42,
    include_clean=True,
    ffmpeg_path=None,
):
    """Generate independent robustness variants and a training-ready CSV manifest.

    Each variant is derived from its original source. Rows retain the input
    labels/splits and add an ``audio_path`` column understood by the datasets.
    """
    import librosa
    import numpy as np
    import pandas as pd
    import soundfile as sf

    from src.robustness.conditions import ConditionSpec
    from src.robustness.degrade import degrade

    manifest_path = Path(manifest_path).resolve()
    output_root = Path(output_root).resolve()
    audio_root_path = Path(audio_root).resolve() if audio_root is not None else None
    if target_sample_rate <= 0:
        raise ValueError("target_sample_rate must be greater than zero.")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    source_df = pd.read_csv(manifest_path)
    missing_columns = {"name", "label", "split"} - set(source_df.columns)
    if missing_columns:
        raise ValueError(f"Manifest is missing columns: {sorted(missing_columns)}")

    condition_specs = available_conditions()
    if conditions is None:
        selected_conditions = list(condition_specs)
    else:
        selected_conditions = list(conditions)
        unknown = set(selected_conditions) - set(condition_specs)
        if unknown:
            raise ValueError(f"Unknown conditions: {sorted(unknown)}")
        if len(selected_conditions) != len(set(selected_conditions)):
            raise ValueError("Conditions must not contain duplicates.")
    if "telephony_300_3400" in selected_conditions and target_sample_rate <= 6800:
        raise ValueError(
            "target_sample_rate must exceed 6800 Hz for the 300-3400 Hz telephony filter."
        )

    records = []
    for row_index, row in enumerate(source_df.to_dict(orient="records")):
        source_path = _resolve_source_path(row, manifest_path.parent, audio_root_path)
        split = str(row["split"])
        if Path(split).name != split or split in ("", ".", ".."):
            raise ValueError(f"Invalid split directory in manifest: {split!r}")

        row_source_id = row.get("file_id")
        source_id = (
            str(row_source_id)
            if row_source_id is not None and pd.notna(row_source_id)
            else Path(str(row["name"])).stem
        )
        base_record = {
            **row,
            "source_id": source_id,
            "source_name": str(row["name"]),
            "source_audio_path": str(source_path),
        }

        if include_clean:
            records.append({
                **base_record,
                "audio_path": str(source_path),
                "condition_type": "clean",
                "condition_family": "identity",
                "condition_name": "clean",
                "codec": "none",
                "bitrate_kbps": "",
                "intermediate_sr": "",
                "snr_db": "",
            })

        for condition_name in selected_conditions:
            family, params, extension = condition_specs[condition_name]
            output_path = (
                output_root
                / condition_name
                / split
                / f"{row_index:08d}_{Path(str(row['name'])).stem}{extension}"
            )

            if family.endswith("_codec"):
                codec = params["codec"]
                compress_audio(
                    source_path,
                    output_path,
                    codec,
                    params["bitrate_kbps"],
                    ffmpeg_path=ffmpeg_path,
                )
            else:
                audio, original_sr = sf.read(
                    str(source_path), dtype="float32", always_2d=True
                )
                waveform = audio.mean(axis=1)
                waveform = np.nan_to_num(
                    waveform, nan=0.0, posinf=0.0, neginf=0.0
                )
                if waveform.size == 0:
                    raise ValueError(f"Empty source audio: {source_path}")
                if original_sr != target_sample_rate:
                    waveform = librosa.resample(
                        waveform,
                        orig_sr=original_sr,
                        target_sr=target_sample_rate,
                        res_type="soxr_hq",
                    )

                condition = ConditionSpec(
                    family=family,
                    name=condition_name,
                    severity=1,
                    params=params,
                    seed=seed,
                )
                waveform, _ = degrade(
                    waveform,
                    target_sample_rate,
                    condition,
                    mode="deterministic",
                    source_id=source_id,
                )
                _write_waveform(output_path, waveform, target_sample_rate)

            records.append({
                **base_record,
                "name": output_path.name,
                "file_id": f"{source_id}__{condition_name}",
                "audio_path": str(output_path),
                "condition_type": family,
                "condition_family": family,
                "condition_name": condition_name,
                "codec": params.get("codec", ""),
                "bitrate_kbps": params.get("bitrate_kbps", ""),
                "intermediate_sr": params.get("intermediate_sr", ""),
                "snr_db": params.get("snr_db", ""),
                "target_sample_rate": (
                    "" if family.endswith("_codec") else target_sample_rate
                ),
            })

    output_root.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(records)
    manifest_out = output_root / "robustness_manifest.csv"
    result.to_csv(manifest_out, index=False)
    return result
