
import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from robustness.codecs import compress_audio


FFMPEG_PATH = Path(r"G:\ffmpeg\bin\ffmpeg.exe")

CODEC_SETTINGS = {
    "mp3": ["64", "128", "192"],
    "aac": ["32", "48", "64"],
    "opus": ["32", "64", "128"],
}


def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def resolve_source_path(audio_root, filename):
    filename = str(filename)
    candidate = Path(filename)

    if candidate.is_absolute() and candidate.is_file():
        return candidate

    candidate = audio_root / filename

    if candidate.is_file():
        return candidate

    # If the manifest contains a path with subdirectories,
    # try resolving only its filename under the audio root.
    candidate = audio_root / Path(filename).name

    if candidate.is_file():
        return candidate

    return None


def generate_subset(
    manifest_path,
    audio_root,
    output_root,
    num_samples=500,
    seed=42,
):
    manifest_path = Path(manifest_path)
    audio_root = Path(audio_root)
    output_root = Path(output_root)

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}"
        )

    if not audio_root.is_dir():
        raise NotADirectoryError(
            f"Audio root not found: {audio_root}"
        )

    if not FFMPEG_PATH.is_file():
        raise FileNotFoundError(
            f"FFmpeg not found: {FFMPEG_PATH}"
        )

    df = pd.read_csv(manifest_path)

    required_columns = {"name", "label"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Manifest is missing columns: {sorted(missing_columns)}"
        )

    print("Manifest columns:")
    print(df.columns.tolist())

    if num_samples <= 0:
        raise ValueError("num_samples must be greater than zero.")

    if len(df) < num_samples:
        raise ValueError(
            f"Requested {num_samples} samples, "
            f"but manifest contains only {len(df)} rows."
        )

    # Select sources deterministically.
    selected = (
        df.sample(n=num_samples, random_state=seed)
        .reset_index(drop=True)
    )

    output_root.mkdir(parents=True, exist_ok=True)

    clean_dir = output_root / "clean"
    compressed_dir = output_root / "compressed"

    clean_dir.mkdir(parents=True, exist_ok=True)
    compressed_dir.mkdir(parents=True, exist_ok=True)

    records = []
    missing_sources = []

    for index, row in selected.iterrows():
        source_name = str(row["name"])
        source_path = resolve_source_path(audio_root, source_name)

        if source_path is None:
            missing_sources.append(source_name)
            continue

        source_id = source_path.stem
        label = row["label"]

        # Keep a copy of each clean source in the subset.
        clean_path = clean_dir / source_path.name

        if (
            not clean_path.exists()
            or sha256_file(clean_path) != sha256_file(source_path)
        ):
            clean_path.write_bytes(source_path.read_bytes())

        clean_record = {
            "source_id": source_id,
            "source_name": source_name,
            "label": label,
            "condition_type": "clean",
            "codec": "none",
            "bitrate_kbps": "none",
            "audio_path": str(clean_path.resolve()),
            "sha256": sha256_file(clean_path),
        }

        for column in df.columns:
            if column not in clean_record:
                clean_record[column] = row[column]

        records.append(clean_record)

        for codec, bitrates in CODEC_SETTINGS.items():
            for bitrate in bitrates:
                print(
                    f"[{index + 1}/{num_samples}] "
                    f"{codec.upper()} {bitrate} kbps -> {source_name}"
                )

                extension = {
                    "mp3": ".mp3",
                    "aac": ".m4a",
                    "opus": ".ogg",
                }[codec]

                output_filename = (
                    f"{source_id}_{codec}_{bitrate}k{extension}"
                )
                output_path = compressed_dir / output_filename

                # Never silently reuse an old file: regenerate it.
                output_path.unlink(missing_ok=True)

                compress_audio(
                    input_path=source_path,
                    output_path=output_path,
                    codec=codec,
                    bitrate=bitrate,
                    ffmpeg_path=FFMPEG_PATH,
                )

                record = {
                    "source_id": source_id,
                    "source_name": source_name,
                    "label": label,
                    "condition_type": "codec_compression",
                    "codec": codec,
                    "bitrate_kbps": int(bitrate),
                    "audio_path": str(output_path.resolve()),
                    "sha256": sha256_file(output_path),
                }

                for column in df.columns:
                    if column not in record:
                        record[column] = row[column]

                records.append(record)

    if missing_sources:
        print(
            f"\nWARNING: {len(missing_sources)} source files were missing."
        )
        for name in missing_sources[:20]:
            print(f"  Missing: {name}")

    metadata = pd.DataFrame(records)
    metadata_path = output_root / "codec_subset_metadata.csv"

    metadata.to_csv(metadata_path, index=False)

    print("\nGeneration summary")
    print("------------------")
    print(f"Requested sources: {num_samples}")
    print(f"Sources successfully processed: {metadata['source_id'].nunique()}")
    print(f"Metadata rows: {len(metadata)}")
    print(f"Expected rows if all sources succeeded: {num_samples * 10}")
    print(f"Metadata saved to: {metadata_path}")

    print("\nCondition counts:")
    print(metadata["condition_type"].value_counts().to_string())

    print("\nCodec and bitrate counts:")
    compressed = metadata[metadata["condition_type"] == "codec_compression"]

    if not compressed.empty:
        print(
            compressed.groupby(["codec", "bitrate_kbps"])
            .size()
            .to_string()
        )

    if missing_sources:
        raise RuntimeError(
            f"{len(missing_sources)} source files were missing. "
            "Review the warnings before evaluation."
        )

    expected_rows = num_samples * 10

    if len(metadata) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} rows, got {len(metadata)}."
        )

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Generate a deterministic codec robustness subset."
    )

    parser.add_argument("--manifest", required=True)
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    generate_subset(
        manifest_path=args.manifest,
        audio_root=args.audio_root,
        output_root=args.output_root,
        num_samples=args.num_samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()