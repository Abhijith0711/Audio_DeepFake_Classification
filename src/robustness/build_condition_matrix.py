import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from src.datasets.atadd_manifest_dataset import (
    DEFAULT_AUDIO_ROOT_NAME,
    audio_path_from_row,
    find_project_root,
    load_manifest_df,
)
from src.preprocessing.preprocess_atadd_v4 import load_audio, repeat_pad_audio
from src.robustness.conditions import all_training_conditions
from src.robustness.degrade import degrade, ffmpeg_available


def build_matrix(
    split,
    out_dir,
    max_samples=None,
    sample_rate=24000,
    segment_seconds=4,
    audio_root_name=DEFAULT_AUDIO_ROOT_NAME,
):
    project_root = find_project_root()
    manifest_rel = f"data/manifests/{split}_manifest.csv"
    df = load_manifest_df(project_root, manifest_rel, split, max_samples=max_samples)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    num_samples = sample_rate * segment_seconds
    rows = []

    conditions = [c for c in all_training_conditions() if c.family != "identity"]

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"condition_matrix_{split}"):
        clean_path = audio_path_from_row(row, project_root, audio_root_name)

        if not clean_path.exists():
            continue

        waveform = load_audio(clean_path, sample_rate)

        if len(waveform) >= num_samples:
            start = (len(waveform) - num_samples) // 2
            waveform = waveform[start : start + num_samples]
        else:
            waveform = repeat_pad_audio(waveform, num_samples)

        source_id = str(row.get("file_id", row["name"]))

        for condition in conditions:
            if condition.family == "mp3_codec" and not ffmpeg_available():
                continue

            spec = condition
            if spec.seed is None:
                spec = type(condition)(
                    family=condition.family,
                    name=condition.name,
                    severity=condition.severity,
                    params=condition.params,
                    seed=42,
                )

            degraded, meta = degrade(
                waveform,
                sample_rate,
                spec,
                mode="deterministic",
                source_id=source_id,
            )

            rel_name = f"{source_id}__{condition.condition_id}.flac"
            degraded_path = out_dir / split / condition.condition_id / rel_name
            degraded_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(degraded_path, degraded, sample_rate)

            rows.append(
                {
                    "pair_id": f"{source_id}__{condition.condition_id}",
                    "split": split,
                    "source_id": source_id,
                    "clean_path": str(clean_path.relative_to(project_root)),
                    "degraded_path": str(degraded_path.relative_to(project_root)),
                    "label": row["label"],
                    "type": row.get("type", "unknown"),
                    "generator": row.get("generator", "-"),
                    **meta,
                }
            )

    matrix_path = out_dir / f"{split}_condition_matrix.csv"
    pd.DataFrame(rows).to_csv(matrix_path, index=False)

    meta_path = out_dir / f"{split}_condition_matrix_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "split": split,
                "sample_rate": sample_rate,
                "segment_seconds": segment_seconds,
                "ffmpeg": ffmpeg_available(),
                "num_rows": len(rows),
            },
            f,
            indent=2,
        )

    print(f"Wrote {matrix_path} ({len(rows)} rows)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="eval", choices=["train", "dev", "eval"])
    parser.add_argument("--out", default="data/manifests/conditioned")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--audio-root", default=DEFAULT_AUDIO_ROOT_NAME)
    args = parser.parse_args()

    build_matrix(
        args.split,
        args.out,
        max_samples=args.max_samples,
        audio_root_name=args.audio_root,
    )


if __name__ == "__main__":
    main()
