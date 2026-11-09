from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from src.datasets.atadd_manifest_dataset import audio_path_from_row
from src.robustness.codecs import available_conditions, generate_robustness_dataset


def test_available_conditions_cover_requested_augmentations():
    conditions = available_conditions()

    assert {"mp3_64k", "mp3_128k", "mp3_192k"} <= conditions.keys()
    assert {"aac_32k", "aac_48k", "aac_64k"} <= conditions.keys()
    assert {"opus_32k", "opus_64k", "opus_128k"} <= conditions.keys()
    assert {"resample_16k", "resample_12k", "resample_8k"} <= conditions.keys()
    assert {"snr_20db", "snr_10db", "snr_5db"} <= conditions.keys()
    assert "telephony_300_3400" in conditions


def test_generate_robustness_dataset_writes_training_manifest(tmp_path):
    source_path = tmp_path / "source.wav"
    sample_rate = 24000
    waveform = np.sin(
        2 * np.pi * 440 * np.arange(sample_rate, dtype=np.float32) / sample_rate
    )
    sf.write(source_path, waveform, sample_rate)

    input_manifest = tmp_path / "input.csv"
    pd.DataFrame(
        [{
            "name": source_path.name,
            "file_id": "clip-1",
            "label": "real",
            "split": "train",
            "type": "speech",
            "audio_path": str(source_path),
        }]
    ).to_csv(input_manifest, index=False)

    conditions = ["resample_16k", "snr_10db", "telephony_300_3400"]
    output_root = tmp_path / "generated"
    result = generate_robustness_dataset(
        input_manifest,
        output_root,
        conditions=conditions,
        seed=7,
    )

    assert len(result) == len(conditions) + 1
    assert result["condition_name"].tolist() == [
        "clean",
        *conditions,
    ]
    assert set(result["label"]) == {"real"}
    assert set(result["split"]) == {"train"}
    assert result["file_id"].is_unique
    assert all(Path(path).is_file() for path in result["audio_path"])

    generated = result[result["condition_name"] != "clean"]
    for _, row in generated.iterrows():
        path = audio_path_from_row(row, tmp_path)
        assert path.is_file()
        info = sf.info(str(path))
        assert info.samplerate == sample_rate
        assert info.frames > 0

    resampled = result[result["condition_name"] == "resample_16k"].iloc[0]
    assert resampled["intermediate_sr"] == 16000
    assert (output_root / "robustness_manifest.csv").is_file()
