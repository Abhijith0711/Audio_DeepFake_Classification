import json

import pandas as pd
import pytest

from src.datasets.atadd_manifest_dataset import audio_path_from_row, find_project_root
from src.evaluation.seed_summary import collect_seed_metrics, experiment_id, summarize_seed_metrics


def test_audio_path_uses_original_folder(tmp_path):
    for split in ("train", "dev", "eval"):
        (tmp_path / "audio" / split).mkdir(parents=True)

    row = {"name": "clip.flac", "split": "dev", "folder": "train"}
    path = audio_path_from_row(row, tmp_path, "audio")
    assert path == tmp_path / "audio" / "train" / "clip.flac"


def test_reworked_subset_sample_files_exist():
    root = find_project_root()
    manifest_dir = root / "reworked_subset" / "manifests"
    if not (manifest_dir / "train_manifest.csv").exists():
        pytest.skip("reworked_subset manifests are not in this checkout")

    for name in ("train_manifest.csv", "dev_manifest.csv", "eval_manifest.csv"):
        frame = pd.read_csv(manifest_dir / name)
        sample = frame.sample(n=min(20, len(frame)), random_state=0)
        for _, row in sample.iterrows():
            assert audio_path_from_row(row, root, "reworked_subset").is_file()


def test_seed_summary_reports_mean_and_variance(tmp_path):
    for seed_tag, macro_f1, eer in (("seed1", 0.50, 0.20), ("seed2", 0.60, 0.10), ("seed3", 0.70, 0.30)):
        experiment_dir = tmp_path / "experiments" / experiment_id("M0", seed_tag)
        experiment_dir.mkdir(parents=True)
        payload = {"macro_f1": macro_f1, "eer": eer, "type_avg_macro_f1": macro_f1 - 0.05}
        (experiment_dir / "eval_metrics_eval.json").write_text(json.dumps(payload), encoding="utf-8")
        (experiment_dir / "eval_metrics_dev.json").write_text(json.dumps(payload), encoding="utf-8")

    runs = collect_seed_metrics(tmp_path, modes=("M0",), seeds=(("seed1", 42), ("seed2", 1337), ("seed3", 2024)))
    summary = summarize_seed_metrics(runs)
    eval_f1 = summary[(summary["mode"] == "M0") & (summary["split"] == "eval") & (summary["metric"] == "macro_f1")].iloc[0]

    assert eval_f1["n_seeds"] == 3
    assert eval_f1["mean"] == pytest.approx(0.60)
    assert eval_f1["variance"] == pytest.approx(0.01)
