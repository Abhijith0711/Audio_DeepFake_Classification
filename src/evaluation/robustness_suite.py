"""Condition-wise robustness evaluation of several checkpoints on one shared degraded eval subset.

Degraded audio is generated once per condition and scored by every model, so differences
between models are not confounded by different noise draws. Seen conditions are those in the
training catalogue; unseen conditions come from `src.robustness.unseen`.
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from src.datasets.atadd_manifest_dataset import (
    audio_path_from_row,
    find_project_root,
    load_manifest_df,
    stratified_indices,
)
from src.evaluation.eval_runner import build_model_from_checkpoint
from src.evaluation.metrics import calculate_metrics
from src.preprocessing.preprocess_atadd_v4 import load_audio, repeat_pad_audio
from src.robustness.conditions import ConditionSpec, all_training_conditions
from src.robustness.degrade import degrade
from src.robustness.unseen import UNSEEN_CONDITIONS, apply_unseen
from src.training.frontend import MelFrontend, build_gpu_logmel

SAMPLE_RATE = 24000
NUM_SAMPLES = SAMPLE_RATE * 4

EXTRA_SEEN_SANITY = [
    # Native audio is 16 kHz, so a 16 kHz round trip is an identity operation (sanity check).
    ConditionSpec("resample", "down_up_16k", 0, {"intermediate_sr": 16000}),
    ConditionSpec("resample", "down_up_8k", 4, {"intermediate_sr": 8000}),
]


def center_crop(waveform):
    if len(waveform) >= NUM_SAMPLES:
        start = (len(waveform) - NUM_SAMPLES) // 2
        return waveform[start : start + NUM_SAMPLES]
    return repeat_pad_audio(waveform, NUM_SAMPLES)


def _degrade_one(args):
    waveform, source_id, name, spec = args
    if spec is not None:
        out, _ = degrade(waveform, SAMPLE_RATE, spec, mode="deterministic", source_id=source_id)
        return out
    return apply_unseen(waveform, name, source_id)


def make_condition(name, spec, clean, ids, workers):
    jobs = [(w, i, name, spec) for w, i in zip(clean, ids)]
    with Pool(workers) as pool:
        return np.stack(pool.map(_degrade_one, jobs, chunksize=16))


def score(model, mode, frontend, waveforms, device, batch_size=64):
    scores = []
    with torch.no_grad():
        for start in range(0, len(waveforms), batch_size):
            batch = torch.from_numpy(np.ascontiguousarray(waveforms[start : start + batch_size])).float().to(device)
            x = frontend(batch)
            logits = model(x) if mode == "M0" else model(x)[0]
            scores.extend(torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy().tolist())
    return np.asarray(scores)


def window_scores(model, mode, frontend, long_waveforms, device, max_windows=8):
    """Mean fake probability over non-overlapping 4 s windows of the full recording."""
    out = []
    for wave in long_waveforms:
        n = int(np.clip(len(wave) // NUM_SAMPLES, 1, max_windows))
        if len(wave) < NUM_SAMPLES:
            windows = [repeat_pad_audio(wave, NUM_SAMPLES)]
        else:
            starts = np.linspace(0, len(wave) - NUM_SAMPLES, n).astype(int)
            windows = [wave[s : s + NUM_SAMPLES] for s in starts]
        out.append(score(model, mode, frontend, np.stack(windows), device).mean())
    return np.asarray(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", required=True, help="Comma-separated experiment ids with best.pt")
    parser.add_argument("--config", default="configs/final/gd_base.yaml")
    parser.add_argument("--n", type=int, default=2400)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default="results/tables/final")
    args = parser.parse_args()

    root = find_project_root()
    out_dir = root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((root / args.config).read_text())
    device = torch.device("cpu")

    df = load_manifest_df(root, config["dataset"]["eval_manifest"], "eval")
    df = df.iloc[stratified_indices(df, args.n, seed=7, columns=("type", "label"))].reset_index(drop=True)

    clean, long_audio, ids = [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="load"):
        wave = load_audio(audio_path_from_row(row, root, config["dataset"]["audio_root"]), SAMPLE_RATE)
        long_audio.append(wave)
        clean.append(center_crop(wave).astype(np.float32))
        ids.append(str(row["file_id"]))
    clean = np.stack(clean)

    models = {}
    for exp_id in args.ids.split(","):
        ckpt = torch.load(root / "experiments" / exp_id / "best.pt", map_location=device, weights_only=False)
        model, mode = build_model_from_checkpoint(ckpt, device)
        frontend = MelFrontend(build_gpu_logmel(config, device), ckpt["mel_mean"], ckpt["mel_std"], device)
        models[exp_id] = (model, mode, frontend)

    conditions = {"clean": None}
    for c in all_training_conditions():
        if c.family != "identity":
            conditions[c.name] = (c, "seen")
    for c in EXTRA_SEEN_SANITY:
        conditions[c.name] = (c, "sanity")
    for name in UNSEEN_CONDITIONS:
        conditions[name] = (None, "unseen")

    rows = []
    for name, spec in tqdm(conditions.items(), desc="conditions"):
        if name == "clean":
            data, kind, cond_spec = clean, "clean", None
        else:
            cond_spec, kind = spec
            data = make_condition(name, cond_spec, clean, ids, args.workers)

        for exp_id, (model, mode, frontend) in models.items():
            s = score(model, mode, frontend, data, device)
            for i in range(len(df)):
                rows.append((exp_id, name, kind, ids[i], df.type[i], int(df.label[i] == "fake"), float(s[i])))

        pd.DataFrame(rows, columns=["model", "condition", "kind", "file_id", "type", "label", "score"]).to_csv(
            out_dir / "robustness_scores.csv", index=False
        )

    # Multi-window scoring of the complete recordings (clean).
    win_rows = []
    for exp_id, (model, mode, frontend) in models.items():
        s = window_scores(model, mode, frontend, long_audio, device)
        for i in range(len(df)):
            win_rows.append((exp_id, ids[i], df.type[i], int(df.label[i] == "fake"), float(s[i])))
    pd.DataFrame(win_rows, columns=["model", "file_id", "type", "label", "score"]).to_csv(
        out_dir / "multiwindow_scores.csv", index=False
    )
    print("done")


if __name__ == "__main__":
    main()
