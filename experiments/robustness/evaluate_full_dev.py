import sys
import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Reuse the frozen E1 evaluator's exact model, frontend,
# waveform preprocessing and metric implementations.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import experiments.robustness.evaluate_codec_subset as baseline

MANIFEST_PATH = Path(r"G:\BTP\Data\manifests\dev.csv")
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "robustness" / "full_dev_clean_results"
BATCH_SIZE = 32


def main():
    print("=" * 70)
    print("FROZEN E1 — FULL CLEAN DEVELOPMENT-SET EVALUATION")
    print("=" * 70)
    print("Manifest:", MANIFEST_PATH)
    print("Checkpoint:", baseline.CHECKPOINT_PATH)
    print("Device: CPU")

    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")
    if not baseline.CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {baseline.CHECKPOINT_PATH}"
        )
    if not baseline.FFMPEG_PATH.is_file():
        raise FileNotFoundError(
            f"FFmpeg not found: {baseline.FFMPEG_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(MANIFEST_PATH)
    required = {"name", "audio_path", "type", "label"}
    missing_columns = required - set(df.columns)
    if missing_columns:
        raise ValueError(f"Manifest is missing columns: {sorted(missing_columns)}")

    df["audio_path"] = df["audio_path"].astype(str)
    df["label_text"] = df["label"].astype(str).str.strip().str.lower()

    unknown = sorted(set(df["label_text"].unique()) - {"real", "fake"})
    if unknown:
        raise ValueError(f"Unexpected labels: {unknown}")

    df["label_binary"] = (df["label_text"] == "fake").astype(int)

    missing_paths = ~df["audio_path"].map(lambda p: Path(p).is_file())
    if missing_paths.any():
        missing_csv = OUTPUT_DIR / "missing_audio_paths.csv"
        df.loc[missing_paths].to_csv(missing_csv, index=False)
        raise FileNotFoundError(
            f"{int(missing_paths.sum())} files missing. Details: {missing_csv}"
        )

    print(f"Manifest rows: {len(df):,}")
    print("Label counts:")
    print(df["label_text"].value_counts().to_string())
    print("\nAudio-type counts:")
    print(df["type"].value_counts(dropna=False).to_string())

    checkpoint = torch.load(
        baseline.CHECKPOINT_PATH,
        map_location=baseline.DEVICE,
        weights_only=False,
    )
    config = checkpoint.get("config", {})
    mel_mean = config.get("mel_mean")
    mel_std = config.get("mel_std")

    if mel_mean is None or mel_std is None:
        raise KeyError("Checkpoint config must contain mel_mean and mel_std.")

    mel_mean, mel_std = float(mel_mean), float(mel_std)
    if not np.isfinite(mel_mean) or not np.isfinite(mel_std) or mel_std <= 0:
        raise ValueError(f"Invalid checkpoint normalization: mean={mel_mean}, std={mel_std}")

    model = baseline.MobileNetV3SmallAudio().to(baseline.DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    frontend = baseline.GPULogMel(
        sample_rate=baseline.SAMPLE_RATE,
        n_fft=baseline.N_FFT,
        win_length=baseline.WIN_LENGTH,
        hop_length=baseline.HOP_LENGTH,
        n_mels=baseline.N_MELS,
        log_eps=baseline.LOG_EPS,
    ).to(baseline.DEVICE)

    print("\nCheckpoint epoch:", checkpoint.get("epoch"))
    print("Best dev Macro-F1 in checkpoint:", checkpoint.get("best_dev_macro_f1"))
    print("Mel mean/std:", mel_mean, mel_std)

    records = []
    errors = []

    with torch.inference_mode():
        for start in tqdm(range(0, len(df), BATCH_SIZE), desc="Evaluating dev"):
            batch_df = df.iloc[start:start + BATCH_SIZE]
            waveforms = []
            valid_rows = []

            # Keep going if a file fails, but record every failure.
            for row in batch_df.to_dict(orient="records"):
                try:
                    wave = baseline.prepare_waveform(row["audio_path"])
                    waveforms.append(wave)
                    valid_rows.append(row)
                except Exception as exc:
                    errors.append({
                        "name": row.get("name"),
                        "audio_path": row["audio_path"],
                        "label": row["label_text"],
                        "type": row.get("type"),
                        "error": repr(exc),
                        "traceback": traceback.format_exc(limit=2),
                    })

            if not waveforms:
                continue

            x = torch.stack(waveforms).to(baseline.DEVICE)
            log_mel = frontend(x)
            log_mel = (log_mel - mel_mean) / mel_std
            logits = model(log_mel.unsqueeze(1))
            probs = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()

            for row, fake_prob in zip(valid_rows, probs):
                pred = int(float(fake_prob) >= 0.5)
                actual = int(row["label_binary"])
                records.append({
                    **row,
                    "fake_probability": float(fake_prob),
                    "predicted_label": pred,
                    "predicted_class": "fake" if pred else "real",
                    "correct": bool(pred == actual),
                })

    predictions = pd.DataFrame(records)
    predictions_path = OUTPUT_DIR / "full_dev_predictions.csv"
    predictions.to_csv(predictions_path, index=False)

    if errors:
        pd.DataFrame(errors).to_csv(
            OUTPUT_DIR / "inference_errors.csv", index=False
        )

    if predictions.empty:
        raise RuntimeError("No predictions were produced.")

    overall = baseline.calculate_metrics(predictions)
    overall.update({
        "manifest_rows": int(len(df)),
        "successfully_evaluated": int(len(predictions)),
        "failed_samples": int(len(errors)),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_best_dev_macro_f1": checkpoint.get("best_dev_macro_f1"),
        "threshold": 0.5,
        "note": (
            "Clean development-set evaluation using the frozen E1 checkpoint. "
            "Not an independent final-test result."
        ),
    })

    with open(OUTPUT_DIR / "overall_metrics.json", "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2)

    # Per-type metrics
    type_rows = []
    for audio_type, group in predictions.groupby("type", dropna=False):
        metrics = baseline.calculate_metrics(group)
        metrics["type"] = str(audio_type)
        type_rows.append(metrics)

    type_metrics = pd.DataFrame(type_rows)
    type_metrics.to_csv(OUTPUT_DIR / "metrics_by_type.csv", index=False)

    print("\n" + "=" * 70)
    print("OVERALL METRICS")
    print("=" * 70)
    for key, value in overall.items():
        print(f"{key}: {value}")

    print("\nMETRICS BY AUDIO TYPE")
    print(type_metrics.to_string(index=False))

    print("\nSaved predictions:", predictions_path)
    print("Saved overall metrics:", OUTPUT_DIR / "overall_metrics.json")
    print("Saved type metrics:", OUTPUT_DIR / "metrics_by_type.csv")
    if errors:
        print(f"Failed files: {len(errors)} — see {OUTPUT_DIR / 'inference_errors.csv'}")
    else:
        print("Failed files: 0")

    print("\nReminder: this is the development set, not an independent final test set.")


if __name__ == "__main__":
    main()
