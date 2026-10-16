import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from torchvision.models import mobilenet_v3_small
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

CHECKPOINT_PATH = (
    PROJECT_ROOT / "checkpoint_import" / "checkpoints" / "E1"
    / "E1_MobileNetV3Small_ATADD_v4_best.pt"
)

DATA_ROOT = Path(r"G:\BTP\Data\robustness\resampling_dev500_v1")
METADATA_PATH = DATA_ROOT / "resampling_metadata.csv"
CLEAN_PREDICTIONS = (
    Path(r"G:\BTP\Data\robustness\codec_subset_dev500_v2")
    / "predictions" / "codec_predictions.csv"
)
OUTPUT_DIR = DATA_ROOT / "predictions"

SAMPLE_RATE = 24000
SEGMENT_SECONDS = 4
NUM_SAMPLES = SAMPLE_RATE * SEGMENT_SECONDS
N_FFT = 1024
WIN_LENGTH = 600
HOP_LENGTH = 240
N_MELS = 80
LOG_EPS = 1e-10
DEVICE = torch.device("cpu")


class MobileNetV3SmallAudio(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = mobilenet_v3_small(weights=None, num_classes=2)
        conv = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(
            1, conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            bias=False,
        )

    def forward(self, x):
        return self.backbone(x)


class GPULogMel(nn.Module):
    def __init__(self):
        super().__init__()
        import librosa

        mel_filter = librosa.filters.mel(
            sr=SAMPLE_RATE,
            n_fft=N_FFT,
            n_mels=N_MELS,
            fmin=0.0,
            fmax=None,
            htk=False,
            norm="slaney",
        )
        self.register_buffer(
            "mel_filter",
            torch.tensor(mel_filter, dtype=torch.float32),
        )
        self.register_buffer("window", torch.hann_window(WIN_LENGTH))

    def forward(self, waveforms):
        stft = torch.stft(
            waveforms,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            window=self.window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )
        power = stft.abs().pow(2)
        mel = torch.einsum("mf,bft->bmt", self.mel_filter, power)
        return torch.log(torch.clamp(mel, min=LOG_EPS))


def prepare_waveform(path):
    # Match the existing codec evaluator's audio preprocessing.
    import librosa
    import soundfile as sf

    waveform, sr = sf.read(
        str(path), dtype="float32", always_2d=False
    )
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    waveform = np.asarray(waveform, dtype=np.float32)
    if sr != SAMPLE_RATE:
        waveform = librosa.resample(
            waveform, orig_sr=sr, target_sr=SAMPLE_RATE
        )
    waveform = np.nan_to_num(
        waveform, nan=0.0, posinf=0.0, neginf=0.0
    )

    if len(waveform) == 0:
        waveform = np.zeros(NUM_SAMPLES, dtype=np.float32)
    elif len(waveform) >= NUM_SAMPLES:
        start = (len(waveform) - NUM_SAMPLES) // 2
        waveform = waveform[start:start + NUM_SAMPLES]
    else:
        repeats = int(np.ceil(NUM_SAMPLES / len(waveform)))
        waveform = np.tile(waveform, repeats)[:NUM_SAMPLES]

    return torch.from_numpy(waveform.astype(np.float32))


def calculate_eer(labels, scores):
    if len(np.unique(labels)) < 2:
        return None
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def metrics(frame):
    y = frame["label_binary"].to_numpy(dtype=int)
    pred = frame["predicted_label"].to_numpy(dtype=int)
    scores = frame["fake_probability"].to_numpy(dtype=float)
    return {
        "samples": int(len(frame)),
        "real_samples": int((y == 0).sum()),
        "fake_samples": int((y == 1).sum()),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(
            f1_score(y, pred, labels=[0, 1],
                     average="macro", zero_division=0)
        ),
        "auroc": (
            float(roc_auc_score(y, scores))
            if len(np.unique(y)) == 2 else None
        ),
        "eer": calculate_eer(y, scores),
    }


def main():
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(CHECKPOINT_PATH)
    if not METADATA_PATH.is_file():
        raise FileNotFoundError(METADATA_PATH)
    if not CLEAN_PREDICTIONS.is_file():
        raise FileNotFoundError(CLEAN_PREDICTIONS)

    df = pd.read_csv(METADATA_PATH)
    required = {
        "source_id", "label", "intermediate_sr", "audio_path"
    }
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns: {sorted(missing_cols)}")
    if df["source_id"].isna().any() or df["source_id"].duplicated(
        keep=False
    ).groupby(df["source_id"]).any().any():
        # Each source should appear once per intermediate sample rate,
        # but never twice for the same source and rate.
        duplicates = df.duplicated(
            ["source_id", "intermediate_sr"], keep=False
        )
        if duplicates.any():
            raise ValueError("Duplicate source/rate entries in manifest.")

    missing_audio = [
        p for p in df["audio_path"].astype(str)
        if not Path(p).is_file()
    ]
    if missing_audio:
        raise FileNotFoundError(
            f"{len(missing_audio)} audio files missing; "
            f"first: {missing_audio[0]}"
        )

    df["label_binary"] = (
        df["label"].astype(str).str.strip().str.lower() == "fake"
    ).astype(int)

    checkpoint = torch.load(
        CHECKPOINT_PATH, map_location=DEVICE, weights_only=False
    )
    config = checkpoint.get("config", {})
    mel_mean = config.get("mel_mean")
    mel_std = config.get("mel_std")
    if mel_mean is None or mel_std is None:
        raise KeyError("Checkpoint config lacks mel_mean/mel_std.")
    mel_mean, mel_std = float(mel_mean), float(mel_std)
    if not np.isfinite(mel_std) or mel_std <= 0:
        raise ValueError(f"Invalid mel_std: {mel_std}")

    model = MobileNetV3SmallAudio().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    frontend = GPULogMel().to(DEVICE)

    records = []
    with torch.inference_mode():
        for row in tqdm(
            df.to_dict(orient="records"),
            total=len(df),
            desc="Evaluating resampling",
        ):
            waveform = prepare_waveform(row["audio_path"]).unsqueeze(0)
            log_mel = frontend(waveform)
            log_mel = (log_mel - mel_mean) / mel_std
            logits = model(log_mel.unsqueeze(1))
            fake_prob = float(torch.softmax(logits.float(), dim=1)[0, 1])
            pred = int(fake_prob >= 0.5)
            records.append({
                **row,
                "fake_probability": fake_prob,
                "predicted_label": pred,
                "predicted_class": "fake" if pred else "real",
                "correct": pred == int(row["label_binary"]),
            })

    predictions = pd.DataFrame(records)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(OUTPUT_DIR / "resampling_predictions.csv", index=False)

    # Reuse the existing clean predictions for the same 500 sources.
    clean = pd.read_csv(CLEAN_PREDICTIONS)
    clean = clean[clean["condition_type"] == "clean"].copy()
    clean["source_id"] = clean["source_id"].astype(str)
    predictions["source_id"] = predictions["source_id"].astype(str)

    if clean["source_id"].duplicated().any():
        raise ValueError("Clean reference contains duplicate source IDs.")

    clean = clean[[
        "source_id", "label", "fake_probability",
        "predicted_label", "correct",
    ]].rename(columns={
        "label": "clean_label",
        "fake_probability": "clean_fake_probability",
        "predicted_label": "clean_predicted_label",
        "correct": "clean_correct",
    })

    paired = predictions.merge(
        clean, on="source_id", how="left",
        validate="many_to_one", indicator=True,
    )
    if (paired["_merge"] != "both").any():
        raise ValueError("Some resampling rows lack clean predictions.")
    paired.drop(columns="_merge", inplace=True)

    if not (
        paired["label"].astype(str).str.lower().to_numpy()
        == paired["clean_label"].astype(str).str.lower().to_numpy()
    ).all():
        raise ValueError("Label mismatch between clean and resampled data.")

    paired["probability_change_from_clean"] = (
        paired["fake_probability"] - paired["clean_fake_probability"]
    )
    paired["prediction_flip_from_clean"] = (
        paired["predicted_label"] != paired["clean_predicted_label"]
    )
    paired["correctness_change_from_clean"] = (
        paired["correct"].astype(int) - paired["clean_correct"].astype(int)
    )
    paired.to_csv(OUTPUT_DIR / "paired_clean_vs_resampling.csv", index=False)

    summary = []
    for rate, group in predictions.groupby("intermediate_sr"):
        result = {"intermediate_sr": int(rate), **metrics(group)}
        summary.append(result)
    pd.DataFrame(summary).to_csv(
        OUTPUT_DIR / "resampling_condition_summary.csv", index=False
    )

    paired_summary = []
    for rate, group in paired.groupby("intermediate_sr"):
        paired_summary.append({
            "intermediate_sr": int(rate),
            "samples": int(len(group)),
            "prediction_flips": int(group["prediction_flip_from_clean"].sum()),
            "mean_absolute_probability_change": float(
                group["probability_change_from_clean"].abs().mean()
            ),
            "became_incorrect": int(
                ((group["clean_correct"] == True) & (group["correct"] == False)).sum()
            ),
            "became_correct": int(
                ((group["clean_correct"] == False) & (group["correct"] == True)).sum()
            ),
        })
    pd.DataFrame(paired_summary).to_csv(
        OUTPUT_DIR / "resampling_paired_summary.csv", index=False
    )

    with open(OUTPUT_DIR / "resampling_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {str(int(rate)): metrics(group)
             for rate, group in predictions.groupby("intermediate_sr")},
            f, indent=2,
        )

    print("\nResampling evaluation complete.")
    print("Predictions:", OUTPUT_DIR / "resampling_predictions.csv")
    print("Condition summary:", OUTPUT_DIR / "resampling_condition_summary.csv")
    print("Paired summary:", OUTPUT_DIR / "resampling_paired_summary.csv")
    print(pd.DataFrame(summary).to_string(index=False))
    print(pd.DataFrame(paired_summary).to_string(index=False))


if __name__ == "__main__":
    main()
