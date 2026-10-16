
import sys
import json
import subprocess
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
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


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoint_import"
    / "checkpoints"
    / "E1"
    / "E1_MobileNetV3Small_ATADD_v4_best.pt"
)

DATA_ROOT = Path(r"G:\BTP\data\robustness\codec_subset_dev500_v2")
METADATA_PATH = DATA_ROOT / "codec_subset_metadata.csv"
OUTPUT_DIR = DATA_ROOT / "predictions"
FFMPEG_PATH = Path(r"G:\ffmpeg\bin\ffmpeg.exe")


# ============================================================
# 2. MODEL AND AUDIO CONFIGURATION
# Must match the baseline checkpoint preprocessing
# ============================================================

SAMPLE_RATE = 24000
SEGMENT_SECONDS = 4
NUM_SAMPLES = SAMPLE_RATE * SEGMENT_SECONDS

N_FFT = 1024
WIN_LENGTH = 600
HOP_LENGTH = 240
N_MELS = 80
LOG_EPS = 1e-10

DEVICE = torch.device("cpu")


# ============================================================
# 3. MODEL ARCHITECTURE
# Checkpoint parameters use the "backbone.*" prefix
# ============================================================

class MobileNetV3SmallAudio(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = mobilenet_v3_small(
            weights=None,
            num_classes=2,
        )

        original_conv = self.backbone.features[0][0]

        self.backbone.features[0][0] = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False,
        )

    def forward(self, x):
        return self.backbone(x)


# ============================================================
# 4. LOG-MEL FEATURE EXTRACTION
# ============================================================

class GPULogMel(nn.Module):
    def __init__(
        self,
        sample_rate,
        n_fft,
        win_length,
        hop_length,
        n_mels,
        log_eps=1e-10,
    ):
        super().__init__()

        mel_filter = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=0.0,
            fmax=None,
            htk=False,
            norm="slaney",
        )

        window = torch.hann_window(win_length)

        self.register_buffer(
            "mel_filter",
            torch.tensor(mel_filter, dtype=torch.float32),
        )
        self.register_buffer("window", window)

        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.log_eps = log_eps

    def forward(self, waveforms):
        stft = torch.stft(
            waveforms,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )

        power = stft.abs().pow(2)

        mel = torch.einsum(
            "mf,bft->bmt",
            self.mel_filter,
            power,
        )

        return torch.log(
            torch.clamp(mel, min=self.log_eps)
        )


# ============================================================
# 5. AUDIO LOADING AND PREPROCESSING
# ============================================================

def decode_with_ffmpeg(path):
    """Decode audio as mono float32 PCM at 24 kHz."""

    command = [
        str(FFMPEG_PATH),
        "-v", "error",
        "-i", str(path),
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "pipe:1",
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    if result.returncode != 0:
        error = result.stderr.decode(
            "utf-8",
            errors="replace",
        )
        raise RuntimeError(
            f"FFmpeg failed for {path}:\n{error}"
        )

    return np.frombuffer(
        result.stdout,
        dtype="<f4",
    ).copy()


def load_audio(path):
    """Load audio, convert to mono, and resample to 24 kHz."""

    path = Path(path)

    try:
        waveform, sample_rate = sf.read(
            str(path),
            dtype="float32",
            always_2d=False,
        )

        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)

        waveform = np.asarray(
            waveform,
            dtype=np.float32,
        )

        if sample_rate != SAMPLE_RATE:
            waveform = librosa.resample(
                waveform,
                orig_sr=sample_rate,
                target_sr=SAMPLE_RATE,
            )

    except Exception:
        # Use FFmpeg for formats SoundFile cannot decode.
        waveform = decode_with_ffmpeg(path)

    waveform = np.nan_to_num(
        waveform,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return waveform.astype(np.float32)


def prepare_waveform(path):
    """Center-crop long audio and repeat-pad short audio."""

    waveform = load_audio(path)

    if len(waveform) == 0:
        waveform = np.zeros(
            NUM_SAMPLES,
            dtype=np.float32,
        )

    elif len(waveform) >= NUM_SAMPLES:
        start = (len(waveform) - NUM_SAMPLES) // 2
        waveform = waveform[start:start + NUM_SAMPLES]

    else:
        repeats = int(
            np.ceil(NUM_SAMPLES / len(waveform))
        )
        waveform = np.tile(waveform, repeats)
        waveform = waveform[:NUM_SAMPLES]

    return torch.from_numpy(
        waveform.astype(np.float32)
    )


# ============================================================
# 6. METRIC FUNCTIONS
# ============================================================

def calculate_eer(labels, scores):
    """Estimate equal error rate from the ROC curve."""

    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)

    if len(np.unique(labels)) < 2:
        return None

    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1.0 - tpr

    index = np.nanargmin(np.abs(fpr - fnr))

    return float(
        (fpr[index] + fnr[index]) / 2.0
    )


def calculate_metrics(frame):
    labels = frame["label_binary"].to_numpy(dtype=int)
    predictions = frame["predicted_label"].to_numpy(dtype=int)
    scores = frame["fake_probability"].to_numpy(dtype=float)

    result = {
        "samples": int(len(frame)),
        "real_samples": int((labels == 0).sum()),
        "fake_samples": int((labels == 1).sum()),
        "accuracy": float(
            accuracy_score(labels, predictions)
        ),
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                labels=[0, 1],
                average="macro",
                zero_division=0,
            )
        ),
        "eer": calculate_eer(labels, scores),
    }

    if len(np.unique(labels)) == 2:
        result["auroc"] = float(
            roc_auc_score(labels, scores)
        )
    else:
        result["auroc"] = None

    return result


# ============================================================
# 7. MAIN EVALUATION
# ============================================================

def main():
    print("=" * 65)
    print("AT-ADD CODEC ROBUSTNESS EVALUATION")
    print("=" * 65)

    print("Project root:", PROJECT_ROOT)
    print("Device:", DEVICE)
    print("Metadata:", METADATA_PATH)
    print("Checkpoint:", CHECKPOINT_PATH)

    # --------------------------------------------------------
    # Check required files
    # --------------------------------------------------------

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT_PATH}"
        )

    if not METADATA_PATH.is_file():
        raise FileNotFoundError(
            f"Metadata CSV not found:\n{METADATA_PATH}"
        )

    if not FFMPEG_PATH.is_file():
        raise FileNotFoundError(
            f"FFmpeg not found:\n{FFMPEG_PATH}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load and validate metadata
    # --------------------------------------------------------

    df = pd.read_csv(METADATA_PATH)

    # Compatibility with the v2 codec metadata schema
    if "bitrate" not in df.columns and "bitrate_kbps" in df.columns:
        df["bitrate"] = df["bitrate_kbps"]

    required_columns = [
        "source_name",
        "condition_type",
        "codec",
        "bitrate",
        "audio_path",
        "label",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing metadata columns: {missing_columns}"
        )

    df["audio_path"] = df["audio_path"].astype(str)

    missing_audio = [
        path
        for path in df["audio_path"]
        if not Path(path).is_file()
    ]

    if missing_audio:
        raise FileNotFoundError(
            f"{len(missing_audio)} audio files are missing.\n"
            f"First missing path: {missing_audio[0]}"
        )

    normalized_labels = (
        df["label"].astype(str).str.strip().str.lower()
    )

    unknown_labels = sorted(
        set(normalized_labels.unique()) - {"real", "fake"}
    )

    if unknown_labels:
        raise ValueError(
            f"Unexpected labels: {unknown_labels}"
        )

    df["label_binary"] = (
        normalized_labels == "fake"
    ).astype(int)

    print("\nMetadata rows:", len(df))
    print("Unique sources:", df["source_name"].nunique())

    print("\nLabel distribution:")
    print(df["label"].value_counts().to_string())

    print("\nCondition distribution:")
    print(df["condition_type"].value_counts().to_string())

    print("\nCodec/bitrate distribution:")
    print(
        df.groupby(
            ["codec", "bitrate"],
            dropna=False,
        ).size().to_string()
    )

    # --------------------------------------------------------
    # Load frozen baseline checkpoint
    # --------------------------------------------------------

    print("\nLoading checkpoint...")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    config = checkpoint.get("config", {})

    mel_mean = config.get("mel_mean")
    mel_std = config.get("mel_std")

    if mel_mean is None or mel_std is None:
        raise KeyError(
            "Checkpoint config must contain mel_mean and mel_std."
        )

    mel_mean = float(mel_mean)
    mel_std = float(mel_std)

    if not np.isfinite(mel_std) or mel_std <= 0:
        raise ValueError(
            f"Invalid mel_std value: {mel_std}"
        )

    model = MobileNetV3SmallAudio().to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.eval()

    frontend = GPULogMel(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        log_eps=LOG_EPS,
    ).to(DEVICE)

    print("Checkpoint loaded successfully.")
    print("Checkpoint epoch:", checkpoint.get("epoch"))
    print(
        "Best dev Macro-F1:",
        checkpoint.get("best_dev_macro_f1"),
    )
    print("Mel mean:", mel_mean)
    print("Mel std:", mel_std)

    # --------------------------------------------------------
    # Run inference on clean and compressed audio
    # --------------------------------------------------------

    records = []

    print("\nRunning inference...")

    with torch.inference_mode():
        for row in tqdm(
            df.to_dict(orient="records"),
            total=len(df),
        ):
            waveform = prepare_waveform(
                row["audio_path"]
            ).unsqueeze(0).to(DEVICE)

            # [batch, waveform_samples]
            log_mel = frontend(waveform)

            # Use the baseline's training-set statistics.
            log_mel = (log_mel - mel_mean) / mel_std

            # MobileNet input: [batch, channels, mel, time]
            log_mel = log_mel.unsqueeze(1)

            logits = model(log_mel)

            probabilities = torch.softmax(
                logits.float(),
                dim=1,
            )

            fake_probability = float(
                probabilities[0, 1].item()
            )

            predicted_label = int(
                fake_probability >= 0.5
            )

            actual_label = int(row["label_binary"])

            records.append({
                **row,
                "fake_probability": fake_probability,
                "predicted_label": predicted_label,
                "predicted_class": (
                    "fake" if predicted_label == 1 else "real"
                ),
                "correct": predicted_label == actual_label,
            })

    predictions = pd.DataFrame(records)

    # --------------------------------------------------------
    # Save all predictions
    # --------------------------------------------------------

    predictions_path = OUTPUT_DIR / "codec_predictions.csv"

    predictions.to_csv(
        predictions_path,
        index=False,
    )

    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

    overall = calculate_metrics(predictions)

    with open(
        OUTPUT_DIR / "overall_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            overall,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # Metrics by clean/compressed condition
    # --------------------------------------------------------

    condition_records = []

    for condition, group in predictions.groupby(
        "condition_type",
        dropna=False,
    ):
        condition_records.append({
            "condition_type": str(condition),
            **calculate_metrics(group),
        })

    condition_summary = pd.DataFrame(condition_records)

    condition_summary.to_csv(
        OUTPUT_DIR / "codec_condition_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Metrics by codec and bitrate
    # --------------------------------------------------------

    codec_records = []

    for (codec, bitrate), group in predictions.groupby(
        ["codec", "bitrate"],
        dropna=False,
    ):
        codec_records.append({
            "codec": str(codec),
            "bitrate": str(bitrate),
            **calculate_metrics(group),
        })

    codec_summary = pd.DataFrame(codec_records)

    codec_summary.to_csv(
        OUTPUT_DIR / "codec_bitrate_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Paired comparison with the same clean source
    # --------------------------------------------------------

    clean_rows = predictions[
        predictions["condition_type"].astype(str).str.lower()
        == "clean"
    ][[
        "source_name",
        "fake_probability",
        "predicted_label",
        "correct",
    ]].rename(
        columns={
            "fake_probability": "clean_fake_probability",
            "predicted_label": "clean_predicted_label",
            "correct": "clean_correct",
        }
    )

    if clean_rows["source_name"].duplicated().any():
        raise ValueError(
            "More than one clean record exists for a source."
        )

    paired = predictions.merge(
        clean_rows,
        on="source_name",
        how="left",
        validate="many_to_one",
    )

    if paired["clean_fake_probability"].isna().any():
        raise ValueError(
            "Some records have no matching clean source."
        )

    paired["probability_change_from_clean"] = (
        paired["fake_probability"]
        - paired["clean_fake_probability"]
    )

    paired["prediction_flip_from_clean"] = (
        paired["predicted_label"]
        != paired["clean_predicted_label"]
    )

    paired["correctness_change_from_clean"] = (
        paired["correct"].astype(int)
        - paired["clean_correct"].astype(int)
    )

    paired.to_csv(
        OUTPUT_DIR / "paired_clean_vs_codec.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("OVERALL METRICS")
    print("=" * 65)

    for key, value in overall.items():
        print(f"{key}: {value}")

    print("\n" + "=" * 65)
    print("METRICS BY CONDITION")
    print("=" * 65)
    print(condition_summary.to_string(index=False))

    print("\n" + "=" * 65)
    print("METRICS BY CODEC AND BITRATE")
    print("=" * 65)
    print(codec_summary.to_string(index=False))

    # --------------------------------------------------------
    # Paired compression analysis
    # --------------------------------------------------------

    compressed = paired[
        paired["condition_type"].astype(str).str.lower()
        != "clean"
    ].copy()

    if len(compressed):
        print("\n" + "=" * 65)
        print("PAIRED COMPRESSION ANALYSIS")
        print("=" * 65)

        print("Compressed samples:", len(compressed))

        print(
            "Prediction flips from clean:",
            int(compressed["prediction_flip_from_clean"].sum()),
        )

        print(
            "Mean absolute probability change:",
            float(
                compressed[
                    "probability_change_from_clean"
                ].abs().mean()
            ),
        )

        print(
            "Compression cases that became incorrect:",
            int(
                (
                    (compressed["clean_correct"] == True)
                    & (compressed["correct"] == False)
                ).sum()
            ),
        )

        print(
            "Compression cases that became correct:",
            int(
                (
                    (compressed["clean_correct"] == False)
                    & (compressed["correct"] == True)
                ).sum()
            ),
        )

        flip_summary = (
            compressed.groupby(
                ["codec", "bitrate"],
                dropna=False,
            )
            .agg(
                prediction_flips=(
                    "prediction_flip_from_clean",
                    "sum",
                ),
                samples=("source_name", "count"),
                mean_absolute_probability_change=(
                    "probability_change_from_clean",
                    lambda values: values.abs().mean(),
                ),
                became_incorrect=(
                    "correctness_change_from_clean",
                    lambda values: (values == -1).sum(),
                ),
                became_correct=(
                    "correctness_change_from_clean",
                    lambda values: (values == 1).sum(),
                ),
            )
            .reset_index()
        )

        flip_summary.to_csv(
            OUTPUT_DIR / "codec_paired_summary.csv",
            index=False,
        )

        print("\nPaired results by codec and bitrate:")
        print(flip_summary.to_string(index=False))

    # --------------------------------------------------------
    # Final output paths
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("EVALUATION FINISHED")
    print("=" * 65)

    print("Output directory:", OUTPUT_DIR)
    print(" - codec_predictions.csv")
    print(" - overall_metrics.json")
    print(" - codec_condition_summary.csv")
    print(" - codec_bitrate_summary.csv")
    print(" - paired_clean_vs_codec.csv")
    print(" - codec_paired_summary.csv")

    print("\nIMPORTANT:")
    print(
        "These are development-set robustness results. "
        "Keep the official evaluation set untouched."
    )


if __name__ == "__main__":
    main()
