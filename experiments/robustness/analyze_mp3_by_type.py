import pandas as pd
from pathlib import Path

p = Path(
    r"G:\BTP\data\robustness\codec_subset_dev500_v2"
    r"\predictions\codec_predictions.csv"
)
df = pd.read_csv(p)

df["label_binary"] = pd.to_numeric(df["label_binary"])
df["predicted_label"] = pd.to_numeric(df["predicted_label"])
df["fake_probability"] = pd.to_numeric(df["fake_probability"])

mp3 = df[
    (df["codec"] == "mp3") &
    (df["condition_type"] == "codec_compression")
].copy()

mp3["audio_type"] = mp3["type"].fillna("unknown")
mp3["is_fake"] = mp3["label_binary"] == 1
mp3["fake_missed"] = (
    mp3["is_fake"] & (mp3["predicted_label"] == 0)
)
mp3["real_false_alarm"] = (
    (~mp3["is_fake"]) & (mp3["predicted_label"] == 1)
)

result = mp3.groupby(
    ["audio_type", "bitrate_kbps"]
).agg(
    samples=("label_binary", "size"),
    fake_samples=("is_fake", "sum"),
    fake_misses=("fake_missed", "sum"),
    real_false_alarms=("real_false_alarm", "sum"),
    mean_fake_probability=("fake_probability", "mean")
).reset_index()

result["fake_recall"] = (
    1 - result["fake_misses"] / result["fake_samples"]
)

print(result.round(4).to_string(index=False))
