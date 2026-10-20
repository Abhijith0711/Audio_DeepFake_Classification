from pathlib import Path
import pandas as pd

root = Path(r"G:\BTP\data\robustness\codec_subset_dev500_v2")
path = root / "predictions" / "codec_predictions.csv"
out = root / "error_analysis"
out.mkdir(exist_ok=True)

df = pd.read_csv(path)

df["fake_probability"] = pd.to_numeric(df["fake_probability"])
df["label_binary"] = pd.to_numeric(df["label_binary"])

clean = df[df["condition_type"] == "clean"][
    ["source_id", "label", "type", "fake_probability", "predicted_label"]
].rename(columns={
    "fake_probability": "clean_fake_probability",
    "predicted_label": "clean_prediction"
})

mp3 = df[
    (df["condition_type"] == "codec_compression") &
    (df["codec"] == "mp3")
][[
    "source_id", "bitrate_kbps", "fake_probability", "predicted_label"
]].rename(columns={
    "fake_probability": "mp3_fake_probability",
    "predicted_label": "mp3_prediction"
})

paired = mp3.merge(clean, on="source_id", validate="many_to_one")

paired["probability_change"] = (
    paired["mp3_fake_probability"] - paired["clean_fake_probability"]
)

paired["clean_correct"] = (
    paired["clean_prediction"] == paired["label"].map({"real": 0, "fake": 1})
)
paired["mp3_correct"] = (
    paired["mp3_prediction"] == paired["label"].map({"real": 0, "fake": 1})
)

summary = paired.groupby(["type", "bitrate_kbps"]).agg(
    samples=("source_id", "size"),
    mean_clean_fake_probability=("clean_fake_probability", "mean"),
    mean_mp3_fake_probability=("mp3_fake_probability", "mean"),
    mean_probability_change=("probability_change", "mean"),
    clean_correct=("clean_correct", "sum"),
    mp3_correct=("mp3_correct", "sum")
).reset_index()

summary["correct_to_wrong"] = paired.groupby(
    ["type", "bitrate_kbps"]
).apply(
    lambda g: (
        g["clean_correct"] & ~g["mp3_correct"]
    ).sum(),
    include_groups=False
).to_numpy()

summary["wrong_to_correct"] = paired.groupby(
    ["type", "bitrate_kbps"]
).apply(
    lambda g: (
        ~g["clean_correct"] & g["mp3_correct"]
    ).sum(),
    include_groups=False
).to_numpy()

paired.to_csv(out / "paired_mp3_predictions.csv", index=False)
summary.to_csv(out / "paired_mp3_by_type.csv", index=False)

print(summary.round(4).to_string(index=False))
print(f"\nSaved to: {out}")
