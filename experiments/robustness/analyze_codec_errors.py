from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    roc_auc_score
)

ROOT = Path(r"G:\BTP\data\robustness\codec_subset_dev500_v2")
PREDICTIONS = ROOT / "predictions" / "codec_predictions.csv"
OUT = ROOT / "error_analysis"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(PREDICTIONS)

# Verify required columns
required = {
    "label_binary", "predicted_label", "condition_type",
    "codec", "bitrate_kbps", "fake_probability"
}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

df["label_binary"] = pd.to_numeric(df["label_binary"])
df["predicted_label"] = pd.to_numeric(df["predicted_label"])
df["fake_probability"] = pd.to_numeric(df["fake_probability"])

# 0 = real, 1 = fake
df["correct_check"] = (
    df["label_binary"] == df["predicted_label"]
)

summary = []
class_rows = []

conditions = [
    ("Clean", df[df["condition_type"] == "clean"])
]

compressed = df[df["condition_type"] == "codec_compression"]

for (codec, bitrate), group in compressed.groupby(
    ["codec", "bitrate_kbps"], dropna=False
):
    conditions.append((f"{codec}_{bitrate}k", group))

for name, group in conditions:
    y_true = group["label_binary"].astype(int)
    y_pred = group["predicted_label"].astype(int)
    probabilities = group["fake_probability"]

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    try:
        auc = roc_auc_score(y_true, probabilities)
    except ValueError:
        auc = float("nan")

    summary.append({
        "condition": name,
        "samples": len(group),
        "real_samples": int((y_true == 0).sum()),
        "fake_samples": int((y_true == 1).sum()),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "auroc": auc,
        "true_real": int(tn),
        "real_predicted_fake": int(fp),
        "fake_predicted_real": int(fn),
        "true_fake": int(tp),
    })

    report = classification_report(
        y_true, y_pred,
        labels=[0, 1],
        target_names=["real", "fake"],
        output_dict=True,
        zero_division=0
    )

    for cls in ["real", "fake"]:
        class_rows.append({
            "condition": name,
            "class": cls,
            "precision": report[cls]["precision"],
            "recall": report[cls]["recall"],
            "f1_score": report[cls]["f1-score"],
            "support": report[cls]["support"],
        })

    # Save a separate confusion matrix image for each condition
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm)
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Real", "Fake"],
        yticklabels=["Real", "Fake"],
        xlabel="Predicted label",
        ylabel="True label",
        title=f"Confusion Matrix: {name}"
    )

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center")

    fig.tight_layout()
    fig.savefig(OUT / f"confusion_{name}.png", dpi=150)
    plt.close(fig)

summary_df = pd.DataFrame(summary)
class_df = pd.DataFrame(class_rows)

summary_df.to_csv(OUT / "condition_metrics.csv", index=False)
class_df.to_csv(OUT / "classwise_metrics.csv", index=False)

print("\n=== CONDITION METRICS ===")
print(summary_df.round(4).to_string(index=False))

print("\n=== CLASS-WISE METRICS ===")
print(class_df.round(4).to_string(index=False))

print(f"\nResults saved to: {OUT}")
