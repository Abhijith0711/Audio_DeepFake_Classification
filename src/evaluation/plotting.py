import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve


def save_training_curves(history_df, out_dir, experiment_id):
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history_df["epoch"], history_df["train_loss"], label="train_loss")
    if "train_acc" in history_df.columns:
        ax.plot(history_df["epoch"], history_df["train_acc"], label="train_acc")
    ax.set_xlabel("epoch")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "training_loss_accuracy.png", dpi=200)
    plt.close(fig)

    if "dev_macro_f1" in history_df.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(history_df["epoch"], history_df["dev_macro_f1"], label="dev_macro_f1")
        if "dev_auroc" in history_df.columns:
            ax.plot(history_df["epoch"], history_df["dev_auroc"], label="dev_auroc")
        ax.set_xlabel("epoch")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "dev_macro_f1_auroc.png", dpi=200)
        plt.close(fig)

    if "dev_eer" in history_df.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(history_df["epoch"], history_df["dev_eer"], label="dev_eer")
        ax.set_xlabel("epoch")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "dev_eer.png", dpi=200)
        plt.close(fig)


def save_roc(y_true, scores, path):
    fpr, tpr, _ = roc_curve(y_true, scores)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_confusion(y_true, predictions, path):
    cm = confusion_matrix(y_true, predictions)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_delta_m_heatmap(delta_df, path):
    if delta_df.empty:
        return

    pivot = delta_df.pivot_table(index="type", columns="condition_name", values="delta_macro_f1", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(pivot, annot=True, fmt=".3f", ax=ax)
    ax.set_title("Delta Macro-F1 (clean - conditioned)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
