"""Figures derived from saved evaluation predictions and training histories."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
TAB = ROOT / "results" / "tables"
OUT = ROOT / "report" / "figures" / "plots" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

IDS = ["E1_v4_btp_p30", "E7_ft_rcl_btp_p30", "E7_star_edge_btp_p30"]
NAME = dict(zip(IDS, ["Baseline (M0)", "FT-RCL (M3)", "STAR-Edge"]))
COL = dict(zip(IDS, ["#0072B2", "#D55E00", "#009E73"]))
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300})


def curves():
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.1))
    for i in IDS:
        h = pd.read_csv(EXP / i / "training_history.csv")
        for a, c, t in zip(ax, ["dev_macro_f1", "dev_eer", "train_acc"], ["Validation Macro-F1", "Validation EER", "Training accuracy"]):
            a.plot(h["epoch"], h[c], color=COL[i], label=NAME[i], lw=1.4, marker="o", ms=2.5)
            a.set_title(t, fontsize=10)
            a.set_xlabel("Epoch")
    ax[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "curves.png", bbox_inches="tight")


def per_type():
    t = pd.read_csv(TAB / "analysis" / "per_type.csv")
    types = ["speech", "singing", "music", "sound"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.3))
    w = 0.26
    for ax_, col, ttl in zip(ax, ["eer_own_threshold", "macro_f1_global_thr"], ["EER per type (own threshold)", "Macro-F1 per type (one global threshold)"]):
        for j, i in enumerate(IDS):
            d = t[t.model == i].set_index("type").loc[types]
            ax_.bar(np.arange(4) + (j - 1) * w, d[col], w, color=COL[i], label=NAME[i])
        ax_.set_xticks(range(4))
        ax_.set_xticklabels(types)
        ax_.set_title(ttl, fontsize=10)
    ax[0].axhline(0.5, color="#888", lw=0.8, ls="--")
    ax[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "per_type.png", bbox_inches="tight")


def generators():
    g = pd.read_csv(TAB / "analysis" / "per_generator.csv")
    a, b, c = (g[g.model == i].set_index("generator") for i in IDS)
    order = a.sort_values("recall_global_thr").index
    fig, ax = plt.subplots(figsize=(6.4, 8.2))
    y = np.arange(len(order))
    for k, (d, i) in enumerate(zip((a, b, c), IDS)):
        ax.barh(y + (k - 1) * 0.27, d.loc[order, "recall_global_thr"], 0.27, color=COL[i], label=NAME[i])
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=6.5)
    ax.set_xlabel("Fake recall at the pooled EER threshold")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.savefig(OUT / "generators.png", bbox_inches="tight")


def calibration():
    fig, ax = plt.subplots(figsize=(4.4, 4))
    edges = np.linspace(0, 1, 11)
    for i in IDS:
        d = pd.read_csv(EXP / i / "eval_predictions_eval.csv")
        idx = np.clip(np.digitize(d["score"], edges) - 1, 0, 9)
        xs = [d["score"][idx == b].mean() for b in range(10) if (idx == b).any()]
        ys = [d["label"][idx == b].mean() for b in range(10) if (idx == b).any()]
        ax.plot(xs, ys, marker="o", ms=3, color=COL[i], label=NAME[i])
    ax.plot([0, 1], [0, 1], "--", color="#999", lw=0.8)
    ax.set_xlabel("Mean predicted fake probability")
    ax.set_ylabel("Observed fake fraction")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(OUT / "calibration.png", bbox_inches="tight")


def shortcuts():
    d = pd.read_csv(TAB / "final" / "shortcut_probe.csv")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True)
    cues = ["duration", "native_sample_rate", "loudness", "silence", "bandwidth", "all_cues"]
    for a, scope in zip(ax, ["all", "music"]):
        s = d[d.scope == scope].set_index("cue").loc[cues]
        a.barh(cues, s["auroc"], color="#56B4E9")
        a.axvline(0.5, color="#333", lw=0.8, ls="--")
        a.set_xlim(0.2, 0.8)
        a.set_title(f"Scope: {scope}", fontsize=10)
        a.set_xlabel("AUROC of a cue-only classifier")
    fig.tight_layout()
    fig.savefig(OUT / "shortcuts.png", bbox_inches="tight")


def dev_vs_eval():
    rows = []
    for i in IDS:
        h = pd.read_csv(EXP / i / "training_history.csv")
        dev = h.loc[h["best_J"].idxmax(), "dev_macro_f1"]
        ev = pd.read_csv(EXP / i / "eval_predictions_eval.csv")
        rows.append((NAME[i], dev, float(pd.read_json(EXP / i / "eval_metrics_eval.json", typ="series")["macro_f1"])))
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    x = np.arange(len(rows))
    ax.bar(x - 0.18, [r[1] for r in rows], 0.36, color="#999999", label="Official development split")
    ax.bar(x + 0.18, [r[2] for r in rows], 0.36, color="#0072B2", label="Evaluation (unseen generators)")
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows])
    ax.set_ylabel("Macro-F1")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(OUT / "dev_vs_eval.png", bbox_inches="tight")


if __name__ == "__main__":
    for f in (curves, per_type, generators, calibration, shortcuts, dev_vs_eval):
        f()
    print("ok")
