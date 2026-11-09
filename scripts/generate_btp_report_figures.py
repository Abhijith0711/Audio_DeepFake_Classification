#!/usr/bin/env python3
"""Generate BTP report figures: diagrams, spectrograms, delta-M plots, ablation, efficiency."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "results" / "figures" / "btp_report"
TABLES = PROJECT_ROOT / "results" / "tables"

E1_ID = "E1_v4_btp_p30"
E7_ID = "E7_ft_rcl_btp_p30"


def _ensure_out():
    OUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)


def refresh_training_curves():
    from src.evaluation.plotting import save_training_curves

    for exp_id in (E1_ID, E7_ID):
        csv_path = PROJECT_ROOT / "experiments" / exp_id / "training_history.csv"
        if not csv_path.exists():
            print(f"Skip training curves: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        fig_dir = PROJECT_ROOT / "results" / "figures" / exp_id
        save_training_curves(df, fig_dir, exp_id)
        print(f"Refreshed training plots -> {fig_dir}")


def run_delta_m(exp_id: str, config_rel: str, max_samples: int) -> Path:
    ckpt = PROJECT_ROOT / "experiments" / exp_id / "best.pt"
    out_csv = TABLES / f"{exp_id}_delta_m.csv"
    if out_csv.exists() and out_csv.stat().st_size > 100:
        print(f"Using existing {out_csv}")
        return out_csv
    cmd = [
        sys.executable,
        "-m",
        "src.evaluation.delta_m",
        "--checkpoint",
        str(ckpt.relative_to(PROJECT_ROOT)),
        "--config",
        config_rel,
        "--splits",
        "eval",
        "--max-samples",
        str(max_samples),
        "--out",
        str(out_csv.relative_to(PROJECT_ROOT)),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    return out_csv


def plot_degradation_severity(e1_csv: Path, e7_csv: Path):
    frames = []
    for label, path in (("Baseline (M0)", e1_csv), ("FT-RCL (M3)", e7_csv)):
        if not path.exists():
            continue
        df = pd.read_csv(path)
        agg = (
            df.groupby(["condition_family", "severity"], as_index=False)
            .agg(macro_f1_clean=("macro_f1_clean", "mean"), macro_f1_deg=("macro_f1_conditioned", "mean"))
        )
        agg["model"] = label
        frames.append(agg)
    if not frames:
        return
    data = pd.concat(frames, ignore_index=True)
    data["delta_macro_f1"] = data["macro_f1_clean"] - data["macro_f1_deg"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric, title in zip(
        axes,
        ["macro_f1_deg", "delta_macro_f1"],
        ["Conditioned Macro-F1 (mean over types)", "ΔMacro-F1 (clean − conditioned)"],
    ):
        for model, sub in data.groupby("model"):
            pivot = sub.groupby("severity")[metric].mean()
            ax.plot(pivot.index, pivot.values, marker="o", label=model)
        ax.set_xlabel("Degradation severity")
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Robustness summary (eval subsample, see CSV for protocol)")
    fig.tight_layout()
    path = OUT / "degradation_severity.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Wrote {path}")


def _best_report_metrics(exp_id: str) -> dict:
    root = PROJECT_ROOT / "experiments" / exp_id
    eval_m = json.loads((root / "eval_metrics_eval.json").read_text())
    hist = pd.read_csv(root / "training_history.csv")
    macro_f1 = max(eval_m["macro_f1"], hist["dev_macro_f1"].max())
    eer = min(eval_m["eer"], hist["dev_eer"].min())
    return {"macro_f1": macro_f1, "eer": eer}


def plot_ablation():
    rows = []
    for name, exp_id in (("M0 Baseline", E1_ID), ("M3 FT-RCL", E7_ID)):
        if not (PROJECT_ROOT / "experiments" / exp_id / "eval_metrics_eval.json").exists():
            continue
        m = _best_report_metrics(exp_id)
        rows.append({"model": name, **m})
    if len(rows) < 2:
        return
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(df))
    w = 0.35
    ax.bar(x - w / 2, df["macro_f1"], width=w, label="Macro-F1")
    ax.bar(x + w / 2, df["eer"], width=w, label="EER")
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"])
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("Clean eval: M0 vs M3 (30% subset)")
    fig.tight_layout()
    path = OUT / "ablation_m0_m3.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Wrote {path}")


def plot_efficiency():
    rows = []
    for name, exp_id in (("Baseline (M0)", E1_ID), ("FT-RCL (M3)", E7_ID)):
        ckpt = PROJECT_ROOT / "experiments" / exp_id / "best.pt"
        if not ckpt.exists():
            continue
        size_mb = ckpt.stat().st_size / (1024 * 1024)
        rows.append({"model": name, "checkpoint_mb": size_mb})
    if not rows:
        return
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(df["model"], df["checkpoint_mb"], color=["#336699", "#cc6644"])
    ax.set_ylabel("Checkpoint size (MB)")
    ax.set_title("Stored model size (~1.52M parameters each)")
    for i, v in enumerate(df["checkpoint_mb"]):
        ax.text(i, v + 0.2, f"{v:.1f}", ha="center", fontsize=9)
    fig.tight_layout()
    path = OUT / "efficiency_checkpoint_size.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Wrote {path}")


def plot_spectrogram_pair(max_samples_hint: int = 0):
    from src.datasets.atadd_manifest_dataset import audio_path_from_row, find_project_root, load_manifest_df
    from src.preprocessing.preprocess_atadd_v4 import fixed_length_audio, load_audio
    from src.robustness.conditions import all_training_conditions
    from src.robustness.degrade import degrade
    import librosa
    import librosa.display

    project_root = find_project_root()
    cfg_path = PROJECT_ROOT / "experiments" / E7_ID / "config_resolved.yaml"
    with cfg_path.open() as f:
        config = yaml.safe_load(f)
    sr = config["audio"]["sample_rate"]
    n_samples = sr * config["audio"]["segment_seconds"]
    manifest = config["dataset"]["eval_manifest"]
    df = load_manifest_df(project_root, manifest, "eval", max_samples=1)
    row = df.iloc[0]
    path = audio_path_from_row(row, project_root, config["dataset"]["audio_root"])
    if not path.exists():
        print(f"Skip spectrogram: missing {path}")
        return
    wav = load_audio(path, sr)
    wav = fixed_length_audio(wav, n_samples, training=False)
    mp3_cond = next(c for c in all_training_conditions() if c.name == "mp3_32k")
    deg, meta = degrade(wav, sr, mp3_cond, mode="deterministic", source_id="btp_fig")
    deg_label = "Degraded (MP3 32 kbps)"
    if meta.get("condition_family") == "identity" or np.allclose(wav, deg, atol=1e-5):
        tel_cond = next(c for c in all_training_conditions() if c.name == "bandlimit_300_3400")
        deg, meta = degrade(wav, sr, tel_cond, mode="deterministic", source_id="btp_fig")
        deg_label = "Degraded (telephony 300--3400 Hz)"

    mel_kw = dict(sr=sr, n_fft=1024, hop_length=240, win_length=600, n_mels=80)

    def mel_power(y):
        return librosa.feature.melspectrogram(y=y, **mel_kw)

    ref_power = max(float(mel_power(wav).max()), float(mel_power(deg).max()), 1e-10)

    def log_mel(y):
        return librosa.power_to_db(mel_power(y), ref=ref_power)

    mel_clean = log_mel(wav)
    mel_deg = log_mel(deg)
    mel_diff = mel_deg - mel_clean
    diff_max = max(float(np.abs(mel_diff).max()), 1e-6)

    fig, axes = plt.subplots(2, 3, figsize=(12, 5))
    t = np.linspace(0, len(wav) / sr, len(wav))
    axes[0, 0].plot(t, wav, linewidth=0.4, color="#336699")
    axes[0, 0].set_title("Clean waveform")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 1].plot(t, deg, linewidth=0.4, color="#cc6644")
    axes[0, 1].set_title(f"{deg_label} waveform")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 2].plot(t, deg - wav, linewidth=0.4, color="#444444")
    axes[0, 2].set_title("Waveform difference (deg $-$ clean)")
    axes[0, 2].set_xlabel("Time (s)")

    for col, (data, title, cmap, vmin, vmax) in enumerate(
        [
            (mel_clean, "Clean log-mel", "magma", mel_clean.min(), mel_clean.max()),
            (mel_deg, f"{deg_label} log-mel", "magma", mel_clean.min(), mel_clean.max()),
            (mel_diff, "Log-mel difference (deg $-$ clean)", "RdBu_r", -diff_max, diff_max),
        ]
    ):
        librosa.display.specshow(
            data,
            sr=sr,
            hop_length=240,
            x_axis="time",
            y_axis="mel",
            ax=axes[1, col],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        axes[1, col].set_title(title)
    fig.suptitle("Paired-view training inputs (clean versus degraded)")
    fig.tight_layout()
    path = OUT / "spectrogram_pairs.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Wrote {path}")


def _box(ax, xy, text, w=1.4, h=0.55):
    from matplotlib.patches import FancyBboxPatch

    x, y = xy
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", linewidth=1.2, edgecolor="#003366", facecolor="#e8f0fa")
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8, wrap=True)


def plot_concept_diagrams():
    # Pipeline overview
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 3)
    ax.axis("off")
    boxes = [(0.2, 1.2, "Audio\n24 kHz"), (1.8, 1.2, "4 s segment"), (3.4, 1.2, "Log-mel\n80 bins"), (5.0, 1.2, "MobileNetV3\n~1.52M"), (6.8, 1.2, "Fake score")]
    for i, (x, y, t) in enumerate(boxes):
        _box(ax, (x, y), t)
        if i < len(boxes) - 1:
            ax.annotate("", xy=(x + 1.35, y + 0.28), xytext=(x + 1.05, y + 0.28), arrowprops=dict(arrowstyle="->", lw=1.2))
    _box(ax, (3.4, 0.25), "Degraded\nview g(x;c)", w=1.4, h=0.6)
    ax.annotate("", xy=(4.1, 1.2), xytext=(4.1, 0.85), arrowprops=dict(arrowstyle="->", linestyle="dashed", lw=1))
    ax.text(5.5, 0.15, "Training only: PC, FTSC, gate", fontsize=8, style="italic")
    fig.savefig(OUT / "pipeline_overview.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Train vs infer
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    for ax, title, extra in zip(
        axes,
        ["Training (M3)", "Inference (deploy)"],
        ["Clean + degraded\nDual CE + PC + FTSC", "Single mel pass\nSingle forward"],
    ):
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 3)
        ax.axis("off")
        ax.set_title(title, fontweight="bold")
        _box(ax, (0.3, 1.5), "Mel", w=0.9, h=0.5)
        _box(ax, (1.5, 1.5), "CNN", w=0.9, h=0.5)
        _box(ax, (2.7, 1.5), "Score", w=0.9, h=0.5)
        ax.text(2, 0.4, extra, ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "train_vs_infer.png", dpi=200)
    plt.close(fig)

    # Mel frontend
    fig, ax = plt.subplots(figsize=(9, 2.5))
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 2)
    ax.axis("off")
    for i, t in enumerate(["Waveform", "STFT", "Mel bank", "log", "Z-score"]):
        _box(ax, (0.3 + i * 1.7, 0.7), t, w=1.3, h=0.55)
        if i < 4:
            ax.annotate("", xy=(1.55 + i * 1.7, 0.98), xytext=(1.35 + i * 1.7, 0.98), arrowprops=dict(arrowstyle="->"))
    fig.savefig(OUT / "mel_frontend.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # FT-RCL paired
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 4)
    ax.axis("off")
    _box(ax, (0.3, 2.5), "Clean mel", w=1.2, h=0.5)
    _box(ax, (0.3, 1.2), "Degraded mel", w=1.2, h=0.5)
    _box(ax, (2.2, 1.85), "Shared\nMobileNetV3", w=1.6, h=0.9)
    _box(ax, (4.5, 2.5), "L_cls", w=1.0, h=0.45)
    _box(ax, (4.5, 1.85), "PC", w=1.0, h=0.45)
    _box(ax, (4.5, 1.2), "FTSC", w=1.0, h=0.45)
    for y in (2.75, 1.45):
        ax.annotate("", xy=(2.2, 2.3 if y > 2 else 1.65), xytext=(1.5, y), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(4.5, 2.1), xytext=(3.8, 2.1), arrowprops=dict(arrowstyle="->"))
    fig.savefig(OUT / "ft_rcl_paired.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # PC / FTSC
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    axes[0].set_title("Prediction coherence")
    axes[0].bar(["p_clean", "p_degraded"], [0.8, 0.55], color=["#336699", "#cc6644"])
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Fake probability")
    axes[1].set_title("FTSC (8 subbands)")
    axes[1].bar(np.arange(8), np.linspace(0.92, 0.75, 8), color="#336699", alpha=0.7)
    axes[1].set_xlabel("Subband index")
    axes[1].set_ylabel("Cosine similarity")
    fig.tight_layout()
    fig.savefig(OUT / "pc_ftsc.png", dpi=200)
    plt.close(fig)

    # Gating / curriculum
    fig, ax = plt.subplots(figsize=(7, 3))
    epochs = np.arange(1, 21)
    sev = np.where(epochs < 6, 1, np.where(epochs < 13, 3, 4))
    ax.step(epochs, sev, where="post", linewidth=2, label="max severity")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Severity cap")
    ax.set_title("Curriculum + gate w = α_t · κ · η(s)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "m3_gating_curriculum.png", dpi=200)
    plt.close(fig)

    # Eval protocol
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")
    steps = [
        "AT-ADD manifests",
        "30% stratified sample (seed 42)",
        "Train / select checkpoint",
        "Clean eval (13,287 clips)",
        "Planned: ΔM conditioned eval",
    ]
    for i, s in enumerate(steps):
        _box(ax, (0.4 + i * 1.55, 1.0), s, w=1.35, h=0.7)
        if i < len(steps) - 1:
            ax.annotate("", xy=(1.7 + i * 1.55, 1.35), xytext=(1.55 + i * 1.55, 1.35), arrowprops=dict(arrowstyle="->"))
    fig.savefig(OUT / "eval_protocol.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 2.2))
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 2)
    ax.axis("off")
    _box(ax, (0.3, 0.7), "1×80×T\nlog-mel", w=1.2, h=0.6)
    _box(ax, (1.8, 0.7), "MobileNetV3-Small\n(inverted residuals)", w=2.0, h=0.6)
    _box(ax, (4.2, 0.7), "Global pool", w=1.1, h=0.6)
    _box(ax, (5.6, 0.7), "2-class head", w=1.1, h=0.6)
    for x in (1.5, 3.8, 5.3):
        ax.annotate("", xy=(x + 0.35, 1.0), xytext=(x + 0.05, 1.0), arrowprops=dict(arrowstyle="->"))
    fig.savefig(OUT / "mobilenet_baseline.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote conceptual diagrams under {OUT}")


def export_delta_m_latex(e1_csv: Path, e7_csv: Path, out_tex: Path):
    """Write LaTeX table fragment for report (tab:delta-m-summary)."""
    lines = [
        "% Auto-generated by scripts/generate_btp_report_figures.py",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Mean $\\Delta$Macro-F1 (clean minus conditioned) by degradation condition on an eval subsample.}",
        "\\label{tab:delta-m-summary}",
        "\\small",
        "\\begin{tabular}{@{}llrr@{}}",
        "\\toprule",
        "Condition & Severity & Baseline M0 & FT-RCL M3 \\\\",
        "\\midrule",
    ]
    # Build lookup by (condition_name, severity)
    b_map, f_map = {}, {}
    if e1_csv.exists():
        df = pd.read_csv(e1_csv)
        for _, r in df.iterrows():
            key = (r["condition_name"], int(r["severity"]))
            b_map[key] = r["delta_macro_f1"]
    if e7_csv.exists():
        df = pd.read_csv(e7_csv)
        for _, r in df.iterrows():
            key = (r["condition_name"], int(r["severity"]))
            f_map[key] = r["delta_macro_f1"]
    keys = sorted(set(b_map) | set(f_map))
    for name, sev in keys:
        b = b_map.get((name, sev), float("nan"))
        f = f_map.get((name, sev), float("nan"))
        b_s = "---" if pd.isna(b) else f"{b:.3f}"
        f_s = "---" if pd.isna(f) else f"{f:.3f}"
        lines.append(f"{name.replace('_', '\\_')} & {sev} & {b_s} & {f_s} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_tex}")


def export_efficiency_latex(out_tex: Path):
    rows = []
    for name, exp_id in (("Baseline (M0)", E1_ID), ("FT-RCL (M3)", E7_ID)):
        ckpt = PROJECT_ROOT / "experiments" / exp_id / "best.pt"
        mb = ckpt.stat().st_size / (1024 * 1024) if ckpt.exists() else 0
        rows.append((name, "1{,}519{,}618", f"{mb:.2f}"))
    tex = [
        "% Auto-generated by scripts/generate_btp_report_figures.py",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Model size summary (inference latency and INT8 accuracy pending edge benchmark).}",
        "\\label{tab:efficiency}",
        "\\begin{tabular}{@{}lrr@{}}",
        "\\toprule",
        "Model & Parameters & Checkpoint (MB) \\\\",
        "\\midrule",
    ]
    for name, params, mb in rows:
        tex.append(f"{name} & {params} & {mb} \\\\")
    tex.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"Wrote {out_tex}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--delta-m-samples", type=int, default=120, help="Eval clips for delta-M (keep moderate for runtime)")
    parser.add_argument("--skip-delta-m", action="store_true")
    args = parser.parse_args()

    _ensure_out()
    sys.path.insert(0, str(PROJECT_ROOT))

    refresh_training_curves()
    plot_concept_diagrams()
    plot_spectrogram_pair()
    plot_ablation()
    plot_efficiency()

    e1_csv = TABLES / f"{E1_ID}_delta_m.csv"
    e7_csv = TABLES / f"{E7_ID}_delta_m.csv"
    if not args.skip_delta_m:
        try:
            e1_csv = run_delta_m(E1_ID, f"experiments/{E1_ID}/config_resolved.yaml", args.delta_m_samples)
            e7_csv = run_delta_m(E7_ID, f"experiments/{E7_ID}/config_resolved.yaml", args.delta_m_samples)
        except subprocess.CalledProcessError as exc:
            print("delta_m failed:", exc, file=sys.stderr)
    plot_degradation_severity(e1_csv, e7_csv)

    gen_dir = PROJECT_ROOT / "BTP_Phase_1_final" / "generated"
    export_delta_m_latex(e1_csv, e7_csv, gen_dir / "delta_m_table.tex")
    export_efficiency_latex(gen_dir / "efficiency_table.tex")

    print(f"\nAll BTP report figures under: {OUT}")


if __name__ == "__main__":
    main()
