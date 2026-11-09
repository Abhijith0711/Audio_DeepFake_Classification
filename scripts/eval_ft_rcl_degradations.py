"""
Evaluate FT_RCL checkpoint (checkpoints/FT_RCL/best.pt) on the full official evaluation set (44,289 clips)
under clean audio and 5 specific degradations from Bhuvan's degradation library:
1. Band-limit 300–3400 Hz (telephony)
2. Resample via 16 kHz (24kHz -> 16kHz -> 24kHz)
3. Noise, SNR 20 dB (additive Gaussian noise)
4. Noise, SNR 10 dB (additive Gaussian noise)
5. Noise, SNR 5 dB (additive Gaussian noise)

Outputs:
- Fixed-threshold (0.50) Macro-F1, Accuracy, AUROC, EER, and Delta F1
- Calibrated-threshold (theta*) Macro-F1, Accuracy, and Delta F1
- Type-averaged Macro-F1 (Speech, Sound, Singing, Music) and Delta F1
- Domain-level breakdown table
"""

import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.eval_runner import build_model_from_checkpoint
from src.preprocessing.preprocess_atadd_rerun import (
    GPULogMel,
    cpu_prepare_waveform,
    normalize_log_mel,
)
from src.robustness.conditions import ConditionSpec
from src.robustness.degrade import degrade

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Target checkpoint & paths
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints/FT_RCL/best.pt"
EVAL_MANIFEST_PATH = PROJECT_ROOT / "data/manifests/AT_ADDReworked/eval_manifest.csv"
OUTPUT_DIR = PROJECT_ROOT / "results/tables/FT_RCL"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Spectrogram and audio parameters
SAMPLE_RATE = 24000
SEGMENT_SECONDS = 4
NUM_SAMPLES = SAMPLE_RATE * SEGMENT_SECONDS
N_FFT = 1024
WIN_LENGTH = 600
HOP_LENGTH = 240
N_MELS = 80
LOG_EPS = 1e-10

DEFAULT_THRESH = 0.50

LABEL_MAP = {"real": 0, "fake": 1}

# 5 requested degradation conditions
DEGRADATION_SPECS = {
    "clean": None,
    "bandlimit_300_3400": ConditionSpec(
        family="telephony",
        name="bandlimit_300_3400",
        severity=3,
        params={"low_hz": 300, "high_hz": 3400},
    ),
    "down_up_16k": ConditionSpec(
        family="resample",
        name="down_up_16k",
        severity=1,
        params={"intermediate_sr": 16000},
    ),
    "snr_20db": ConditionSpec(
        family="additive_noise",
        name="snr_20db",
        severity=2,
        params={"snr_db": 20},
    ),
    "snr_10db": ConditionSpec(
        family="additive_noise",
        name="snr_10db",
        severity=3,
        params={"snr_db": 10},
    ),
    "snr_5db": ConditionSpec(
        family="additive_noise",
        name="snr_5db",
        severity=4,
        params={"snr_db": 5},
    ),
}


def audio_path_from_row(row):
    folder = row["folder"] if "folder" in row and pd.notna(row["folder"]) else row["split"]
    return PROJECT_ROOT / "data/raw/AT_ADD" / str(folder) / str(row["name"])


class MultiConditionDataset(Dataset):
    def __init__(self, df):
        self.rows = df.to_dict("records")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        file_id = str(row["name"])
        audio_path = audio_path_from_row(row)

        clean_w = cpu_prepare_waveform(
            audio_path, sample_rate=SAMPLE_RATE, segment_seconds=SEGMENT_SECONDS, training=False
        ).numpy()

        waveforms = {"clean": torch.from_numpy(clean_w).float()}
        for name, spec in DEGRADATION_SPECS.items():
            if spec is not None:
                deg_w, _ = degrade(clean_w, SAMPLE_RATE, spec, mode="deterministic", source_id=file_id)
                waveforms[name] = torch.from_numpy(deg_w).float()

        label = LABEL_MAP[row["label"]]
        audio_type = str(row.get("type", "unknown"))
        return waveforms, label, file_id, audio_type


def compute_eer(y_true, scores):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    thresh = float(thresholds[idx])
    return eer, thresh


def compute_metrics(y_true, scores, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    preds = (scores >= threshold).astype(int)
    f1 = float(f1_score(y_true, preds, average="macro", zero_division=0))
    acc = float(accuracy_score(y_true, preds))
    auroc = float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) == 2 else float("nan")
    eer, eer_thresh = compute_eer(y_true, scores)
    return {
        "macro_f1": f1,
        "accuracy": acc,
        "auroc": auroc,
        "eer": eer,
        "eer_threshold": eer_thresh,
    }


def find_best_threshold(y_true, scores):
    best_f1 = -1.0
    best_th = 0.50
    for th in np.arange(0.01, 1.00, 0.01):
        preds = (scores >= th).astype(int)
        f1 = f1_score(y_true, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
    return float(best_th), float(best_f1)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # 1. Load model
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model, mode = build_model_from_checkpoint(ckpt, device)
    model.eval()
    logger.info("Loaded FT-RCL model (mode: %s) from epoch %s", mode, ckpt.get("epoch"))

    # Normalization parameters from checkpoint
    mel_mean = float(ckpt.get("mel_mean", -9.191030070182308))
    mel_std = float(ckpt.get("mel_std", 6.850966223389033))
    logger.info("Using Mel normalization: mean=%.6f, std=%.6f", mel_mean, mel_std)

    # 2. Frontend
    gpu_logmel = GPULogMel(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        log_eps=LOG_EPS,
    ).to(device)

    # 3. Load dataset
    df = pd.read_csv(EVAL_MANIFEST_PATH)
    total_clips = len(df)
    logger.info("Loaded evaluation manifest with %d clips across types: %s", total_clips, df["type"].value_counts().to_dict())

    dataset = MultiConditionDataset(df)
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=8,
        pin_memory=(device.type == "cuda"),
        prefetch_factor=2,
    )

    scores_by_cond = {name: [] for name in DEGRADATION_SPECS}
    all_labels = []
    all_file_ids = []
    all_types = []

    logger.info("Starting FT-RCL evaluation across 6 conditions (clean + 5 degradations)...")
    t0 = time.time()
    with torch.no_grad():
        for batch_waves, batch_labels, batch_ids, batch_types in tqdm(loader, desc="Evaluating FT-RCL"):
            all_labels.extend(batch_labels.numpy().tolist())
            all_file_ids.extend(batch_ids)
            all_types.extend(batch_types)

            for cond_name, w in batch_waves.items():
                w = w.to(device, non_blocking=True)
                mel = gpu_logmel(w)
                feats = normalize_log_mel(mel, mel_mean, mel_std).unsqueeze(1)
                with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                    out = model(feats)
                    logits = out[0] if isinstance(out, tuple) else out
                cond_scores = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy().tolist()
                scores_by_cond[cond_name].extend(cond_scores)

    elapsed = time.time() - t0
    logger.info("Inference completed in %.2f seconds (%.1f clips/s)", elapsed, total_clips / elapsed)

    y_arr = np.array(all_labels)
    types_arr = np.array(all_types)
    domains = ["speech", "sound", "singing", "music"]

    # 4. Save detailed predictions dataframe
    preds_df = pd.DataFrame({
        "file_id": all_file_ids,
        "type": all_types,
        "label": all_labels,
    })
    for cond_name in DEGRADATION_SPECS:
        preds_df[f"score_{cond_name}"] = scores_by_cond[cond_name]
    preds_csv_path = OUTPUT_DIR / "FT_RCL_degradations_predictions.csv"
    preds_df.to_csv(preds_csv_path, index=False)
    logger.info("Saved prediction scores to %s", preds_csv_path)

    # 5. Compute overall and per-domain metrics
    clean_scores = np.array(scores_by_cond["clean"])
    
    # Determine calibrated threshold on clean evaluation scores
    calibrated_th, best_clean_f1 = find_best_threshold(y_arr, clean_scores)
    logger.info("Threshold optimization on clean: best theta* = %.2f (Clean Macro-F1: %.4f)", calibrated_th, best_clean_f1)

    clean_overall_05 = compute_metrics(y_arr, clean_scores, threshold=DEFAULT_THRESH)
    clean_overall_opt = compute_metrics(y_arr, clean_scores, threshold=calibrated_th)

    # Clean domain metrics
    clean_domain_f1_05 = {}
    clean_domain_f1_opt = {}
    for d in domains:
        mask = (types_arr == d)
        m_05 = compute_metrics(y_arr[mask], clean_scores[mask], threshold=DEFAULT_THRESH)
        m_opt = compute_metrics(y_arr[mask], clean_scores[mask], threshold=calibrated_th)
        clean_domain_f1_05[d] = m_05["macro_f1"]
        clean_domain_f1_opt[d] = m_opt["macro_f1"]

    clean_type_avg_05 = float(np.mean(list(clean_domain_f1_05.values())))
    clean_type_avg_opt = float(np.mean(list(clean_domain_f1_opt.values())))

    logger.info("\n" + "="*80)
    logger.info("OVERALL CLEAN BASELINE PERFORMANCE (FT-RCL, N=44,289)")
    logger.info("="*80)
    logger.info("Threshold tau=0.50 : Pooled Macro-F1 = %.4f | Type-Avg Macro-F1 = %.4f | Acc = %.4f | AUROC = %.4f | EER = %.2f%%",
                clean_overall_05["macro_f1"], clean_type_avg_05, clean_overall_05["accuracy"], clean_overall_05["auroc"], clean_overall_05["eer"]*100)
    logger.info("Threshold theta*=%.2f: Pooled Macro-F1 = %.4f | Type-Avg Macro-F1 = %.4f | Acc = %.4f",
                calibrated_th, clean_overall_opt["macro_f1"], clean_type_avg_opt, clean_overall_opt["accuracy"])

    overall_rows = []
    domain_rows = []
    detailed_metrics = {
        "calibrated_threshold": calibrated_th,
    }

    for cond_name, spec in DEGRADATION_SPECS.items():
        s_arr = np.array(scores_by_cond[cond_name])
        m_05 = compute_metrics(y_arr, s_arr, threshold=DEFAULT_THRESH)
        m_opt = compute_metrics(y_arr, s_arr, threshold=calibrated_th)

        # Domain breakdown for this condition
        cond_domain_f1_05 = {}
        cond_domain_f1_opt = {}
        for d in domains:
            mask = (types_arr == d)
            d_m05 = compute_metrics(y_arr[mask], s_arr[mask], threshold=DEFAULT_THRESH)
            d_mopt = compute_metrics(y_arr[mask], s_arr[mask], threshold=calibrated_th)
            cond_domain_f1_05[d] = d_m05["macro_f1"]
            cond_domain_f1_opt[d] = d_mopt["macro_f1"]

            domain_rows.append({
                "condition": cond_name,
                "domain": d,
                "n_samples": int(mask.sum()),
                "f1_clean_0.50": clean_domain_f1_05[d],
                "f1_deg_0.50": d_m05["macro_f1"],
                "delta_f1_0.50": clean_domain_f1_05[d] - d_m05["macro_f1"],
                "rel_drop_pct_0.50": ((clean_domain_f1_05[d] - d_m05["macro_f1"]) / clean_domain_f1_05[d]) * 100 if clean_domain_f1_05[d] > 0 else 0.0,
                "f1_clean_opt": clean_domain_f1_opt[d],
                "f1_deg_opt": d_mopt["macro_f1"],
                "delta_f1_opt": clean_domain_f1_opt[d] - d_mopt["macro_f1"],
                "rel_drop_pct_opt": ((clean_domain_f1_opt[d] - d_mopt["macro_f1"]) / clean_domain_f1_opt[d]) * 100 if clean_domain_f1_opt[d] > 0 else 0.0,
                "auroc": d_m05["auroc"],
                "eer_pct": d_m05["eer"] * 100,
            })

        type_avg_05 = float(np.mean(list(cond_domain_f1_05.values())))
        type_avg_opt = float(np.mean(list(cond_domain_f1_opt.values())))

        # Delta F1 = Clean F1 - Conditioned F1
        delta_pooled_05 = clean_overall_05["macro_f1"] - m_05["macro_f1"]
        delta_pooled_opt = clean_overall_opt["macro_f1"] - m_opt["macro_f1"]
        delta_type_avg_05 = clean_type_avg_05 - type_avg_05
        delta_type_avg_opt = clean_type_avg_opt - type_avg_opt

        overall_rows.append({
            "condition": cond_name,
            "family": spec.family if spec else "identity",
            "severity": spec.severity if spec else 0,
            # Pooled F1 (tau = 0.50)
            "pooled_f1_clean_0.50": clean_overall_05["macro_f1"],
            "pooled_f1_deg_0.50": m_05["macro_f1"],
            "delta_pooled_f1_0.50": delta_pooled_05,
            "rel_drop_pooled_0.50_pct": (delta_pooled_05 / clean_overall_05["macro_f1"]) * 100,
            # Type-Averaged F1 (tau = 0.50)
            "type_avg_f1_clean_0.50": clean_type_avg_05,
            "type_avg_f1_deg_0.50": type_avg_05,
            "delta_type_avg_f1_0.50": delta_type_avg_05,
            "rel_drop_type_avg_0.50_pct": (delta_type_avg_05 / clean_type_avg_05) * 100,
            # Pooled F1 (theta*)
            "pooled_f1_clean_opt": clean_overall_opt["macro_f1"],
            "pooled_f1_deg_opt": m_opt["macro_f1"],
            "delta_pooled_f1_opt": delta_pooled_opt,
            "rel_drop_pooled_opt_pct": (delta_pooled_opt / clean_overall_opt["macro_f1"]) * 100,
            # Type-Averaged F1 (theta*)
            "type_avg_f1_clean_opt": clean_type_avg_opt,
            "type_avg_f1_deg_opt": type_avg_opt,
            "delta_type_avg_f1_opt": delta_type_avg_opt,
            "rel_drop_type_avg_opt_pct": (delta_type_avg_opt / clean_type_avg_opt) * 100,
            # Accuracy, AUROC, EER
            "accuracy_clean_0.50": clean_overall_05["accuracy"],
            "accuracy_deg_0.50": m_05["accuracy"],
            "auroc_clean": clean_overall_05["auroc"],
            "auroc_deg": m_05["auroc"],
            "eer_clean_pct": clean_overall_05["eer"] * 100,
            "eer_deg_pct": m_05["eer"] * 100,
        })

        detailed_metrics[cond_name] = {
            "metrics_0.50": m_05,
            "metrics_opt": m_opt,
            "type_avg_f1_0.50": type_avg_05,
            "type_avg_f1_opt": type_avg_opt,
            "domain_f1_0.50": cond_domain_f1_05,
            "domain_f1_opt": cond_domain_f1_opt,
        }

    summary_df = pd.DataFrame(overall_rows)
    summary_csv_path = OUTPUT_DIR / "FT_RCL_degradations_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    logger.info("Saved summary table to %s", summary_csv_path)

    domain_df = pd.DataFrame(domain_rows)
    domain_csv_path = OUTPUT_DIR / "FT_RCL_degradations_by_domain.csv"
    domain_df.to_csv(domain_csv_path, index=False)
    logger.info("Saved domain breakdown to %s", domain_csv_path)

    metrics_json_path = OUTPUT_DIR / "FT_RCL_degradations_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(detailed_metrics, f, indent=2)
    logger.info("Saved detailed metrics JSON to %s", metrics_json_path)

    # 6. Print clean formatted comparison table
    print("\n" + "="*115)
    print("ROBUSTNESS EVALUATION: FT-RCL (M3) under 5 Degradations (Full Eval, N=44,289)")
    print("="*115)
    header = f"{'Condition':<22} | {'Clean F1':<10} | {'Deg F1':<10} | {'Delta F1':<10} | {'Drop (%)':<10} | {'TypeAvg Clean':<13} | {'TypeAvg Deg':<11} | {'Delta TypeAvg':<13}"
    print(header)
    print("-" * len(header))
    for r in overall_rows:
        print(f"{r['condition']:<22} | {r['pooled_f1_clean_0.50']:<10.4f} | {r['pooled_f1_deg_0.50']:<10.4f} | {r['delta_pooled_f1_0.50']:<+10.4f} | {r['rel_drop_pooled_0.50_pct']:<10.2f}% | {r['type_avg_f1_clean_0.50']:<13.4f} | {r['type_avg_f1_deg_0.50']:<11.4f} | {r['delta_type_avg_f1_0.50']:<+13.4f}")

    print("\n" + "="*115)
    print(f"CALIBRATED THRESHOLD EVALUATION (theta* = {calibrated_th:.2f})")
    print("="*115)
    header_cal = f"{'Condition':<22} | {'Clean F1*':<10} | {'Deg F1*':<10} | {'Delta F1*':<10} | {'Drop (%)*':<10} | {'TypeAvg Cln*':<13} | {'TypeAvg Deg*':<12} | {'Delta TypeAvg*':<14}"
    print(header_cal)
    print("-" * len(header_cal))
    for r in overall_rows:
        print(f"{r['condition']:<22} | {r['pooled_f1_clean_opt']:<10.4f} | {r['pooled_f1_deg_opt']:<10.4f} | {r['delta_pooled_f1_opt']:<+10.4f} | {r['rel_drop_pooled_opt_pct']:<10.2f}% | {r['type_avg_f1_clean_opt']:<13.4f} | {r['type_avg_f1_deg_opt']:<12.4f} | {r['delta_type_avg_f1_opt']:<+14.4f}")

    print("\n" + "="*115)
    print("PER-DOMAIN MACRO-F1 BREAKDOWN (Threshold tau = 0.50)")
    print("="*115)
    print(f"{'Condition':<22} | {'Speech (27k)':<14} | {'Sound (6k)':<14} | {'Singing (6.7k)':<16} | {'Music (4.5k)':<14}")
    print("-" * 88)
    for cond_name in DEGRADATION_SPECS:
        sub = domain_df[domain_df["condition"] == cond_name].set_index("domain")
        sp = sub.loc["speech", "f1_deg_0.50"]
        so = sub.loc["sound", "f1_deg_0.50"]
        si = sub.loc["singing", "f1_deg_0.50"]
        mu = sub.loc["music", "f1_deg_0.50"]
        print(f"{cond_name:<22} | {sp:<14.4f} | {so:<14.4f} | {si:<16.4f} | {mu:<14.4f}")
    print("="*115)


if __name__ == "__main__":
    main()

