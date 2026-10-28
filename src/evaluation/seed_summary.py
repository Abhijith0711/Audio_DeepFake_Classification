"""Mean and variance of macro-F1 and EER across the fixed reworked seeds."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

BENCHMARK_SEEDS = (("seed1", 42), ("seed2", 1337), ("seed3", 2024))
MODES = ("M0", "M1", "M2", "M3")
METRICS = ("macro_f1", "eer", "type_avg_macro_f1")


def experiment_id(mode, seed_tag):
    return f"E7_{mode}_reworked_{seed_tag}"


def _metric_row(mode, seed_tag, seed, split, metrics):
    return {
        "mode": mode,
        "seed_tag": seed_tag,
        "seed": seed,
        "split": split,
        "macro_f1": metrics.get("macro_f1"),
        "eer": metrics.get("eer"),
        "type_avg_macro_f1": metrics.get("type_avg_macro_f1"),
    }


def _best_dev_metrics(history_path):
    history = pd.read_csv(history_path)
    score_column = "selection_score" if "selection_score" in history.columns else "dev_macro_f1"
    best = history.loc[pd.to_numeric(history[score_column], errors="coerce").idxmax()]
    type_avg = best["dev_type_avg_macro_f1"] if "dev_type_avg_macro_f1" in history.columns else None
    return {
        "macro_f1": float(best["dev_macro_f1"]),
        "eer": float(best["dev_eer"]),
        "type_avg_macro_f1": None if pd.isna(type_avg) else float(type_avg),
    }


def collect_seed_metrics(project_root, modes=MODES, seeds=BENCHMARK_SEEDS):
    root = Path(project_root)
    rows = []

    for mode in modes:
        for seed_tag, seed in seeds:
            experiment_dir = root / "experiments" / experiment_id(mode, seed_tag)
            dev_metrics_path = experiment_dir / "eval_metrics_dev.json"
            history_path = experiment_dir / "training_history.csv"

            if dev_metrics_path.exists():
                with dev_metrics_path.open("r", encoding="utf-8") as handle:
                    dev_metrics = json.load(handle)
                rows.append(_metric_row(mode, seed_tag, seed, "dev", dev_metrics))
            elif history_path.exists():
                rows.append(_metric_row(mode, seed_tag, seed, "dev", _best_dev_metrics(history_path)))

            eval_metrics_path = experiment_dir / "eval_metrics_eval.json"
            if eval_metrics_path.exists():
                with eval_metrics_path.open("r", encoding="utf-8") as handle:
                    eval_metrics = json.load(handle)
                rows.append(_metric_row(mode, seed_tag, seed, "eval", eval_metrics))

    return pd.DataFrame(rows)


def summarize_seed_metrics(runs):
    columns = ["mode", "split", "metric", "n_seeds", "mean", "variance", "std"]
    if runs.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for (mode, split), group in runs.groupby(["mode", "split"], sort=False):
        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(dtype=float)
            if len(values) == 0:
                continue
            variance = float(np.var(values, ddof=1)) if len(values) > 1 else 0.0
            rows.append(
                {
                    "mode": mode,
                    "split": split,
                    "metric": metric,
                    "n_seeds": int(len(values)),
                    "mean": float(np.mean(values)),
                    "variance": variance,
                    "std": float(np.sqrt(variance)),
                }
            )

    return pd.DataFrame(rows, columns=columns)


def write_seed_summary(project_root, out_dir=None):
    root = Path(project_root)
    output_dir = Path(out_dir) if out_dir is not None else root / "results" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = collect_seed_metrics(root)
    summary = summarize_seed_metrics(runs)
    runs_path = output_dir / "e7_reworked_seed_runs.csv"
    summary_path = output_dir / "e7_reworked_seed_summary.csv"
    runs.to_csv(runs_path, index=False)
    summary.to_csv(summary_path, index=False)
    return runs_path, summary_path
