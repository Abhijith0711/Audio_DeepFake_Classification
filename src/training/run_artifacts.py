import json
import logging
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
import yaml


def git_hash():
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def setup_run_logging(log_dir):
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / "stdout.txt"

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    for handler in list(root.handlers):
        if isinstance(handler, logging.FileHandler) and handler.baseFilename.endswith("stdout.txt"):
            root.removeHandler(handler)

    file_handler = logging.FileHandler(stdout_path)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root.addHandler(file_handler)

    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(stream_handler)

    return stdout_path


def write_run_meta(log_dir, seed, device):
    meta = {
        "start_time": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "seed": seed,
    }

    path = log_dir / "run_meta.json"

    with path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return path


def save_config_resolved(experiment_dir, config):
    path = experiment_dir / "config_resolved.yaml"
    payload = dict(config)
    payload["git_hash"] = git_hash()

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)

    return path


def append_train_log(experiment_dir, row):
    path = experiment_dir / "train_log.jsonl"

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def save_training_history(experiment_dir, history_rows):
    df = pd.DataFrame(history_rows)
    csv_path = experiment_dir / "training_history.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def print_metrics_table(metrics, title="Metrics"):
    df = pd.DataFrame([metrics])
    print(f"\n{title}\n{df.to_string(index=False)}")
