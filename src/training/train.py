import argparse
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.datasets.atadd_manifest_dataset import (
    audio_root_from_config,
    collate_paired_batch,
    find_project_root,
    load_manifest_df,
    make_dataloader,
    resolve_sample_fraction,
)
from src.datasets.paired_robustness_dataset import PairedRobustnessDataset
from src.evaluation.eval_runner import compare_to_e1_baseline, run_split_evaluation
from src.training.frontend import MelFrontend, build_gpu_logmel, load_or_compute_mel_stats
from src.training.run_artifacts import (
    print_metrics_table,
    save_config_resolved,
    setup_run_logging,
    write_run_meta,
)
from src.training.trainer import Trainer

logger = logging.getLogger(__name__)


def load_yaml_config(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_train_loader(
    config,
    project_root,
    mode,
    max_samples,
    audio_root_name,
    sample_fraction,
    sample_seed,
):
    train_df = load_manifest_df(
        project_root,
        config["dataset"]["train_manifest"],
        "train",
        max_samples=max_samples,
        sample_fraction=sample_fraction,
        sample_seed=sample_seed,
    )

    train_cfg = config["training"]

    if mode == "M0":
        return make_dataloader(
            train_df,
            project_root,
            batch_size=train_cfg["batch_size"],
            training=True,
            seed=train_cfg["seed"],
            num_workers=train_cfg.get("num_workers", 4),
            prefetch_factor=train_cfg.get("prefetch_factor", 2),
            pin_memory=train_cfg.get("pin_memory", True),
            sample_rate=config["audio"]["sample_rate"],
            segment_seconds=config["audio"]["segment_seconds"],
            audio_root_name=audio_root_name,
        )

    dataset = PairedRobustnessDataset(
        train_df,
        project_root,
        sample_rate=config["audio"]["sample_rate"],
        segment_seconds=config["audio"]["segment_seconds"],
        training=True,
        max_severity=99,
        seed=train_cfg["seed"],
        audio_root_name=audio_root_name,
    )

    loader_kwargs = {
        "batch_size": train_cfg["batch_size"],
        "shuffle": True,
        "num_workers": train_cfg.get("num_workers", 4),
        "pin_memory": train_cfg.get("pin_memory", True),
        "collate_fn": collate_paired_batch,
    }

    if train_cfg.get("num_workers", 4) > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = train_cfg.get("prefetch_factor", 2)

    return torch.utils.data.DataLoader(dataset, **loader_kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", default="M0", choices=["M0", "M1", "M2", "M3", "clean"])
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--checkpoint", default=None, help="Skip training; load weights for eval only")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--data-fraction",
        type=float,
        default=None,
        help="Use this fraction of each split (train/dev/eval), stratified by label. E.g. 0.3",
    )
    parser.add_argument("--eval-splits", default="eval", help="Comma-separated splits after train")
    args = parser.parse_args()

    mode = "M0" if args.mode == "clean" else args.mode

    project_root = find_project_root()
    config = load_yaml_config(project_root / args.config)

    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs

    sample_fraction = resolve_sample_fraction(config, args.data_fraction)
    sample_seed = config["training"]["seed"]
    if sample_fraction is not None and sample_fraction < 1.0:
        config.setdefault("dataset", {})["sample_fraction"] = sample_fraction
        logger.info(
            "Using %.1f%% of each manifest split (stratified by label, seed=%s)",
            sample_fraction * 100,
            sample_seed,
        )

    experiment_dir = project_root / "experiments" / args.experiment_id
    figure_dir = project_root / "results" / "figures" / args.experiment_id
    log_dir = project_root / "results" / "logs" / args.experiment_id
    for path in (experiment_dir, figure_dir, log_dir):
        path.mkdir(parents=True, exist_ok=True)

    setup_run_logging(log_dir)
    save_config_resolved(experiment_dir, config)

    seed_everything(config["training"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    write_run_meta(log_dir, config["training"]["seed"], device)

    audio_root_name = audio_root_from_config(config)
    train_loader = build_train_loader(
        config,
        project_root,
        mode,
        args.max_samples,
        audio_root_name,
        sample_fraction,
        sample_seed,
    )

    stats_loader = make_dataloader(
        train_loader.dataset.df if hasattr(train_loader.dataset, "df") else load_manifest_df(
            project_root,
            config["dataset"]["train_manifest"],
            "train",
            max_samples=args.max_samples,
            sample_fraction=sample_fraction,
            sample_seed=sample_seed,
        ),
        project_root,
        batch_size=config["training"]["batch_size"],
        training=False,
        seed=config["training"]["seed"],
        num_workers=config["training"].get("num_workers", 4),
        sample_rate=config["audio"]["sample_rate"],
        segment_seconds=config["audio"]["segment_seconds"],
        audio_root_name=audio_root_name,
    )

    gpu_logmel = build_gpu_logmel(config, device)
    stats_path = project_root / config.get(
        "logmel_stats_path",
        "experiments/baselines/E1_v4/E1_ATADD_logmel_stats_v4.json",
    )

    mel_mean, mel_std, _stats = load_or_compute_mel_stats(
        config,
        stats_loader,
        gpu_logmel,
        device,
        stats_path,
        max_batches=None if stats_path.exists() else args.max_samples,
    )

    config.setdefault("mel", {})["mean"] = mel_mean
    config.setdefault("mel", {})["std"] = mel_std

    dev_df = load_manifest_df(
        project_root,
        config["dataset"]["dev_manifest"],
        "dev",
        max_samples=args.max_samples,
        sample_fraction=sample_fraction,
        sample_seed=sample_seed,
    )

    dev_loader = make_dataloader(
        dev_df,
        project_root,
        batch_size=config["training"]["batch_size"],
        training=False,
        seed=config["training"]["seed"],
        num_workers=config["training"].get("num_workers", 4),
        sample_rate=config["audio"]["sample_rate"],
        segment_seconds=config["audio"]["segment_seconds"],
        audio_root_name=audio_root_name,
    )

    frontend = MelFrontend(gpu_logmel, mel_mean, mel_std, device)
    trainer = Trainer(config, project_root, experiment_dir, figure_dir, frontend, device, mode=mode)

    import torch.nn as nn

    train_df_for_weights = load_manifest_df(
        project_root,
        config["dataset"]["train_manifest"],
        "train",
        max_samples=args.max_samples,
        sample_fraction=sample_fraction,
        sample_seed=sample_seed,
    )
    label_ids = train_df_for_weights["label"].map({"real": 0, "fake": 1})
    class_counts = np.zeros(2, dtype=np.float64)
    for class_id, count in label_ids.value_counts().items():
        class_counts[int(class_id)] = float(count)
    if class_counts.min() <= 0:
        raise ValueError(
            "Training manifest must contain both 'real' and 'fake' labels for class-weighted CE."
        )
    class_weights = class_counts.sum() / (len(class_counts) * class_counts)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))

    best_path = experiment_dir / "best.pt"

    if args.checkpoint:
        try:
            ckpt = torch.load(project_root / args.checkpoint, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(project_root / args.checkpoint, map_location=device)
        trainer.model.load_state_dict(ckpt["model_state_dict"], strict=True)
        best_path = Path(project_root / args.checkpoint)
        logger.info("Loaded checkpoint from %s", best_path)
    else:
        best_path, last_path = trainer.fit(
            train_loader,
            dev_loader,
            criterion,
            epochs=config["training"]["epochs"],
            patience=config["training"]["patience"],
        )
        logger.info("Training complete. best=%s last=%s", best_path, last_path)

    baseline_path = project_root / config.get(
        "baseline_metrics_path",
        "experiments/baselines/E1_v4/E1_ATADD_eval_metrics_v4.json",
    )

    for split in [s.strip() for s in args.eval_splits.split(",") if s.strip()]:
        metrics, paths = run_split_evaluation(
            config,
            project_root,
            split,
            best_path,
            experiment_dir,
            figure_dir,
            device,
            max_samples=args.max_samples,
            sample_fraction=sample_fraction,
            sample_seed=sample_seed,
        )

        print_metrics_table(metrics, title=f"Eval metrics ({split})")

        if split == "eval":
            cmp_df = compare_to_e1_baseline(metrics, baseline_path)

            if cmp_df is not None:
                print("\nRegression vs E1_v4 baseline\n", cmp_df.to_string(index=False))

        print("\nArtifacts:")
        for key, path in paths.items():
            print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
