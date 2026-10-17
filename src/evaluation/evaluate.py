import argparse
import logging
from pathlib import Path

import torch
import yaml

from src.datasets.atadd_manifest_dataset import find_project_root
from src.evaluation.eval_runner import compare_to_e1_baseline, run_split_evaluation
from src.training.run_artifacts import print_metrics_table, setup_run_logging, write_run_meta

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="eval")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    project_root = find_project_root()
    config_path = project_root / args.config

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    experiment_dir = project_root / "experiments" / args.experiment_id
    figure_dir = project_root / "results" / "figures" / args.experiment_id
    log_dir = project_root / "results" / "logs" / args.experiment_id
    log_dir.mkdir(parents=True, exist_ok=True)

    setup_run_logging(log_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    write_run_meta(log_dir, config["training"]["seed"], device)

    checkpoint_path = project_root / args.checkpoint

    metrics, paths = run_split_evaluation(
        config,
        project_root,
        args.split,
        checkpoint_path,
        experiment_dir,
        figure_dir,
        device,
        max_samples=args.max_samples,
    )

    print_metrics_table(metrics, title=f"Eval metrics ({args.split})")

    if args.split == "eval":
        baseline_path = project_root / config.get(
            "baseline_metrics_path",
            "experiments/baselines/E1_v4/E1_ATADD_eval_metrics_v4.json",
        )
        cmp_df = compare_to_e1_baseline(metrics, baseline_path)

        if cmp_df is not None:
            print("\nRegression vs E1_v4 baseline\n", cmp_df.to_string(index=False))

    print("\nArtifacts:")
    for key, path in paths.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
