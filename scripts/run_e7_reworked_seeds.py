"""Train M0, M1, M2, and M3 on the full reworked subset for each benchmark seed.

Seeds match the reworked notebooks: seed1=42, seed2=1337, seed3=2024.
Each run writes experiments/E7_{mode}_reworked_{seed_tag}/.
After the runs, results/tables/e7_reworked_seed_summary.csv holds the mean and
variance of macro-F1, EER, and type-averaged macro-F1.
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.atadd_manifest_dataset import find_project_root
from src.evaluation.seed_summary import BENCHMARK_SEEDS, MODES, experiment_id, write_seed_summary

CONFIG = "configs/proposed/e7_ft_rcl_atadd_v1.yaml"


def build_command(mode, seed_tag, seed):
    return [
        sys.executable,
        "-m",
        "src.training.train",
        "--config",
        CONFIG,
        "--mode",
        mode,
        "--seed",
        str(seed),
        "--experiment-id",
        experiment_id(mode, seed_tag),
        "--eval-splits",
        "dev,eval",
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=list(MODES))
    parser.add_argument(
        "--seed-tags",
        nargs="+",
        default=[tag for tag, _ in BENCHMARK_SEEDS],
        choices=[tag for tag, _ in BENCHMARK_SEEDS],
    )
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = find_project_root()
    seed_map = dict(BENCHMARK_SEEDS)

    if not args.summarize_only:
        for mode in args.modes:
            for seed_tag in args.seed_tags:
                command = build_command(mode, seed_tag, seed_map[seed_tag])
                print(" ".join(command))
                if args.dry_run:
                    continue
                subprocess.run(command, cwd=root, check=True)

    if args.dry_run:
        return

    runs_path, summary_path = write_seed_summary(root)
    print(runs_path)
    print(summary_path)


if __name__ == "__main__":
    main()
