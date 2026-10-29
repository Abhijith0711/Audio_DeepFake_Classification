"""Train M3 ablations for seed 42 under the reworked FT-RCL config.

Each variant changes one setting (gate, prediction-coherence loss, gamma, or beta)
and leaves the rest of the config alone. ``--dry-run`` prints the commands and
does not train.
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.atadd_manifest_dataset import find_project_root
from src.training.ablations import ABLATION_VARIANTS, build_ablation_command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = find_project_root()
    for name, overrides in ABLATION_VARIANTS:
        command = build_ablation_command(name, overrides)
        print(" ".join(command))
        if args.dry_run:
            continue
        subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
