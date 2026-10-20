"""
Verify manifest CSVs against on-disk audio under AT_ADD_BTP (or another audio_root).

Run from the project root:
    python -m src.datasets.verify_at_add
    python src/datasets/verify_at_add.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python src/datasets/verify_at_add.py` without installing the package.
if __name__ == "__main__" and __package__ is None:
    _project_root = Path(__file__).resolve().parents[2]
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))

import pandas as pd

from src.datasets.atadd_manifest_dataset import (
    DEFAULT_AUDIO_ROOT_NAME,
    find_project_root,
    resolve_audio_root,
)


def main():
    parser = argparse.ArgumentParser(description="Verify AT-ADD manifest files exist on disk.")
    parser.add_argument(
        "--audio-root",
        default=DEFAULT_AUDIO_ROOT_NAME,
        help=f"Dataset folder name under project root (default: {DEFAULT_AUDIO_ROOT_NAME})",
    )
    args = parser.parse_args()

    project_root = find_project_root()
    root = resolve_audio_root(project_root, args.audio_root)
    manifest_dir = project_root / "data" / "manifests"

    print(f"Project root: {project_root}")
    print(f"Using audio root: {root}")

    for split in ("train", "dev", "eval"):
        manifest_path = manifest_dir / f"{split}_manifest.csv"
        df = pd.read_csv(manifest_path)

        print(f"\n{split.upper()}")
        print(f"Manifest samples: {len(df)}")

        missing = []
        found = 0

        for name in df["name"]:
            path = root / split / name
            if path.is_file():
                found += 1
            else:
                missing.append(name)

        print(f"Files found: {found}")
        print(f"Files missing: {len(missing)}")

        if missing:
            print("First missing files:")
            for name in missing[:10]:
                print(name)

    print("\nVerification complete.")


if __name__ == "__main__":
    main()
