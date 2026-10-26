"""Build a generator-disjoint train/validation split from the AT-ADD Track 2 train and dev lists.

The official dev split shares all 29 fake generators with the training split, whereas the
evaluation split uses 38 unseen generators. Validation on dev therefore overestimates
performance on unseen generators. Here a few generators per audio type are removed from
training and their dev clips (plus all dev real clips) form the validation set.
"""
import argparse
from pathlib import Path

import pandas as pd

HELD_OUT = {
    ("speech", "Kokoro"),
    ("speech", "MelGAN"),
    ("speech", "Tacotron2"),
    ("speech", "WaveNet"),
    ("speech", "Parler-TTS (mini)"),
    ("music", "MusicLDM"),
    ("singing", "SeedVC"),
    ("sound", "Audiogen"),
}


def build(manifest_dir, out_dir):
    manifest_dir, out_dir = Path(manifest_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(manifest_dir / "train_manifest.csv").drop(columns=["caption"], errors="ignore")
    dev = pd.read_csv(manifest_dir / "dev_manifest.csv").drop(columns=["caption"], errors="ignore")

    train_held = [(t, g) in HELD_OUT and l == "fake" for t, g, l in zip(train["type"], train["generator"], train["label"])]
    dev_held = [(t, g) in HELD_OUT and l == "fake" for t, g, l in zip(dev["type"], dev["generator"], dev["label"])]

    new_train = train[[not h for h in train_held]]
    new_dev = dev[(dev["label"] == "real") | pd.Series(dev_held, index=dev.index)]

    new_train.to_csv(out_dir / "train_manifest.csv", index=False)
    new_dev.to_csv(out_dir / "dev_manifest.csv", index=False)

    for name, df in (("train", new_train), ("dev", new_dev)):
        print(name, len(df), df.groupby(["type", "label"]).size().to_dict())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", default="data/manifests")
    parser.add_argument("--out-dir", default="data/manifests/generator_disjoint")
    args = parser.parse_args()
    build(args.manifest_dir, args.out_dir)
