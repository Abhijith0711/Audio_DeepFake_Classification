import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.preprocessing.preprocess_atadd_v4 import cpu_prepare_waveform

LABEL_MAP = {"real": 0, "fake": 1}

MANIFEST_HINT = (
    "Manifest not found. Run src/datasets/create_btp_dataset.py to build AT-ADD splits under data/manifests/."
)

DEFAULT_AUDIO_ROOT_NAME = "AT_ADD_BTP"
_SPLIT_DIRS = ("train", "dev", "eval")


def find_project_root(start_path=None):
    start_path = Path(start_path or Path.cwd()).resolve()

    for candidate in [start_path, *start_path.parents]:
        if (candidate / "data" / "manifests" / "train_manifest.csv").exists() and (
            candidate / "src" / "preprocessing" / "preprocess_atadd_v4.py"
        ).exists():
            return candidate

    raise FileNotFoundError("Could not locate Audio_DeepFake_Classification project root.")


def resolve_manifest_path(project_root, manifest_rel):
    path = project_root / manifest_rel

    if not path.exists():
        raise FileNotFoundError(f"{path} not found. {MANIFEST_HINT}")

    return path


def resolve_audio_root(project_root, audio_root_name=None):
    """Root directory containing train/dev/eval subfolders with FLAC files."""
    project_root = Path(project_root)
    name = audio_root_name or DEFAULT_AUDIO_ROOT_NAME

    candidates = [
        project_root / name,
        project_root / "data" / "shared" / name,
    ]
    if name in ("AT_ADD", "AT_ADD_BTP"):
        candidates.append(project_root / "data" / "raw" / "AT_ADD")

    for root in candidates:
        if root.is_dir() and any((root / split).is_dir() for split in _SPLIT_DIRS):
            return root.resolve()

    tried = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Audio root for '{name}' not found. Tried: {tried}. "
        f"Expected layout: {{root}}/{{train,dev,eval}}/{{filename}}. "
        f"Set dataset.audio_root in your YAML config."
    )


def audio_root_from_config(config):
    if not config:
        return DEFAULT_AUDIO_ROOT_NAME
    return config.get("dataset", {}).get("audio_root", DEFAULT_AUDIO_ROOT_NAME)


def audio_path_from_row(row, project_root, audio_root_name=None):
    """Resolve a clip under its original folder.

    Reworked manifests keep `split` as the experiment split and `folder` as the
    directory that actually holds the file. Looking up `split/name` misses clips
    that were reassigned across train, dev, and eval.
    """
    root = resolve_audio_root(project_root, audio_root_name)
    folder = row["folder"] if "folder" in row and pd.notna(row["folder"]) else row["split"]
    return root / str(folder) / str(row["name"])


def _split_sample_seed(base_seed, split_name):
    return int(base_seed) + sum(ord(c) for c in split_name) * 997


def sample_manifest_fraction(df, fraction, seed, split_name):
    """Stratified random subset by label (reproducible). Keeps at least one row per label."""
    if fraction is None or fraction >= 1.0:
        return df.reset_index(drop=True)

    if not 0 < fraction < 1.0:
        raise ValueError(f"sample fraction must be in (0, 1], got {fraction}")

    split_seed = _split_sample_seed(seed, split_name)
    parts = []

    for label, group in df.groupby("label", sort=False):
        n_take = max(1, int(round(len(group) * fraction)))
        n_take = min(n_take, len(group))
        label_seed = split_seed + LABEL_MAP.get(str(label), 0) * 17
        parts.append(group.sample(n=n_take, random_state=label_seed))

    out = pd.concat(parts, ignore_index=True)
    return out.sample(frac=1, random_state=split_seed).reset_index(drop=True)


def stratified_indices(df, num_samples, seed=42, columns=("type", "label")):
    """Reproducible subset of row positions, balanced across the `columns` strata.

    Manifests are sorted (e.g. the first rows of dev/eval are all music/fake), so taking the
    first N rows gives a single-class, single-type sample. This draws an equal share from
    every (type, label) stratum instead.
    """
    columns = [c for c in columns if c in df.columns]

    if not columns or num_samples >= len(df):
        return list(range(len(df)))

    groups = [g.index.to_numpy() for _, g in df.reset_index(drop=True).groupby(columns, sort=True)]
    rng = np.random.default_rng(seed)
    per_group = max(1, num_samples // len(groups))
    picked = []

    for positions in groups:
        take = min(per_group, len(positions))
        picked.extend(rng.choice(positions, size=take, replace=False).tolist())

    return sorted(picked)


def resolve_sample_fraction(config, cli_fraction=None):
    if cli_fraction is not None:
        return cli_fraction
    return config.get("dataset", {}).get("sample_fraction")


def load_manifest_df(
    project_root,
    manifest_rel,
    split_name,
    max_samples=None,
    sample_fraction=None,
    sample_seed=42,
):
    path = resolve_manifest_path(project_root, manifest_rel)
    df = pd.read_csv(path)

    required = {"name", "label", "split"}

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"{split_name} manifest missing columns: {sorted(missing)}")

    if "file_id" not in df.columns:
        df["file_id"] = df["name"].astype(str)

    unknown = set(df["label"].dropna().astype(str).unique()) - set(LABEL_MAP)

    if unknown:
        raise ValueError(f"{split_name} contains unknown labels: {unknown}")

    if sample_fraction is not None and sample_fraction < 1.0:
        df = sample_manifest_fraction(df, sample_fraction, sample_seed, split_name)

    if max_samples is not None:
        df = df.iloc[: max_samples].copy()

    return df.reset_index(drop=True)


class ATADDManifestDataset(Dataset):
    def __init__(
        self,
        df,
        project_root,
        sample_rate=24000,
        segment_seconds=4,
        training=False,
        audio_root_name=None,
    ):
        self.df = df.reset_index(drop=True)
        self.project_root = Path(project_root)
        self.audio_root_name = audio_root_name or DEFAULT_AUDIO_ROOT_NAME
        self.sample_rate = sample_rate
        self.segment_seconds = segment_seconds
        self.training = training

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        audio_path = audio_path_from_row(row, self.project_root, self.audio_root_name)

        if not audio_path.exists():
            root = resolve_audio_root(self.project_root, self.audio_root_name)
            raise FileNotFoundError(
                f"Missing audio: {audio_path}. Expected files under {root}/{{train,dev,eval}}/ "
                f"(dataset.audio_root={self.audio_root_name!r})."
            )

        waveform = cpu_prepare_waveform(
            audio_path,
            sample_rate=self.sample_rate,
            segment_seconds=self.segment_seconds,
            training=self.training,
        )

        label = LABEL_MAP[str(row["label"])]

        meta = {
            "file_id": str(row["file_id"]),
            "type": str(row["type"]) if "type" in row and pd.notna(row["type"]) else "unknown",
            "generator": str(row["generator"]) if "generator" in row and pd.notna(row["generator"]) else "-",
            "label_name": str(row["label"]),
        }

        return waveform, torch.tensor(label, dtype=torch.long), meta


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_dataloader(
    df,
    project_root,
    batch_size,
    training,
    seed,
    num_workers=4,
    prefetch_factor=2,
    pin_memory=True,
    sample_rate=24000,
    segment_seconds=4,
    audio_root_name=None,
):
    dataset = ATADDManifestDataset(
        df,
        project_root,
        sample_rate=sample_rate,
        segment_seconds=segment_seconds,
        training=training,
        audio_root_name=audio_root_name,
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": training,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": seed_worker,
    }

    generator = torch.Generator()
    generator.manual_seed(seed)

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(
        dataset,
        generator=generator,
        collate_fn=collate_manifest_batch,
        **loader_kwargs,
    )


def collate_manifest_batch(batch):
    waveforms, labels, metas = zip(*batch)
    return torch.stack(waveforms), torch.stack(labels), list(metas)


def collate_paired_batch(batch):
    clean, degraded, labels, types, metas = zip(*batch)
    return (
        torch.stack(clean),
        torch.stack(degraded),
        torch.stack(labels),
        list(types),
        list(metas),
    )
