import random

import numpy as np
import torch
from torch.utils.data import Dataset

from src.datasets.atadd_manifest_dataset import ATADDManifestDataset, audio_path_from_row
from src.preprocessing.preprocess_atadd_v4 import load_audio, repeat_pad_audio
from src.robustness.conditions import ConditionSpec, conditions_for_max_severity, identity_condition
from src.robustness.degrade import degrade


class PairedRobustnessDataset(Dataset):
    def __init__(
        self,
        df,
        project_root,
        sample_rate=24000,
        segment_seconds=4,
        training=True,
        max_severity=99,
        condition_pool=None,
        seed=42,
        audio_root_name=None,
    ):
        self.base = ATADDManifestDataset(
            df,
            project_root,
            sample_rate=sample_rate,
            segment_seconds=segment_seconds,
            training=training,
            audio_root_name=audio_root_name,
        )
        self.sample_rate = sample_rate
        self.segment_seconds = segment_seconds
        self.num_samples = sample_rate * segment_seconds
        self.training = training
        self.project_root = project_root
        self.rng = random.Random(seed)

        if condition_pool is None:
            condition_pool = conditions_for_max_severity(max_severity)

        self.condition_pool = [c for c in condition_pool if c.family != "identity"]
        self.max_severity = max_severity

    @property
    def df(self):
        return self.base.df

    def set_max_severity(self, max_severity):
        self.max_severity = max_severity
        self.condition_pool = [
            c
            for c in conditions_for_max_severity(max_severity)
            if c.family != "identity"
        ]

    def __len__(self):
        return len(self.base)

    def _load_segment(self, row, crop_start=None):
        audio_path = audio_path_from_row(row, self.base.project_root, self.base.audio_root_name)
        waveform = load_audio(audio_path, self.sample_rate)

        if len(waveform) >= self.num_samples:
            if crop_start is None:
                if self.training:
                    crop_start = self.rng.randint(0, len(waveform) - self.num_samples)
                else:
                    crop_start = (len(waveform) - self.num_samples) // 2
            waveform = waveform[crop_start : crop_start + self.num_samples]
        else:
            waveform = repeat_pad_audio(waveform, self.num_samples)
            crop_start = 0

        return waveform.astype(np.float32), crop_start

    def __getitem__(self, index):
        row = self.base.df.iloc[index]
        clean_np, crop_start = self._load_segment(row)

        if self.training and self.condition_pool:
            condition = self.rng.choice(self.condition_pool)
        elif self.condition_pool:
            condition = self.condition_pool[index % len(self.condition_pool)]
        else:
            condition = identity_condition()

        source_id = str(row.get("file_id", row["name"]))
        degraded_np, meta = degrade(
            clean_np,
            self.sample_rate,
            condition,
            mode="random" if self.training else "deterministic",
            source_id=source_id,
        )

        label = torch.tensor(
            {"real": 0, "fake": 1}[str(row["label"])],
            dtype=torch.long,
        )

        type_name = str(row["type"]) if "type" in row else "unknown"

        batch_meta = {
            **meta,
            "type": type_name,
            "generator": str(row.get("generator", "-")),
            "crop_start": crop_start,
        }

        return (
            torch.from_numpy(clean_np),
            torch.from_numpy(degraded_np),
            label,
            type_name,
            batch_meta,
        )
