import random

import numpy as np
import torch
from torch.utils.data import Dataset

from src.datasets.atadd_manifest_dataset import ATADDManifestDataset, audio_path_from_row
from src.preprocessing.preprocess_atadd_v4 import load_audio, repeat_pad_audio
from src.robustness.conditions import ConditionSpec, conditions_for_max_severity, identity_condition
from src.robustness.degrade import degrade, ffmpeg_available


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
        self.seed = seed
        # Created lazily inside each DataLoader worker so workers do not share one RNG stream.
        self._rng = None

        self._fixed_pool = None
        if condition_pool is not None:
            self._fixed_pool = [c for c in condition_pool if c.family != "identity"]
            self._check_codec_tools(self._fixed_pool)

        # Shared-memory scalar: persistent DataLoader workers keep their own copy of the
        # dataset, so a plain attribute set in the main process would never reach them.
        self._max_severity = torch.tensor([int(max_severity)], dtype=torch.long).share_memory_()
        self._check_codec_tools(self.condition_pool)

    @staticmethod
    def _check_codec_tools(pool):
        if any(c.family == "mp3_codec" for c in pool) and not ffmpeg_available():
            raise RuntimeError(
                "The condition pool contains MP3 conditions but ffmpeg was not found on PATH. "
                "Install ffmpeg or pass an explicit condition_pool without mp3_codec."
            )

    @property
    def max_severity(self):
        return int(self._max_severity[0])

    @property
    def condition_pool(self):
        if self._fixed_pool is not None:
            return self._fixed_pool

        return [
            c
            for c in conditions_for_max_severity(self.max_severity)
            if c.family != "identity"
        ]

    @property
    def rng(self):
        if self._rng is None:
            worker = torch.utils.data.get_worker_info()
            worker_id = worker.id if worker is not None else 0
            self._rng = random.Random(self.seed * 1000003 + worker_id)
        return self._rng

    @property
    def df(self):
        return self.base.df

    def set_max_severity(self, max_severity):
        self._max_severity[0] = int(max_severity)
        self._check_codec_tools(self.condition_pool)

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

        pool = self.condition_pool

        if self.training and pool:
            condition = self.rng.choice(pool)
        elif pool:
            condition = pool[index % len(pool)]
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
