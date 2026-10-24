import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.datasets.atadd_manifest_dataset import collate_manifest_batch, collate_paired_batch
from src.preprocessing.preprocess_atadd_v4 import GPULogMel
from src.robustness.conditions import all_training_conditions
from src.training.frontend import MelFrontend
from src.training.trainer import Trainer

SR, SECONDS = 8000, 1
CONFIG = {
    "audio": {"sample_rate": SR, "segment_seconds": SECONDS},
    "training": {"learning_rate": 1e-3, "weight_decay": 1e-4, "seed": 0},
    "selection": {"delta_num_samples": 16},
    "ft_rcl": {},
}


class _Dev(Dataset):
    def __init__(self):
        rows = [(t, l) for t in ("music", "speech") for l in ("fake", "real") for _ in range(8)]
        self.df = pd.DataFrame(rows, columns=["type", "label"])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        wave = torch.from_numpy(np.random.default_rng(i).standard_normal(SR * SECONDS).astype(np.float32) * 0.1)
        label = torch.tensor(int(self.df.label[i] == "fake"))
        return wave, label, {"file_id": f"f{i}", "type": self.df.type[i]}


class _Paired(Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, i):
        wave = torch.randn(SR * SECONDS) * 0.1
        cond = [c for c in all_training_conditions() if c.family in ("additive_noise", "telephony")][i % 4]
        meta = {"severity": cond.severity, "condition_family": cond.family, "condition_name": cond.name}
        return wave, wave + 0.01 * torch.randn_like(wave), torch.tensor(i % 2), "speech", meta


def _trainer(tmp_path, mode):
    device = torch.device("cpu")
    logmel = GPULogMel(SR, 512, 400, 160, 80)
    frontend = MelFrontend(logmel, 0.0, 1.0, device)
    return Trainer(CONFIG, tmp_path, tmp_path, tmp_path, frontend, device, mode=mode)


def test_m3_training_step_is_finite_and_updates_weights(tmp_path):
    trainer = _trainer(tmp_path, "M3")
    before = [p.detach().clone() for p in trainer.model.parameters()]
    loader = DataLoader(_Paired(), batch_size=4, collate_fn=collate_paired_batch)
    loss, _ = trainer.train_one_epoch(loader, nn.CrossEntropyLoss(), epoch=1)
    assert np.isfinite(loss)
    assert any(not torch.equal(b, p) for b, p in zip(before, trainer.model.parameters()))


def test_dev_delta_uses_stratified_subset_and_is_non_negative(tmp_path, monkeypatch):
    # Keep the test independent of ffmpeg; MP3 without ffmpeg is covered in test_bugfixes.py.
    no_mp3 = [c for c in all_training_conditions() if c.family != "mp3_codec"]
    monkeypatch.setattr("src.training.trainer.all_training_conditions", lambda: no_mp3)
    trainer = _trainer(tmp_path, "M3")
    dev_loader = DataLoader(_Dev(), batch_size=8, collate_fn=collate_manifest_batch)
    delta = trainer.estimate_dev_delta_macro_f1(dev_loader)
    assert delta >= 0.0
    labels = trainer._delta_cache[2]
    assert set(labels.tolist()) == {0, 1}
