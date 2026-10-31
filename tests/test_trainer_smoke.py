import copy
import json

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.datasets.atadd_manifest_dataset import collate_manifest_batch, collate_paired_batch
from src.preprocessing.preprocess_atadd_v4 import GPULogMel
from src.robustness.conditions import all_training_conditions
from src.training.frontend import MelFrontend
from src.training.schedulers.robustness_curriculum import early_stop_floor, should_early_stop
from src.training.trainer import Trainer

SR, SECONDS = 8000, 1
CONFIG = {
    "audio": {"sample_rate": SR, "segment_seconds": SECONDS},
    "training": {"learning_rate": 1e-3, "weight_decay": 1e-4, "seed": 0, "min_epoch_for_early_stop": 14},
    "ft_rcl": {"curriculum": {"epoch_milestones": [1, 6, 13], "severity_milestones": [1, 3, 4]}},
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


def _trainer(tmp_path, mode, config=None):
    device = torch.device("cpu")
    logmel = GPULogMel(SR, 512, 400, 160, 80)
    frontend = MelFrontend(logmel, 0.0, 1.0, device)
    return Trainer(config or CONFIG, tmp_path, tmp_path, tmp_path, frontend, device, mode=mode)


@pytest.mark.parametrize("mode", ["M0", "M1", "M2", "M3"])
def test_mode_training_step_is_finite_and_updates_weights(tmp_path, mode):
    trainer = _trainer(tmp_path, mode)
    before = [p.detach().clone() for p in trainer.model.parameters()]
    if mode == "M0":
        loader = DataLoader(_Dev(), batch_size=4, collate_fn=collate_manifest_batch)
    else:
        loader = DataLoader(_Paired(), batch_size=4, collate_fn=collate_paired_batch)
    loss, _ = trainer.train_one_epoch(loader, nn.CrossEntropyLoss(), epoch=1)
    assert np.isfinite(loss)
    assert any(not torch.equal(b, p) for b, p in zip(before, trainer.model.parameters()))


def test_checkpoint_eligibility_follows_the_config(tmp_path):
    cfg = copy.deepcopy(CONFIG)
    cfg["selection"] = {"eligible_from_final_stage": True}
    m3 = _trainer(tmp_path, "M3", cfg)
    m0 = _trainer(tmp_path, "M0", cfg)
    assert m3.checkpoint_eligible(7) is False
    assert m3.checkpoint_eligible(13) is True
    assert m3.checkpoint_eligible(4) is False
    assert m0.checkpoint_eligible(3) is True
    assert m3.model.signature_stage == 3


def test_run_summary_records_severity_and_skips_ineligible_epochs(tmp_path):
    cfg = copy.deepcopy(CONFIG)
    cfg["selection"] = {"eligible_from_final_stage": True}
    trainer = _trainer(tmp_path, "M3", cfg)
    train_loader = DataLoader(_Paired(), batch_size=4, collate_fn=collate_paired_batch)
    dev_loader = DataLoader(_Dev(), batch_size=8, collate_fn=collate_manifest_batch)
    trainer.fit(train_loader, dev_loader, nn.CrossEntropyLoss(), epochs=1, patience=5)
    summary = json.loads((tmp_path / "run_summary.json").read_text())
    assert summary["mode"] == "M3"
    assert summary["best_epoch"] is None
    assert summary["selected_max_severity"] is None
    assert not (tmp_path / "best.pt").exists()
    history = pd.read_csv(tmp_path / "training_history.csv")
    assert int(history.loc[0, "max_severity"]) == 1
    assert str(history.loc[0, "checkpoint_eligible"]).lower() in ("false", "0")


def test_early_stop_waits_until_the_epoch_after_severity_4():
    assert early_stop_floor([1, 6, 13]) == 14
    assert should_early_stop(13, epochs_without_improvement=20, patience=5, min_epoch=14) is False
    assert should_early_stop(14, epochs_without_improvement=5, patience=5, min_epoch=14) is True
    assert should_early_stop(14, epochs_without_improvement=4, patience=5, min_epoch=14) is False
