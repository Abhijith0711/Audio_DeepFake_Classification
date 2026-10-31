import numpy as np
import pandas as pd
import pytest
import torch

from src.robustness.conditions import ConditionSpec, all_training_conditions
from src.training.ablations import ABLATION_VARIANTS, ablation_commands
from src.training.config_overrides import apply_overrides
from src.training.selection import (
    cache_degraded_dev,
    checkpoint_is_eligible,
    select_degraded_dev_indices,
    selection_score,
)


def test_checkpoint_is_eligible_only_from_the_final_stage_for_robust_modes():
    assert checkpoint_is_eligible(7, "M3", True, 13) is False
    assert checkpoint_is_eligible(4, "M2", True, 13) is False
    assert checkpoint_is_eligible(13, "M1", True, 13) is True
    assert checkpoint_is_eligible(3, "M0", True, 13) is True
    assert checkpoint_is_eligible(3, "M3", False, 13) is True


def test_selection_score_is_the_mean_when_degraded_dev_is_on():
    assert selection_score(0.8) == 0.8
    assert selection_score(0.8, 0.4, use_degraded_dev=False) == 0.8
    assert selection_score(0.8, 0.4, use_degraded_dev=True) == pytest.approx(0.6)
    with pytest.raises(ValueError):
        selection_score(0.8, use_degraded_dev=True)


def test_degraded_dev_subset_is_not_the_head_of_a_sorted_manifest():
    rows = [{"type": t, "label": label} for t in ("music", "speech", "sound", "singing") for label in ("fake", "real") for _ in range(20)]
    df = pd.DataFrame(rows)
    assert list(df.iloc[:8]["type"].unique()) == ["music"]
    indices = select_degraded_dev_indices(df, 16, seed=1)
    assert indices != list(range(16))
    picked = df.iloc[indices]
    assert len(picked.groupby(["type", "label"])) == 8
    assert select_degraded_dev_indices(df, 16, seed=1) == indices


class _Waves:
    sample_rate = 8000

    def __init__(self):
        rows = [{"type": t, "label": label, "file_id": f"{t}-{label}-{i}"} for t in ("music", "speech") for label in ("fake", "real") for i in range(4)]
        self.df = pd.DataFrame(rows)

    def __getitem__(self, index):
        wave = np.zeros(self.sample_rate, dtype=np.float32)
        label = 1 if self.df.loc[index, "label"] == "fake" else 0
        meta = {"file_id": self.df.loc[index, "file_id"], "type": self.df.loc[index, "type"]}
        return torch.from_numpy(wave), torch.tensor(label), meta


def test_degraded_cache_uses_the_stratified_subset():
    dataset = _Waves()
    noise = ConditionSpec("additive_noise", "snr_20db", 2, {"snr_db": 20})
    cache = cache_degraded_dev(dataset, num_samples=8, seed=0, conditions=[noise])
    assert cache["indices"] == select_degraded_dev_indices(dataset.df, 8, 0)
    assert len(cache["clean_waves"]) == 8
    assert cache["degraded_waves"][0].shape == cache["clean_waves"][0].shape
    assert set(cache["types"]) == {"music", "speech"}


def test_set_overrides_parse_numbers_and_leave_gate_none_as_text():
    config = {"ft_rcl": {"gamma_ftsc": 0.5, "beta": 0.5, "gate": "full"}}
    apply_overrides(
        config,
        ["ft_rcl.gamma_ftsc=0.0", "ft_rcl.beta=1", "ft_rcl.gate=none", "selection.use_degraded_dev=true"],
    )
    assert config["ft_rcl"]["gamma_ftsc"] == 0.0
    assert config["ft_rcl"]["beta"] == 1
    assert config["ft_rcl"]["gate"] == "none"
    assert config["selection"]["use_degraded_dev"] is True


def test_ablation_commands_cover_m3_variants_for_seed_42():
    commands = [" ".join(command) for command in ablation_commands(python="python")]
    joined = "\n".join(commands)
    assert len(commands) == len(ABLATION_VARIANTS)
    for snippet in (
        "--mode M3",
        "--seed 42",
        "ft_rcl.gate=no_confidence",
        "ft_rcl.gate=no_type",
        "ft_rcl.gate=none",
        "ft_rcl.pc_loss=js",
        "ft_rcl.gamma_ftsc=0.0",
        "ft_rcl.gamma_pc=0.0",
        "ft_rcl.beta=0.0",
        "ft_rcl.beta=0.25",
        "ft_rcl.beta=1.0",
    ):
        assert snippet in joined


def test_severity_note_names_every_training_condition():
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / "docs" / "SEVERITY_SCALE.md").read_text(encoding="utf-8")
    assert "ordinal" in text.lower()
    assert "not calibrated" in text.lower() or "not a calibrated" in text.lower()
    for condition in all_training_conditions():
        assert condition.name in text or (condition.family == "identity" and "clean" in text)
