import pytest
import torch
import torch.nn as nn

from src.training.losses.ft_rcl import ft_rcl_loss, ftsc_loss, prediction_coherence_loss


def test_prediction_coherence_forward():
    p_clean = torch.rand(4)
    p_deg = torch.rand(4)
    loss = prediction_coherence_loss(p_clean, p_deg)
    assert loss.ndim == 0
    assert loss.item() >= 0


def test_ftsc_forward():
    s_clean = torch.randn(4, 8)
    s_deg = torch.randn(4, 8)
    s_clean = s_clean / s_clean.norm(dim=-1, keepdim=True)
    s_deg = s_deg / s_deg.norm(dim=-1, keepdim=True)
    loss = ftsc_loss(s_clean, s_deg)
    assert 0.0 <= loss.item() <= 2.0


def test_ft_rcl_loss_forward():
    logits = torch.randn(4, 2)
    labels = torch.tensor([0, 1, 0, 1])
    s = torch.randn(4, 8)
    s = s / s.norm(dim=-1, keepdim=True)
    total, parts = ft_rcl_loss(
        logits,
        logits,
        labels,
        s,
        s,
        ["speech", "music", "singing", "sound"],
        severity_eta=1.0,
        type_alpha={"speech": 1.0, "music": 0.3},
        criterion=nn.CrossEntropyLoss(),
    )
    assert total.ndim == 0
    assert "l_pc" in parts


try:
    from src.datasets.atadd_manifest_dataset import find_project_root, load_manifest_df

    _ = find_project_root()
    _has_data = True
except Exception:
    _has_data = False


@pytest.mark.skipif(not _has_data, reason="manifests missing")
def test_manifest_paths_or_skip():
    root = find_project_root()
    try:
        load_manifest_df(root, "data/manifests/train_manifest.csv", "train", max_samples=1)
    except FileNotFoundError as exc:
        if "data/raw/AT_ADD" in str(exc) or "Missing audio" in str(exc):
            pytest.skip(str(exc))
        raise
