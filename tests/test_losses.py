import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training.losses.ft_rcl import (
    ft_rcl_loss,
    ftsc_loss,
    prediction_coherence_loss,
    robustness_weight,
)


def test_prediction_coherence_forward():
    p_clean = torch.rand(4)
    p_deg = torch.rand(4)
    loss = prediction_coherence_loss(p_clean, p_deg)
    assert loss.ndim == 0
    assert loss.item() >= 0


def _reference_js(p, q, eps=1e-6):
    p = np.clip(p, eps, None)
    p = p / p.sum(axis=-1, keepdims=True)
    q = np.clip(q, eps, None)
    q = q / q.sum(axis=-1, keepdims=True)
    m = np.clip(0.5 * (p + q), eps, None)
    m = m / m.sum(axis=-1, keepdims=True)
    kl_pm = np.sum(p * (np.log(p) - np.log(m)), axis=-1)
    kl_qm = np.sum(q * (np.log(q) - np.log(m)), axis=-1)
    return float(np.mean(np.clip(0.5 * (kl_pm + kl_qm), 0.0, None)))


def test_jensen_shannon_matches_reference():
    torch.manual_seed(0)
    logits_clean = torch.randn(6, 2)
    logits_deg = torch.randn(6, 2)
    probs_clean = F.softmax(logits_clean, dim=1)
    probs_deg = F.softmax(logits_deg, dim=1)
    got = prediction_coherence_loss(probs_clean, probs_deg, pc_loss="js")
    reference = _reference_js(probs_clean.numpy(), probs_deg.numpy())
    assert got.dtype == torch.float32
    assert got.item() == pytest.approx(reference, abs=1e-5)
    mse = prediction_coherence_loss(probs_clean, probs_deg)
    assert mse.item() == pytest.approx(torch.mean((probs_clean[:, 1] - probs_deg[:, 1]) ** 2).item())


def test_gate_modes_and_uniform_type_weights():
    p_clean = torch.tensor([1.0, 0.5])
    types = ["speech", "music"]
    alpha = {"speech": 1.0, "music": 0.3}
    assert robustness_weight(types, alpha, p_clean, 1.0, gate="full").item() == pytest.approx(0.5)
    assert robustness_weight(types, alpha, p_clean, 1.0, gate="no_confidence").item() == pytest.approx(0.65)
    assert robustness_weight(types, alpha, p_clean, 1.0, gate="no_type").item() == pytest.approx(0.5)
    assert robustness_weight(types, alpha, p_clean, 0.4, gate="none").item() == pytest.approx(0.4)
    uniform = robustness_weight(types, alpha, p_clean, 1.0, gate="full", uniform_type_weights=True)
    assert uniform.item() == pytest.approx(0.5)


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
