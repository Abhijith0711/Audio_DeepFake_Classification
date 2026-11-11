import numpy as np
import pandas as pd
import pytest
import torch

from torch.utils.data import DataLoader

from src.datasets.atadd_manifest_dataset import collate_paired_batch, stratified_indices
from src.datasets.paired_robustness_dataset import PairedRobustnessDataset
from src.models.ft_rcl_detector import FTRCLDetector
from src.models.subband_pool import SubbandPool
from src.robustness.conditions import ConditionSpec, all_training_conditions
from src.robustness.degrade import degrade, ffmpeg_available
from src.training.losses.ft_rcl import ft_rcl_loss

SR = 24000


def test_subband_pool_stays_finite_under_fp16_autocast():
    feat = (torch.randn(2, 576, 3, 13) * 1e-7).half()
    with torch.amp.autocast("cpu", dtype=torch.float16):
        sig = SubbandPool(8)(feat)
    assert torch.isfinite(sig).all()
    assert sig.dtype == torch.float32


def test_subband_pool_never_returns_empty_bands():
    for height in (1, 3, 5, 8, 20):
        sig = SubbandPool(8)(torch.randn(2, 16, height, 7))
        assert sig.shape == (2, 8, 16)
        assert torch.isfinite(sig).all()
        assert torch.allclose(sig.norm(dim=-1), torch.ones(2, 8), atol=1e-5)


def test_eight_bands_are_distinct_on_an_80_bin_input_and_finite_under_autocast():
    rows = torch.zeros(1, 24, 10, 4)
    for height in range(10):
        rows[0, height, height, :] = 1.0
    pooled = SubbandPool(8)(rows)
    assert torch.all(torch.pdist(pooled[0]) > 1e-4)

    model = FTRCLDetector(signature_stage=3)
    ramp = torch.linspace(-1, 1, 80).view(1, 1, 80, 1).expand(1, 1, 80, 32).contiguous()
    _logits, _feat, mid = model.backbone(ramp, tap=3)
    assert mid.shape[1] == 24
    assert mid.shape[2] >= 8

    with torch.amp.autocast("cpu", dtype=torch.float16):
        logits, signature, _ = model(ramp)
    assert signature.shape == (1, 8, 24)
    assert signature.dtype == torch.float32
    assert torch.isfinite(signature).all()
    assert torch.isfinite(logits.float()).all()
    assert torch.pdist(signature[0].float()).max() > 1e-3


def test_real_backbone_signature_and_loss_are_finite():
    model = FTRCLDetector()
    x = torch.randn(4, 1, 80, 401)
    logits_c, sig_c, _ = model(x)
    logits_d, sig_d, _ = model(x + 0.1 * torch.randn_like(x))
    assert torch.isfinite(sig_c).all()

    labels = torch.tensor([0, 1, 0, 1])
    loss, parts = ft_rcl_loss(
        logits_c, logits_d, labels, sig_c, sig_d, ["speech"] * 4, 1.0, {"speech": 1.0}
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_loss_runs_in_fp32_for_half_inputs():
    logits = torch.randn(4, 2).half()
    sig = torch.nn.functional.normalize(torch.randn(4, 8, 5), dim=-1).half()
    loss, _ = ft_rcl_loss(logits, logits, torch.tensor([0, 1, 0, 1]), sig, sig,
                          ["speech"] * 4, 1.0, {})
    assert loss.dtype == torch.float32


def test_deterministic_noise_differs_per_clip_but_is_reproducible():
    cond = next(c for c in all_training_conditions() if c.family == "additive_noise")
    wave = np.random.default_rng(0).standard_normal(SR).astype(np.float32) * 0.1

    a, _ = degrade(wave, SR, cond, mode="deterministic", source_id="clip_a")
    a2, _ = degrade(wave, SR, cond, mode="deterministic", source_id="clip_a")
    b, _ = degrade(wave, SR, cond, mode="deterministic", source_id="clip_b")

    assert np.array_equal(a, a2)
    assert not np.allclose(a - wave, b - wave)


@pytest.mark.skipif(ffmpeg_available(), reason="only meaningful without ffmpeg")
def test_mp3_without_ffmpeg_raises_instead_of_returning_clean_audio():
    cond = ConditionSpec("mp3_codec", "mp3_128k", 2, {"bitrate_kbps": 128})
    with pytest.raises(RuntimeError, match="ffmpeg"):
        degrade(np.zeros(SR, dtype=np.float32), SR, cond)


def test_stratified_indices_cover_every_type_and_label():
    types = ["music", "speech", "sound", "singing"]
    rows = [{"type": t, "label": l} for t in types for l in ("fake", "real") for _ in range(50)]
    df = pd.DataFrame(rows)

    head = df.iloc[:100]
    assert head["type"].nunique() < 4  # sorted manifest: first rows are not representative

    picked = df.iloc[stratified_indices(df, 80, seed=1)]
    assert len(picked.groupby(["type", "label"])) == 8
    assert stratified_indices(df, 80, seed=1) == stratified_indices(df, 80, seed=1)


class _SyntheticPaired(PairedRobustnessDataset):
    """Synthetic audio and no MP3 conditions, so it runs without audio files or ffmpeg."""

    @property
    def condition_pool(self):
        return [
            c
            for c in all_training_conditions()
            if c.family in ("resample", "additive_noise", "telephony") and c.severity <= self.max_severity
        ]

    def _load_segment(self, row, crop_start=None):
        return np.random.default_rng(0).standard_normal(self.num_samples).astype(np.float32) * 0.1, 0


def _paired_df(n=32):
    return pd.DataFrame({"name": [f"f{i}.flac" for i in range(n)], "file_id": [f"f{i}" for i in range(n)],
                         "label": ["real"] * n, "type": ["speech"] * n, "split": ["train"] * n})


def test_set_max_severity_reaches_persistent_workers():
    ds = _SyntheticPaired(_paired_df(), ".", sample_rate=2000, segment_seconds=1, max_severity=4)
    loader = DataLoader(ds, batch_size=8, num_workers=2, persistent_workers=True,
                        collate_fn=collate_paired_batch)

    first_epoch = {m["severity"] for batch in loader for m in batch[4]}
    assert max(first_epoch) > 1

    ds.set_max_severity(1)  # main-process update, as Trainer.fit does each epoch
    second_epoch = {m["severity"] for batch in loader for m in batch[4]}
    assert second_epoch == {1}


def test_workers_use_different_rng_streams():
    ds = _SyntheticPaired(_paired_df(), ".", sample_rate=2000, segment_seconds=1, max_severity=4)
    loader = DataLoader(ds, batch_size=4, num_workers=2, shuffle=False, collate_fn=collate_paired_batch)
    names = [m["condition_name"] for batch in loader for m in batch[4]]
    worker0, worker1 = names[0:4] + names[8:12], names[4:8] + names[12:16]
    assert worker0 != worker1
