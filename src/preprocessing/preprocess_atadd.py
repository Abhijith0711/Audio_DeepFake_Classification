"""
AT-ADD preprocessing for E1/E2/E3 experiments.

This module contains the model-input audio preprocessing used by the
AT-ADD baseline experiments. It intentionally does not contain:
- dataset split/manifest creation
- model definitions
- training loops
- robustness transformations
- compression/edge deployment logic

Pipeline:
    audio file -> mono -> 16 kHz -> fixed 4 s waveform
    -> GPU STFT -> 80-bin Mel -> log power
    -> training-set normalization (normalization stats supplied externally)
"""

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn


def load_audio(path,target_sr=16000):
    """Load an audio file, convert to mono, and resample to target_sr."""
    waveform,sr=sf.read(path,dtype="float32")

    if waveform.ndim>1:
        waveform=waveform.mean(axis=1)

    if sr!=target_sr:
        waveform=librosa.resample(
            waveform,
            orig_sr=sr,
            target_sr=target_sr
        )

    waveform=waveform.astype(np.float32)
    waveform=np.nan_to_num(
        waveform,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return waveform


def fixed_length_audio(waveform,num_samples,training=False):
    """
    Make an audio waveform exactly num_samples long.

    Training uses a random crop for clips longer than the target.
    Validation/evaluation use a deterministic center crop.
    Shorter clips are zero-padded.
    """
    if len(waveform)>=num_samples:

        if training:
            start=np.random.randint(
                0,
                len(waveform)-num_samples+1
            )
        else:
            start=(len(waveform)-num_samples)//2

        waveform=waveform[start:start+num_samples]

    else:
        padded=np.zeros(
            num_samples,
            dtype=np.float32
        )

        padded[:len(waveform)]=waveform
        waveform=padded

    return waveform


def cpu_prepare_waveform(
    path,
    sample_rate=16000,
    segment_seconds=4,
    training=False
):
    """Load and prepare one waveform for batched GPU feature extraction."""
    num_samples=sample_rate*segment_seconds

    waveform=load_audio(
        path,
        sample_rate
    )

    waveform=fixed_length_audio(
        waveform,
        num_samples,
        training
    )

    return torch.from_numpy(waveform).float()


class GPULogMel(nn.Module):
    """
    Batched GPU log-Mel extraction.

    The Mel filterbank is created with librosa and stored as a buffer.
    STFT, power computation, Mel projection and log compression run on
    the device where this module is placed.
    """

    def __init__(
        self,
        sample_rate,
        n_fft,
        win_length,
        hop_length,
        n_mels,
        log_eps=1e-10
    ):
        super().__init__()

        mel_filter=librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=0.0,
            fmax=None,
            htk=False,
            norm="slaney"
        )

        window=torch.hann_window(win_length)

        self.register_buffer(
            "mel_filter",
            torch.tensor(mel_filter,dtype=torch.float32)
        )

        self.register_buffer(
            "window",
            window
        )

        self.n_fft=n_fft
        self.win_length=win_length
        self.hop_length=hop_length
        self.log_eps=log_eps

    def forward(self,waveforms):

        stft=torch.stft(
            waveforms,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            pad_mode="reflect",
            return_complex=True
        )

        power=stft.abs().pow(2)

        mel=torch.einsum(
            "mf,bft->bmt",
            self.mel_filter,
            power
        )

        log_mel=torch.log(
            torch.clamp(
                mel,
                min=self.log_eps
            )
        )

        return log_mel


def normalize_log_mel(log_mel,mean,std):
    """Apply fixed training-set log-Mel normalization."""
    return (log_mel-mean)/std


@torch.no_grad()
def compute_logmel_stats_gpu(
    loader,
    gpu_logmel,
    device,
    max_batches=None
):
    """
    Compute global mean/std from the training log-Mel representation.

    The loader should contain training audio only and should use
    deterministic/non-random cropping for statistics calculation.
    """
    total_sum=torch.tensor(
        0.0,
        device=device,
        dtype=torch.float64
    )

    total_sq_sum=torch.tensor(
        0.0,
        device=device,
        dtype=torch.float64
    )

    total_count=0

    for batch_index,(waveforms,_,_) in enumerate(loader):

        if max_batches is not None and batch_index>=max_batches:
            break

        waveforms=waveforms.to(
            device,
            non_blocking=True
        )

        log_mel=gpu_logmel(
            waveforms
        ).to(torch.float64)

        total_sum+=log_mel.sum()
        total_sq_sum+=(log_mel**2).sum()
        total_count+=log_mel.numel()

    mean=total_sum/total_count

    variance=torch.clamp(
        total_sq_sum/total_count-mean**2,
        min=1e-8
    )

    std=torch.sqrt(variance)

    return float(mean.cpu()),float(std.cpu())
