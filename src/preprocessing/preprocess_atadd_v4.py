"""
AT-ADD v4 preprocessing for the E1-v4 MobileNetV3-Small experiment.

v4 changes relative to the original E1 frontend:
- 16 kHz -> 24 kHz audio
- zero padding -> repeat-based padding for short clips
- 4 s target duration remains unchanged
- GPU STFT + Mel extraction remains unchanged

This module does not contain:
- dataset split/manifest creation
- model definitions
- training loops
- robustness transformations
- compression/edge deployment logic
"""

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn


def load_audio(path,target_sr=24000):
    waveform,sr=sf.read(path,dtype="float32")

    if waveform.ndim>1:
        waveform=waveform.mean(axis=1)

    if sr!=target_sr:
        waveform=librosa.resample(waveform,orig_sr=sr,target_sr=target_sr)

    waveform=waveform.astype(np.float32)
    waveform=np.nan_to_num(waveform,nan=0.0,posinf=0.0,neginf=0.0)

    return waveform


def repeat_pad_audio(waveform,num_samples):
    if len(waveform)==0:
        return np.zeros(num_samples,dtype=np.float32)

    if len(waveform)>=num_samples:
        return waveform[:num_samples]

    repeats=int(np.ceil(num_samples/len(waveform)))

    padded=np.tile(waveform,repeats)

    return padded[:num_samples]


def fixed_length_audio(waveform,num_samples,training=False):
    if len(waveform)>=num_samples:

        if training:
            start=np.random.randint(0,len(waveform)-num_samples+1)
        else:
            start=(len(waveform)-num_samples)//2

        waveform=waveform[start:start+num_samples]

    else:
        waveform=repeat_pad_audio(waveform,num_samples)

    return waveform.astype(np.float32)


def cpu_prepare_waveform(path,sample_rate=24000,segment_seconds=4,training=False):
    num_samples=(sample_rate*segment_seconds)

    waveform=load_audio(path,sample_rate)

    waveform=fixed_length_audio(waveform,num_samples,training)

    return torch.from_numpy(waveform).float()


class GPULogMel(nn.Module):
    def __init__(self,sample_rate,n_fft,win_length,hop_length,n_mels,log_eps=1e-10):
        super().__init__()

        mel_filter=librosa.filters.mel(sr=sample_rate,n_fft=n_fft,n_mels=n_mels,fmin=0.0,fmax=None,htk=False,norm="slaney")

        window=torch.hann_window(win_length)

        self.register_buffer("mel_filter",torch.tensor(mel_filter,dtype=torch.float32))

        self.register_buffer("window",window)

        self.n_fft=n_fft
        self.win_length=win_length
        self.hop_length=hop_length
        self.log_eps=log_eps

    def forward(self,waveforms):

        stft=torch.stft(waveforms,n_fft=self.n_fft,hop_length=self.hop_length,win_length=self.win_length,window=self.window,center=True,pad_mode="reflect",return_complex=True)

        power=stft.abs().pow(2)

        mel=torch.einsum("mf,bft->bmt",self.mel_filter,power)

        log_mel=torch.log(torch.clamp(mel,min=self.log_eps))

        return log_mel


def normalize_log_mel(log_mel,mean,std):
    return (log_mel-mean)/std


@torch.no_grad()
def compute_logmel_stats_gpu(loader,gpu_logmel,device,max_batches=None):
    total_sum=torch.tensor(0.0,device=device,dtype=torch.float64)

    total_sq_sum=torch.tensor(0.0,device=device,dtype=torch.float64)

    total_count=0

    for batch_index,(waveforms,_,_) in enumerate(loader):

        if max_batches is not None and batch_index>=max_batches:
            break

        waveforms=waveforms.to(device, non_blocking=True)

        log_mel=gpu_logmel(waveforms).to(torch.float64)

        total_sum+=log_mel.sum()
        total_sq_sum+=(log_mel**2).sum()
        total_count+=log_mel.numel()

    mean=total_sum/total_count

    variance=torch.clamp(total_sq_sum/total_count-mean**2,min=1e-8)

    std=torch.sqrt(variance)

    return (float(mean.cpu()),float(std.cpu()))
