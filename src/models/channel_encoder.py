import numpy as np
import torch
import torch.nn as nn


def waveform_channel_stats(waveform):
    x = waveform.detach().cpu().numpy().astype(np.float64)
    rms = np.sqrt(np.mean(x ** 2) + 1e-12)
    crest = np.max(np.abs(x)) / (rms + 1e-12)
    flatness = np.exp(np.mean(np.log(np.abs(x) + 1e-12))) / (np.mean(np.abs(x)) + 1e-12)
    zcr = np.mean(np.abs(np.diff(np.sign(x)))) / 2.0
    return np.array([rms, crest, flatness, zcr], dtype=np.float32)


class ChannelEncoder(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=16, output_dim=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, waveforms):
        stats = []

        for waveform in waveforms:
            stats.append(waveform_channel_stats(waveform))

        stats_tensor = torch.tensor(np.stack(stats), device=waveforms.device, dtype=torch.float32)
        return self.mlp(stats_tensor)
