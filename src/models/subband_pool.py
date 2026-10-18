import torch
import torch.nn as nn
import torch.nn.functional as F


class SubbandPool(nn.Module):
    def __init__(self, num_bands=8):
        super().__init__()
        self.num_bands = num_bands

    def forward(self, feat_map):
        _, _, height, _ = feat_map.shape
        band_size = max(1, height // self.num_bands)
        bands = []

        for band_index in range(self.num_bands):
            start = band_index * band_size
            end = height if band_index == self.num_bands - 1 else (band_index + 1) * band_size
            band = feat_map[:, :, start:end, :].mean(dim=(2, 3))
            bands.append(band)

        signature = torch.stack(bands, dim=1)
        return F.normalize(signature, p=2, dim=-1)
