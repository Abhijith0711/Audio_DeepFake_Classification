import torch
import torch.nn as nn
import torch.nn.functional as F


class SubbandPool(nn.Module):
    """Pool a (B, C, F, T) feature map into `num_bands` frequency-band descriptors.

    MobileNetV3-Small downsamples frequency by 32, so an 80-bin mel input leaves
    only 3 rows at the last layer. Fixed-size slicing then produces empty (NaN)
    bands, so bands are made with adaptive average pooling over frequency, which
    never yields an empty band (neighbouring bands overlap when F < num_bands).
    """

    def __init__(self, num_bands=8):
        super().__init__()
        self.num_bands = num_bands

    def forward(self, feat_map):
        freq_profile = feat_map.mean(dim=3)
        bands = F.adaptive_avg_pool1d(freq_profile, self.num_bands)
        signature = bands.transpose(1, 2)
        return F.normalize(signature, p=2, dim=-1)
