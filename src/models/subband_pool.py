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
        # Autocast leaves these maps near the float16 underflow limit. Normalizing
        # them in float16 divides by zero and returns NaN, which makes the M3 loss NaN.
        with torch.amp.autocast(device_type=feat_map.device.type, enabled=False):
            freq_profile = feat_map.float().mean(dim=3)  # (B, C, H)
            height = freq_profile.shape[2]
            bands = []

            # Same bin boundaries as adaptive average pooling; written with slices because
            # adaptive pooling with non-divisible sizes is not implemented on every backend.
            for k in range(self.num_bands):
                start = (k * height) // self.num_bands
                end = -((-(k + 1) * height) // self.num_bands)
                bands.append(freq_profile[:, :, start:end].mean(dim=2))

            signature = torch.stack(bands, dim=1)
            return F.normalize(signature, p=2, dim=-1)
