import torch
import torch.nn as nn
import torch.nn.functional as F


class SubbandPool(nn.Module):
    """Pool a (B, C, F, T) feature map into `num_bands` L2-normalized band vectors.

    Band edges match adaptive average pooling, so a band is never empty. When F
    is smaller than `num_bands`, neighbours overlap and some vectors match. The
    FT-RCL signature is therefore taken from an earlier block (about 10 frequency
    rows), not the last map (3 rows for 80 mel bins). Pooling stays in FP32:
    float16 normalization underflows and returns NaN.
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
