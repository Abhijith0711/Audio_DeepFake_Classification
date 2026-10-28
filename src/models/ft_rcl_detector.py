import torch.nn as nn

from src.models.mobilenet_v3_audio import MobileNetV3SmallAudio
from src.models.subband_pool import SubbandPool


class FTRCLDetector(nn.Module):
    def __init__(self, num_classes=2, num_bands=8):
        super().__init__()
        self.backbone = MobileNetV3SmallAudio(num_classes=num_classes)
        self.subband_pool = SubbandPool(num_bands=num_bands)

    def forward(self, x, waveforms=None, detach_u_epochs=3, current_epoch=1):
        del waveforms, detach_u_epochs, current_epoch
        logits, feat_map = self.backbone(x, return_features=True)
        signature = self.subband_pool(feat_map)
        return logits, signature, None
