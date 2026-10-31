import torch.nn as nn

from src.models.mobilenet_v3_audio import MobileNetV3SmallAudio
from src.models.subband_pool import SubbandPool


class FTRCLDetector(nn.Module):
    """MobileNetV3-Small plus a band signature from an intermediate block.

    ``signature_stage`` is the feature-block index. Stage 3 has stride 8 in
    frequency: about 10 rows and 24 channels for an 80-bin mel input, so eight
    bands are different mixtures of those rows. The last block has only 3 rows.
    """

    def __init__(self, num_classes=2, num_bands=8, signature_stage=3):
        super().__init__()
        self.backbone = MobileNetV3SmallAudio(num_classes=num_classes)
        self.subband_pool = SubbandPool(num_bands=num_bands)
        self.signature_stage = int(signature_stage)

    def forward(self, x, waveforms=None, detach_u_epochs=3, current_epoch=1):
        del waveforms, detach_u_epochs, current_epoch
        logits, _feat_map, mid = self.backbone(x, tap=self.signature_stage)
        signature = self.subband_pool(mid)
        return logits, signature, None
