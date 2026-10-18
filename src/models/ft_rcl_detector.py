import torch
import torch.nn as nn

from src.models.artifact_projector import project_artifact
from src.models.channel_encoder import ChannelEncoder
from src.models.mobilenet_v3_audio import MobileNetV3SmallAudio
from src.models.subband_pool import SubbandPool


class FTRCLDetector(nn.Module):
    def __init__(self, num_classes=2, num_bands=8, use_cad=False):
        super().__init__()
        self.backbone = MobileNetV3SmallAudio(num_classes=num_classes)
        self.subband_pool = SubbandPool(num_bands=num_bands)
        self.use_cad = use_cad
        self.channel_encoder = ChannelEncoder() if use_cad else None

    def forward(self, x, waveforms=None, detach_u_epochs=3, current_epoch=1):
        logits, feat_map = self.backbone(x, return_features=True)

        z = torch.flatten(self.backbone.backbone.avgpool(feat_map), 1)

        if self.use_cad and waveforms is not None:
            u = self.channel_encoder(waveforms)
            detach_u = current_epoch <= detach_u_epochs
            z = project_artifact(z, u, detach_u=detach_u)
            feat_for_sig = z.unsqueeze(-1).unsqueeze(-1)
            signature = self.subband_pool(feat_for_sig)
            return logits, signature, u

        signature = self.subband_pool(feat_map)
        return logits, signature, None
