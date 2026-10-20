import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small


class MobileNetV3SmallAudio(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.backbone = mobilenet_v3_small(weights=None, num_classes=num_classes)

        old_conv = self.backbone.features[0][0]

        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            dilation=old_conv.dilation,
            groups=old_conv.groups,
            bias=False,
        )

        self.backbone.features[0][0] = new_conv

    def forward(self, x, return_features=False):
        x = self.backbone.features(x)
        feat_map = x
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.backbone.classifier(x)

        if return_features:
            return logits, feat_map

        return logits

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
