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

    def forward(self, x, return_features=False, tap=None):
        """Logits, optionally with the last feature map and the output of block ``tap``.

        ``tap`` is a feature-block index. FT-RCL uses it to read a signature from an
        earlier block, where the frequency axis still has enough rows for separate bands.
        """
        mid = None

        if tap is None:
            x = self.backbone.features(x)
        else:
            for index, block in enumerate(self.backbone.features):
                x = block(x)
                if index == tap:
                    mid = x
            if mid is None:
                raise ValueError(
                    f"signature tap {tap} is outside MobileNetV3-Small "
                    f"({len(self.backbone.features)} feature blocks)"
                )

        feat_map = x
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.backbone.classifier(x)

        if tap is not None:
            return logits, feat_map, mid

        if return_features:
            return logits, feat_map

        return logits

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


class MobileNetV3SmallGRUAudio(nn.Module):
    def __init__(
        self,
        num_classes=2,
        gru_hidden_size=128,
        gru_layers=1,
        dropout=0.2
    ):
        super().__init__()
        backbone = mobilenet_v3_small(weights=None)
        old_conv = backbone.features[0][0]
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
        backbone.features[0][0] = new_conv

        self.feature_extractor = backbone.features
        self.feature_dim = backbone.classifier[0].in_features
        self.frequency_pool = nn.AdaptiveAvgPool2d((1, None))
        self.gru = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=gru_hidden_size,
            num_layers=gru_layers,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(gru_hidden_size, num_classes)
        self.gru_hidden_size = gru_hidden_size
        self.gru_layers = gru_layers

    def extract_sequence(self, x):
        x = self.feature_extractor(x)
        x = self.frequency_pool(x)
        x = x.squeeze(2)
        x = x.transpose(1, 2)
        return x

    def forward(self, x):
        sequence = self.extract_sequence(x)
        _, hidden = self.gru(sequence)
        representation = hidden[-1]
        logits = self.classifier(self.dropout(representation))
        return logits

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

