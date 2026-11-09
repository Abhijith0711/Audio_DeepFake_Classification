import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn(inp, oup, stride):
    layers = [
        nn.Conv2d(inp, oup, 3, 1, 1, bias=False),
        nn.AvgPool2d(stride),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    ]
    return nn.Sequential(*layers)


def conv_1x1_bn(inp, oup):
    layers = [
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    ]
    return nn.Sequential(*layers)


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            layers = [
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False),
                nn.AvgPool2d(stride),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            ]
        else:
            layers = [
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False),
                nn.AvgPool2d(stride),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class PANNsMobileNetV2Audio(nn.Module):
    """
    PANNs MobileNetV2 Audio Backbone for Deepfake Detection.
    Directly addresses Reviewer Comment #30 / #38 by using an AudioSet pre-trained
    audio model.

    Transfer Learning Architecture:
      - PANNs MobileNetV2 Backbone (features.0 through features.18, 1280 channels)
      - Dual spatial pooling (Global Average + Global Max pooling across time-frequency)
      - Pre-trained 1024-dimensional AudioSet embedding layer (fc1)
      - Binary classification head (1024 -> 2) replacing the original 527-class head
        to output Real (0) vs. Fake (1) probabilities.
    """

    def __init__(
        self,
        pretrained_path=None,
        num_classes=2,
        dropout=0.2,
        auto_download=True
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_rate = dropout

        block = InvertedResidual
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 2],
            [6, 160, 3, 1],
            [6, 320, 1, 1],
        ]

        input_channel = 32
        features = [conv_bn(1, input_channel, 2)]

        for t, c, n, s in inverted_residual_setting:
            output_channel = c
            for i in range(n):
                if i == 0:
                    features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel

        features.append(conv_1x1_bn(input_channel, 1280))
        self.features = nn.Sequential(*features)

        # Pre-trained 1024-dim AudioSet embedding layer
        self.fc1 = nn.Linear(1280, 1024, bias=True)
        self.dropout = nn.Dropout(dropout)

        # Binary classification head: Real (0) vs. Fake (1)
        self.classifier = nn.Linear(1024, num_classes, bias=True)

        if pretrained_path is not None:
            self.load_pretrained(pretrained_path, auto_download=auto_download)

    def load_pretrained(self, pretrained_path, auto_download=True):
        pretrained_path = Path(pretrained_path)

        if not pretrained_path.exists() and auto_download:
            pretrained_path.parent.mkdir(parents=True, exist_ok=True)
            zenodo_url = "https://zenodo.org/records/3987831/files/MobileNetV2_mAP=0.383.pth?download=1"
            print(f"Downloading PANNs MobileNetV2 pre-trained weights from Zenodo to {pretrained_path}...")
            os.system(f'curl -L -o "{pretrained_path}" "{zenodo_url}"')

        if not pretrained_path.exists():
            raise FileNotFoundError(
                f"PANNs pre-trained checkpoint not found at {pretrained_path}"
            )

        ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model", ckpt)

        model_dict = self.state_dict()
        pretrained_dict = {}
        for k, v in state_dict.items():
            if k in model_dict and model_dict[k].shape == v.shape:
                pretrained_dict[k] = v

        loaded_count = len(pretrained_dict)
        total_count = len(model_dict)
        print(
            f"Successfully loaded {loaded_count}/{total_count} pre-trained weight tensors from {pretrained_path}"
        )
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def forward(self, x, return_features=False):
        # Input shape: [Batch, 1, Mel, Time]
        feat = self.features(x)
        feat_map = feat

        # Dual global pooling (Mean + Max pooling over spatial dimensions)
        x_mean = torch.mean(feat, dim=(2, 3))
        x_max, _ = torch.max(feat.flatten(2), dim=2)
        x = x_mean + x_max

        # Acoustic embedding & projection
        x = F.relu_(self.fc1(x))
        x = self.dropout(x)
        logits = self.classifier(x)

        if return_features:
            return logits, feat_map
        return logits

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


def conv_dw_v1(inp, oup, stride):
    layers = [
        nn.Conv2d(inp, inp, 3, 1, 1, groups=inp, bias=False),
        nn.AvgPool2d(stride),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    ]
    return nn.Sequential(*layers)


class PANNsMobileNetV1Audio(nn.Module):
    """
    PANNs MobileNetV1 Audio Backbone for Deepfake Detection.
    """

    def __init__(
        self,
        pretrained_path=None,
        num_classes=2,
        dropout=0.2,
        auto_download=True
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_rate = dropout

        self.features = nn.Sequential(
            conv_bn(1, 32, 2),
            conv_dw_v1(32, 64, 1),
            conv_dw_v1(64, 128, 2),
            conv_dw_v1(128, 128, 1),
            conv_dw_v1(128, 256, 2),
            conv_dw_v1(256, 256, 1),
            conv_dw_v1(256, 512, 2),
            conv_dw_v1(512, 512, 1),
            conv_dw_v1(512, 512, 1),
            conv_dw_v1(512, 512, 1),
            conv_dw_v1(512, 512, 1),
            conv_dw_v1(512, 512, 1),
            conv_dw_v1(512, 1024, 2),
            conv_dw_v1(1024, 1024, 1),
        )

        self.fc1 = nn.Linear(1024, 1024, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(1024, num_classes, bias=True)

        if pretrained_path is not None:
            self.load_pretrained(pretrained_path, auto_download=auto_download)

    def load_pretrained(self, pretrained_path, auto_download=True):
        pretrained_path = Path(pretrained_path)

        if not pretrained_path.exists() and auto_download:
            pretrained_path.parent.mkdir(parents=True, exist_ok=True)
            zenodo_url = "https://zenodo.org/records/3987831/files/MobileNetV1_mAP=0.389.pth?download=1"
            print(f"Downloading PANNs MobileNetV1 pre-trained weights from Zenodo to {pretrained_path}...")
            os.system(f'curl -L -o "{pretrained_path}" "{zenodo_url}"')

        if not pretrained_path.exists():
            raise FileNotFoundError(
                f"PANNs pre-trained checkpoint not found at {pretrained_path}"
            )

        ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model", ckpt)

        model_dict = self.state_dict()
        pretrained_dict = {}
        for k, v in state_dict.items():
            if k in model_dict and model_dict[k].shape == v.shape:
                pretrained_dict[k] = v

        loaded_count = len(pretrained_dict)
        total_count = len(model_dict)
        print(
            f"Successfully loaded {loaded_count}/{total_count} pre-trained weight tensors from {pretrained_path}"
        )
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def forward(self, x, return_features=False):
        feat = self.features(x)
        feat_map = feat
        x_mean = torch.mean(feat, dim=(2, 3))
        x_max, _ = torch.max(feat.flatten(2), dim=2)
        x = x_mean + x_max
        x = F.relu_(self.fc1(x))
        x = self.dropout(x)
        logits = self.classifier(x)
        if return_features:
            return logits, feat_map
        return logits

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


# Default alias: MobileNetV2 is the designated backbone for E3
PANNsMobileNetAudio = PANNsMobileNetV2Audio
