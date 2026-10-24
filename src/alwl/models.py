"""Beam-selection classifiers. Each exposes ``embed`` (penultimate features) and ``fc``."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class ImageNet(nn.Module):
    """Standard torchvision ResNet18 on (3, 90, 160) camera frames.

    ``pretrained`` starts the backbone from ImageNet weights and normalises inputs with
    the ImageNet channel statistics those weights expect; the head is always fresh.
    """

    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(self, K: int, pretrained: bool = False):
        super().__init__()
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.backbone = nn.Sequential(*list(net.children())[:-1])
        self.fc = nn.Linear(512, K)
        self.pretrained = pretrained
        self.register_buffer("mean", torch.tensor(self.MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(self.STD).view(1, 3, 1, 1))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        if self.pretrained:
            x = (x - self.mean) / self.std
        return self.backbone(x).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.embed(x))


class _Residual3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(channels),
            nn.ReLU(),
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x + self.block(x))


class LidarNet(nn.Module):
    """Residual 3-D CNN on (1, 20, 20, 20) LiDAR occupancy grids."""

    def __init__(self, K: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            _Residual3D(32),
            nn.Conv3d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            _Residual3D(64),
            nn.Conv3d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
        )
        self.fc = nn.Linear(128, K)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.embed(x))


def build_model(modality: str, K: int, pretrained: bool = False) -> nn.Module:
    if modality == "image":
        return ImageNet(K, pretrained)
    if modality == "lidar":
        return LidarNet(K)
    raise ValueError(f"Unknown modality {modality!r}; use 'image' or 'lidar'")
