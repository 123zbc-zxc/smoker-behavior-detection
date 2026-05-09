from __future__ import annotations

import torch
from torch import nn


class SEAttention(nn.Module):
    """Lightweight squeeze-and-excitation attention."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.channels = max(int(channels), 1)
        self.reduction = max(int(reduction), 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1: nn.Conv2d | None = None
        self.act = nn.SiLU(inplace=True)
        self.fc2: nn.Conv2d | None = None
        self.gate = nn.Sigmoid()

    def _build_layers(self, actual_channels: int, device: torch.device, dtype: torch.dtype) -> None:
        reduced_channels = max(actual_channels // self.reduction, 4)
        self.fc1 = nn.Conv2d(actual_channels, reduced_channels, kernel_size=1, bias=True).to(device=device, dtype=dtype)
        self.fc2 = nn.Conv2d(reduced_channels, actual_channels, kernel_size=1, bias=True).to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fc1 is None or self.fc2 is None or self.fc1.in_channels != x.shape[1]:
            self._build_layers(x.shape[1], x.device, x.dtype)
        weights = self.avg_pool(x)
        weights = self.fc1(weights)
        weights = self.act(weights)
        weights = self.fc2(weights)
        weights = self.gate(weights)
        return x * weights
