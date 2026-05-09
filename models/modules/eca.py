from __future__ import annotations

import math

import torch
from torch import nn


class ECA(nn.Module):
    """Efficient Channel Attention for lightweight channel reweighting."""

    def __init__(self, channels: int, gamma: int = 2, b: int = 1) -> None:
        super().__init__()
        kernel = int(abs((math.log2(max(channels, 1)) + b) / gamma))
        kernel = kernel if kernel % 2 else kernel + 1
        kernel = max(kernel, 3)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel, padding=kernel // 2, bias=False)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.avg_pool(x)
        weights = weights.squeeze(-1).transpose(-1, -2)
        weights = self.conv(weights)
        weights = self.activation(weights.transpose(-1, -2).unsqueeze(-1))
        return x * weights.expand_as(x)
