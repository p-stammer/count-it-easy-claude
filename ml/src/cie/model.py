"""The digit classifier.

Same tiny architecture as the original Keras models (~18k parameters), but a
single network with 9 outputs (digits 1-9 grouped by direction) instead of
three separate 3-class networks: every crop now trains the shared features,
which triples the effective data per class family. At inference only the 3
logits of the crop's direction are used.
"""
from __future__ import annotations

import torch
from torch import nn


def _conv(cin: int, cout: int) -> nn.Module:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(inplace=True))


class DigitNet(nn.Module):
    def __init__(self, width: int = 1):
        super().__init__()
        c = [4 * width, 8 * width, 16 * width, 32 * width]
        self.features = nn.Sequential(
            _conv(3, c[0]), _conv(c[0], c[0]), nn.AvgPool2d(2), nn.BatchNorm2d(c[0]),
            _conv(c[0], c[1]), _conv(c[1], c[1]), nn.AvgPool2d(2), nn.BatchNorm2d(c[1]),
            _conv(c[1], c[2]), nn.AvgPool2d(2), nn.BatchNorm2d(c[2]),
            _conv(c[2], c[3]), nn.AvgPool2d(2), nn.BatchNorm2d(c[3]),
            _conv(c[3], c[3]),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.head = nn.Sequential(nn.Linear(c[3], 30), nn.ReLU(inplace=True), nn.Linear(30, 9))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, 3, 128, 128) in [0, 1] -> (N, 9) logits."""
        return self.head(self.features(x))


def direction_logits(logits: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Pick the 3 logits belonging to each sample's direction -> (N, 3)."""
    idx = direction[:, None] * 3 + torch.arange(3, device=logits.device)[None]
    return logits.gather(1, idx)
