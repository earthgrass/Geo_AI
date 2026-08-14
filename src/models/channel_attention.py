"""Squeeze-and-Excitation (SE) Channel Attention module.

Reference: Hu et al. (2018) "Squeeze-and-Excitation Networks", CVPR.

Adaptively recalibrates channel-wise feature responses by explicitly
modelling interdependencies between channels. This allows the model
to learn which input channels (precipitation, wind, pressure, DEM, etc.)
are most important for the current prediction.

Architecture:
    Input [B, C, H, W]
      -> Global Average Pooling -> [B, C]
      -> FC(C -> C/r) -> ReLU -> FC(C/r -> C) -> Sigmoid -> [B, C]
      -> Channel-wise multiply with input -> [B, C, H, W]
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel attention for ConvLSTM features.

    Args:
        channels: Number of input feature channels.
        reduction: Reduction ratio for bottleneck. Default 16.
                   Smaller values = more parameters, potentially more expressivity.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(1, channels // reduction)

        self.gap = nn.AdaptiveAvgPool2d(1)  # Global average pooling

        self.excitation = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel attention.

        Args:
            x: Input feature tensor [B, C, H, W].

        Returns:
            Reweighted feature tensor [B, C, H, W].
        """
        b, c, _, _ = x.shape

        # Squeeze: global spatial information into channel descriptor
        y = self.gap(x).view(b, c)  # [B, C]

        # Excitation: learn channel dependencies
        y = self.excitation(y).view(b, c, 1, 1)  # [B, C, 1, 1]

        # Scale: channel-wise multiplication
        return x * y
