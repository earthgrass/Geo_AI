"""Minimal TrajGRU baseline for precipitation nowcasting.

Implements the core idea of Shi et al. (2017), "Deep Learning for Precipitation
Nowcasting: A Benchmark and A New Model" (NeurIPS): location-variant recurrent
connections via a learned warp of the previous hidden state.

This is a MINIMAL, reproducible implementation (precipitation-only, no TC or
terrain inputs). It is intentionally kept close to the classic formulation and
does NOT introduce novel variants.

Reference:
    Shi, X., Gao, Z., Lausen, L., Wang, H., Yeung, D.-Y., Wong, W.-K., & Woo, W.-C.
    (2017). Deep Learning for Precipitation Nowcasting: A Benchmark and A New Model.
    NeurIPS. arXiv:1706.03458.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


def _warp(x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Bilinear-sample `x` according to a 2-channel (dy, dx) flow field.

    Args:
        x: [B, C, H, W] tensor to warp.
        flow: [B, 2, H, W] where channel 0 = dy (row offset), channel 1 = dx.

    Returns:
        Warped tensor of the same shape as `x`.
    """
    B, C, H, W = x.shape
    # Normalized grid in [-1, 1].
    gy, gx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, H, device=x.device),
        torch.linspace(-1.0, 1.0, W, device=x.device),
        indexing="ij",
    )
    base = torch.stack([gx, gy], dim=0)[None]  # [1, 2, H, W]
    # flow is (dy, dx); grid expects (x, y) so swap order.
    flow_xy = torch.stack([flow[:, 1], flow[:, 0]], dim=1)  # [B, 2, H, W]
    grid = base + flow_xy * 2.0 / torch.tensor([W, H], device=x.device).view(1, 2, 1, 1)
    grid = grid.permute(0, 2, 3, 1)  # [B, H, W, 2]
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)


class TrajGRUCell(nn.Module):
    """Single trajectory-GRU cell with a learned warp of the previous state."""

    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.hidden_channels = hidden_channels

        # Flow generator: (input + hidden) -> 2 (dy, dx).
        self.flow_conv = nn.Conv2d(input_channels + hidden_channels, 2, kernel_size=5, padding=2)

        # Input-to-hidden gate convolution.
        self.conv_x = nn.Conv2d(input_channels, 3 * hidden_channels, kernel_size, padding=pad)
        # Hidden-to-hidden gate convolution (applied to the WARPED hidden state).
        self.conv_h = nn.Conv2d(hidden_channels, 3 * hidden_channels, kernel_size, padding=pad)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # Warp the previous hidden state.
        flow = self.flow_conv(torch.cat([x, h], dim=1))
        h_warped = _warp(h, flow)

        gates = self.conv_x(x) + self.conv_h(h_warped)
        z, r, c = torch.split(gates, self.hidden_channels, dim=1)
        z = torch.sigmoid(z)
        r = torch.sigmoid(r)
        c = torch.tanh(c)
        h_new = (1.0 - z) * h_warped + z * c
        return h_new


class TrajGRU(nn.Module):
    """Stacked TrajGRU encoder-decoder for precipitation-only nowcasting.

    Input:  [B, K, 1, H, W]  (precipitation-only sequence)
    Output: [B, 1, H, W]     (next-frame precipitation)
    """

    def __init__(
        self,
        input_channels: int = 1,
        hidden_dims: List[int] = None,
        kernel_size: int = 3,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 128]
        self.hidden_dims = hidden_dims
        self.num_layers = len(hidden_dims)

        self.cells = nn.ModuleList()
        in_ch = input_channels
        for hd in hidden_dims:
            self.cells.append(TrajGRUCell(in_ch, hd, kernel_size))
            in_ch = hd

        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dims[-1], hidden_dims[0], 3, padding=1),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(hidden_dims[0], 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, K, C, H, W = x.shape
        states = [torch.zeros(B, hd, H, W, device=x.device) for hd in self.hidden_dims]

        for t in range(K):
            x_t = x[:, t]
            for i, cell in enumerate(self.cells):
                states[i] = cell(x_t, states[i])
                x_t = states[i]

        out = self.decoder(states[-1])
        return F.relu(self.head(out))
