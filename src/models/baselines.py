"""Baseline models for ablation studies.

Provides:
    1. PersistenceBaseline:  P_hat_{t+1} = P_t  (no-change forecast)
    2. PlainConvLSTM:        Simple ConvLSTM with precipitation-only input
    3. ResConvLSTM:          Residual ConvLSTM without attention or physics loss

These baselines are essential for demonstrating that each component
(residual learning, physics-informed loss, channel attention) contributes
meaningfully to prediction quality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from .convlstm_cell import ConvLSTMCell


# ---------------------------------------------------------------------------
# 1. Persistence Baseline (no parameters)
# ---------------------------------------------------------------------------

class PersistenceBaseline:
    """Predicts zero change: P_hat_{t+1} = P_t.

    The simplest possible forecast — assumes the precipitation field
    does not change. Any model that cannot beat persistence is useless
    for operational forecasting.

    Args:
        precip_channel_idx: Index of precipitation channel (default 0).
    """

    def __init__(self, precip_channel_idx: int = 0):
        self.precip_channel_idx = precip_channel_idx

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Return last frame's precipitation as prediction.

        Args:
            x: [B, K, C, H, W] input tensor.

        Returns:
            [B, 1, H, W] prediction (same as P_t).
        """
        return x[:, -1, self.precip_channel_idx:
                 self.precip_channel_idx + 1, :, :]


# ---------------------------------------------------------------------------
# 2. Plain ConvLSTM (no residual, no physics channels)
# ---------------------------------------------------------------------------

class PlainConvLSTM(nn.Module):
    """Simple ConvLSTM: precipitation-only input, direct absolute prediction.

    This is the "pure ML" baseline — ConvLSTM without any architectural
    innovations. Uses only the precipitation channel as input.

    Args:
        hidden_dims: Hidden dimensions for stacked ConvLSTM.
        kernel_size: ConvLSTM kernel size.
    """

    def __init__(
        self,
        hidden_dims: List[int] = None,
        kernel_size: int = 3,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 128]

        self.hidden_dims = hidden_dims
        self.num_layers = len(hidden_dims)

        # Encoder: precipitation-only input (1 channel)
        self.encoder_cells = nn.ModuleList()
        in_ch = 1
        for hd in hidden_dims:
            self.encoder_cells.append(
                ConvLSTMCell(in_ch, hd, kernel_size)
            )
            in_ch = hd

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dims[-1], hidden_dims[0], 3, padding=1),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.ReLU(inplace=True),
        )
        self.pred_head = nn.Conv2d(hidden_dims[0], 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict absolute precipitation.

        Args:
            x: [B, K, 1, H, W] precipitation-only input sequence.

        Returns:
            [B, 1, H, W] absolute precipitation prediction.
        """
        B, K, _, H, W = x.shape
        states = [
            cell.init_hidden(B, (H, W))
            for cell in self.encoder_cells
        ]

        for t in range(K):
            x_t = x[:, t, :, :, :]
            for i, cell in enumerate(self.encoder_cells):
                h_new, c_new = cell(x_t, states[i])
                states[i] = (h_new, c_new)
                x_t = h_new

        h_deep = states[-1][0]
        dec = self.decoder(h_deep)
        return F.relu(self.pred_head(dec))


# ---------------------------------------------------------------------------
# 3. ResConvLSTM (residual learning, no attention, no physics loss)
# ---------------------------------------------------------------------------

class ResConvLSTM(nn.Module):
    """Residual ConvLSTM: temporal residual learning without attention.

    Uses the same architecture as PIResConvLSTM but:
    - No channel attention
    - Trained with plain MSE loss (no physics terms)
    - This serves as the ablation baseline for "does physics loss help?"

    Args:
        input_channels: Number of input channels.
        precip_channel_idx: Index of precipitation channel.
        hidden_dims: Hidden dimensions.
        kernel_size: ConvLSTM kernel size.
    """

    def __init__(
        self,
        input_channels: int = 4,
        precip_channel_idx: int = 0,
        hidden_dims: List[int] = None,
        kernel_size: int = 3,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 128]

        self.input_channels = input_channels
        self.precip_channel_idx = precip_channel_idx
        self.hidden_dims = hidden_dims
        self.num_layers = len(hidden_dims)

        # Encoder
        self.encoder_cells = nn.ModuleList()
        in_ch = input_channels
        for hd in hidden_dims:
            self.encoder_cells.append(ConvLSTMCell(in_ch, hd, kernel_size))
            in_ch = hd

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dims[-1], hidden_dims[0], 3, padding=1),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.ReLU(inplace=True),
        )
        self.pred_head = nn.Conv2d(hidden_dims[0], 1, kernel_size=1)

        # Residual refinement network
        self.refine_net = nn.Sequential(
            nn.Conv2d(hidden_dims[-1] + 1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict precipitation change ΔP.

        Args:
            x: [B, K, C, H, W] input tensor.

        Returns:
            delta_p: [B, 1, H, W] predicted precipitation change.
        """
        B, K, _, H, W = x.shape
        states = [
            cell.init_hidden(B, (H, W))
            for cell in self.encoder_cells
        ]

        for t in range(K):
            x_t = x[:, t, :, :, :]
            for i, cell in enumerate(self.encoder_cells):
                h_new, c_new = cell(x_t, states[i])
                states[i] = (h_new, c_new)
                x_t = h_new

        h_deep = states[-1][0]
        dec_feat = self.decoder(h_deep)
        p_base = self.pred_head(dec_feat)

        refine_in = torch.cat([h_deep, p_base], dim=1)
        delta_p = self.refine_net(refine_in)

        return delta_p

    def compute_prediction(self, x: torch.Tensor) -> torch.Tensor:
        """Compute absolute prediction: P_hat = ReLU(P_t + ΔP)."""
        P_last = x[:, -1, self.precip_channel_idx:
                   self.precip_channel_idx + 1, :, :]
        delta_p = self.forward(x)
        return F.relu(P_last + delta_p)
