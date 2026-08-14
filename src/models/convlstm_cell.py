"""ConvLSTM cell implementation.

Reference: Shi et al. (2015) "Convolutional LSTM Network: A Machine Learning
Approach for Precipitation Nowcasting", NeurIPS.

The ConvLSTM cell replaces the fully-connected transforms in standard
LSTM with convolutional operations, preserving spatial structure.
"""

import torch
import torch.nn as nn
from typing import Tuple


class ConvLSTMCell(nn.Module):
    """Single ConvLSTM cell.

    Performs the following operations (all using convolution):
        i_t = sigmoid(W_xi * X_t + W_hi * H_{t-1} + b_i)
        f_t = sigmoid(W_xf * X_t + W_hf * H_{t-1} + b_f)
        o_t = sigmoid(W_xo * X_t + W_ho * H_{t-1} + b_o)
        g_t = tanh(W_xg * X_t + W_hg * H_{t-1} + b_g)
        C_t = f_t * C_{t-1} + i_t * g_t
        H_t = o_t * tanh(C_t)

    where * denotes convolution and the four gates are computed in a single
    fused convolution for efficiency.

    Args:
        input_channels: Number of input feature channels.
        hidden_channels: Number of hidden/cell state channels.
        kernel_size: Convolutional kernel size.
        bias: Whether to include bias in convolutions.
        layer_norm: Apply GroupNorm after gate convolution.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
        bias: bool = True,
        layer_norm: bool = False,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.layer_norm = layer_norm

        padding = kernel_size // 2

        # Fused convolution: produces 4 * hidden_channels (i, f, o, g gates)
        self.conv = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

        if layer_norm:
            self.norm = nn.GroupNorm(4, 4 * hidden_channels)

    def forward(
        self,
        x: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for one timestep.

        Args:
            x: Input tensor [B, input_channels, H, W].
            state: Tuple of (h, c) each [B, hidden_channels, H, W].

        Returns:
            Tuple of (h_next, c_next) each [B, hidden_channels, H, W].
        """
        h_prev, c_prev = state

        # Concatenate input with previous hidden state along channel dim
        combined = torch.cat([x, h_prev], dim=1)  # [B, C_in + C_hid, H, W]

        # Fused gate computation
        gates = self.conv(combined)  # [B, 4 * C_hid, H, W]

        if self.layer_norm:
            gates = self.norm(gates)

        # Split into four gates
        i, f, o, g = torch.split(gates, self.hidden_channels, dim=1)

        i = torch.sigmoid(i)  # Input gate
        f = torch.sigmoid(f)  # Forget gate
        o = torch.sigmoid(o)  # Output gate
        g = torch.tanh(g)     # Candidate cell state

        # Cell state update
        c_next = f * c_prev + i * g

        # Hidden state update
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(
        self,
        batch_size: int,
        spatial_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden and cell states with zeros.

        Args:
            batch_size: Batch size.
            spatial_size: (height, width) of the spatial grid.

        Returns:
            Tuple of (h0, c0) zero tensors.
        """
        device = self.conv.weight.device
        shape = (batch_size, self.hidden_channels, *spatial_size)
        return (
            torch.zeros(shape, device=device),
            torch.zeros(shape, device=device),
        )
