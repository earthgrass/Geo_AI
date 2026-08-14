"""PI-ResConvLSTM: Physics-informed Residual ConvLSTM.

Core model for typhoon precipitation simulation with TRUE temporal
residual learning. The model predicts precipitation CHANGE (ΔP)
rather than absolute values.

Architecture:
    Input:  [B, K, C_in, H, W]     K frames, C_in channels
    Output: [B, 1, H, W]            ΔP (precipitation change)

    P_hat_{t+1} = ReLU(P_t + ΔP)   Caller computes absolute prediction

Key improvements over competition version:
    1. TRUE temporal residual: ΔP = Model(X), P_hat = P_t + ΔP
       (vs. old self-residual: p_pred + delta_p internal refinement)
    2. Configurable input channels (4 to 16+)
    3. Optional Channel Attention (SE block) after ConvLSTM layers
    4. Clean separation: model predicts delta, caller adds to last frame
    5. No hard-coded magic numbers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

from .convlstm_cell import ConvLSTMCell
from .channel_attention import ChannelAttention


class PIResConvLSTM(nn.Module):
    """Physics-informed Residual ConvLSTM for typhoon precipitation prediction.

    Args:
        input_channels: Number of input channels per frame.
        precip_channel_idx: Index of precipitation channel (default 0).
        hidden_dims: Hidden dimensions for stacked ConvLSTM layers.
        kernel_size: ConvLSTM kernel size.
        use_attention: Apply Channel Attention between encoder layers.
        attention_reduction: SE block reduction ratio.
        dropout: Dropout rate applied to hidden states between layers.
        use_layer_norm: Apply LayerNorm inside ConvLSTM cells.
        use_batch_norm: Apply BatchNorm in decoder path.
    """

    def __init__(
        self,
        input_channels: int = 4,
        precip_channel_idx: int = 0,
        hidden_dims: List[int] = None,
        kernel_size: int = 3,
        use_attention: bool = False,
        attention_reduction: int = 16,
        dropout: float = 0.0,
        use_layer_norm: bool = False,
        use_batch_norm: bool = True,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 128]

        self.input_channels = input_channels
        self.precip_channel_idx = precip_channel_idx
        self.hidden_dims = hidden_dims
        self.use_attention = use_attention
        self.num_layers = len(hidden_dims)

        # ---- Encoder: Stacked ConvLSTM cells ----
        self.encoder_cells = nn.ModuleList()
        in_ch = input_channels
        for hd in hidden_dims:
            self.encoder_cells.append(
                ConvLSTMCell(
                    input_channels=in_ch,
                    hidden_channels=hd,
                    kernel_size=kernel_size,
                    layer_norm=use_layer_norm,
                )
            )
            in_ch = hd  # Output of layer i is input to layer i+1

        # ---- Optional Channel Attention after each layer ----
        if use_attention:
            self.attention_layers = nn.ModuleList([
                ChannelAttention(hd, reduction=attention_reduction)
                for hd in hidden_dims
            ])
        else:
            self.attention_layers = None

        # ---- Dropout between layers ----
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        # ---- Decoder: Reduce deepest hidden to intermediate features ----
        decoder_layers = [
            nn.Conv2d(hidden_dims[-1], hidden_dims[0], kernel_size=3, padding=1),
        ]
        if use_batch_norm:
            decoder_layers.append(nn.BatchNorm2d(hidden_dims[0]))
        decoder_layers.append(nn.ReLU(inplace=True))
        self.decoder = nn.Sequential(*decoder_layers)

        # ---- Base prediction head ----
        self.pred_head = nn.Conv2d(hidden_dims[0], 1, kernel_size=1)

        # ---- Residual refinement network ----
        # Combines deepest hidden features with base prediction
        # to produce the final ΔP
        self.refine_net = nn.Sequential(
            nn.Conv2d(hidden_dims[-1] + 1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        """Kaiming initialization for Conv2d layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — predicts precipitation CHANGE (ΔP).

        Args:
            x: [B, K, C, H, W] input tensor.
               K = number of input frames.
               C = input_channels.
               H, W = spatial dimensions.

        Returns:
            delta_p: [B, 1, H, W] predicted precipitation CHANGE.
                     Call compute_prediction() to get absolute precipitation.
        """
        B, K, _, H, W = x.shape

        # Initialize hidden states for all encoder layers
        states = [
            cell.init_hidden(B, (H, W))
            for cell in self.encoder_cells
        ]

        # Process temporal sequence through stacked ConvLSTM
        for t in range(K):
            x_t = x[:, t, :, :, :]  # [B, C, H, W]

            for i, cell in enumerate(self.encoder_cells):
                h_new, c_new = cell(x_t, states[i])
                states[i] = (h_new, c_new)

                # Apply channel attention to hidden state
                if self.attention_layers is not None:
                    h_attn = self.attention_layers[i](h_new)
                    states[i] = (h_attn, c_new)

                # Dropout between layers (not after last layer)
                if i < self.num_layers - 1:
                    h_drop = self.dropout(states[i][0])
                    states[i] = (h_drop, states[i][1])

                # Current layer's output becomes next layer's input
                x_t = states[i][0]

        # Deepest hidden state (output of last ConvLSTM layer)
        h_deep = states[-1][0]  # [B, H_dims[-1], H, W]

        # Decode to intermediate features
        dec_feat = self.decoder(h_deep)     # [B, H_dims[0], H, W]

        # Base prediction
        p_base = self.pred_head(dec_feat)   # [B, 1, H, W]

        # Residual refinement: combine deepest features with base prediction
        refine_in = torch.cat([h_deep, p_base], dim=1)  # [B, H_dims[-1]+1, H, W]
        delta_p = self.refine_net(refine_in)              # [B, 1, H, W]

        return delta_p

    def compute_prediction(self, x: torch.Tensor) -> torch.Tensor:
        """Compute absolute precipitation prediction.

        P_hat_{t+1} = ReLU(P_t + ΔP)

        Args:
            x: [B, K, C, H, W] input tensor.

        Returns:
            P_hat: [B, 1, H, W] absolute precipitation prediction (>= 0).
        """
        # Last frame's precipitation channel
        P_last = x[:, -1, self.precip_channel_idx:
                   self.precip_channel_idx + 1, :, :]  # [B, 1, H, W]

        delta_p = self.forward(x)

        return F.relu(P_last + delta_p)
