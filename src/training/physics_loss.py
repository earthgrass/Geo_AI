"""Physics-informed loss functions for typhoon precipitation prediction.

Implements the composite loss from the PI-ResConvLSTM blueprint:

    L_total = L_rain + λ1·L_nonneg + λ2·L_oro + λ3·L_smooth + λ4·L_extreme

Each loss component encodes a physical prior:
    L_rain:    Weighted MSE (higher weight on heavy rain pixels)
    L_nonneg:  Penalize negative precipitation
    L_oro:     Orographic uplift consistency (terrain constraint)
    L_smooth:  Weak spatial + temporal smoothness
    L_extreme: Extra focus on extreme precipitation regions

Each component can be individually disabled for ablation studies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List


class PhysicsInformedLoss(nn.Module):
    """Composite physics-informed loss for precipitation prediction.

    Args:
        lambda_nonneg: Weight for non-negativity penalty.
        lambda_oro: Weight for orographic uplift constraint.
        lambda_smooth: Weight for spatiotemporal smoothness.
        lambda_extreme: Weight for extreme precipitation focus.
        extreme_threshold: Precipitation threshold (mm/h) for extreme focus.
        heavy_rain_alpha: Extra weight multiplier for heavy rain pixels.
        orographic_corr_weight: Use correlation-based oro loss (True)
                                vs threshold-based (False).
        components: List of loss component names to enable.
                    Subset for ablation: e.g. ["rain", "nonneg"].
    """

    def __init__(
        self,
        lambda_nonneg: float = 0.1,
        lambda_oro: float = 0.1,
        lambda_smooth: float = 0.01,
        lambda_extreme: float = 0.5,
        extreme_threshold: float = 10.0,
        heavy_rain_alpha: float = 2.0,
        orographic_corr_weight: bool = True,
        components: Optional[List[str]] = None,
    ):
        super().__init__()
        self.lambda_nonneg = lambda_nonneg
        self.lambda_oro = lambda_oro
        self.lambda_smooth = lambda_smooth
        self.lambda_extreme = lambda_extreme
        self.extreme_threshold = extreme_threshold
        self.heavy_rain_alpha = heavy_rain_alpha
        self.orographic_corr_weight = orographic_corr_weight

        if components is None:
            self.components = ["rain", "nonneg", "oro", "smooth", "extreme"]
        else:
            self.components = components

    def forward(
        self,
        P_hat: torch.Tensor,
        P_true: torch.Tensor,
        aux: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute physics-informed loss.

        Args:
            P_hat: [B, 1, H, W] predicted precipitation.
            P_true: [B, 1, H, W] ground truth precipitation.
            aux: Auxiliary tensors. Expected keys:
                - 'oro_lift': [B, 1, H, W] orographic uplift O=u·∇h
                - 'P_prev':  [B, 1, H, W] previous timestep precipitation

        Returns:
            Dict mapping component name to loss value (all scalars).
            Always contains 'total' key.
        """
        if aux is None:
            aux = {}

        losses = {}

        # ---- 1. Weighted MSE (L_rain) ----
        if "rain" in self.components:
            losses['rain'] = self._weighted_mse(P_hat, P_true)

        # ---- 2. Non-negativity (L_nonneg) ----
        if "nonneg" in self.components:
            losses['nonneg'] = F.relu(-P_hat).pow(2).mean()

        # ---- 3. Orographic uplift (L_oro) ----
        if "oro" in self.components and 'oro_lift' in aux:
            losses['oro'] = self._oro_loss(P_hat, aux['oro_lift'])

        # ---- 4. Spatiotemporal smoothness (L_smooth) ----
        if "smooth" in self.components and 'P_prev' in aux:
            losses['smooth'] = self._smoothness_loss(P_hat, aux['P_prev'])

        # ---- 5. Extreme precipitation (L_extreme) ----
        if "extreme" in self.components:
            losses['extreme'] = self._extreme_loss(P_hat, P_true)

        # ---- Combine ----
        total = torch.tensor(0.0, device=P_hat.device)
        total = total + losses.get('rain', 0.0)

        if 'nonneg' in losses:
            total = total + self.lambda_nonneg * losses['nonneg']
        if 'oro' in losses:
            total = total + self.lambda_oro * losses['oro']
        if 'smooth' in losses:
            total = total + self.lambda_smooth * losses['smooth']
        if 'extreme' in losses:
            total = total + self.lambda_extreme * losses['extreme']

        losses['total'] = total
        return losses

    # ---- Private component implementations ----

    def _weighted_mse(self, P_hat: torch.Tensor,
                      P_true: torch.Tensor) -> torch.Tensor:
        """Weighted MSE: higher weight on heavy rain regions.

        w(P_true) = 1 + alpha * I(P_true > threshold)
        """
        weight = 1.0 + self.heavy_rain_alpha * (
            P_true > self.extreme_threshold
        ).float()
        sq_error = (P_hat - P_true).pow(2)
        return (weight * sq_error).mean()

    def _oro_loss(self, P_hat: torch.Tensor,
                  oro_lift: torch.Tensor) -> torch.Tensor:
        """Orographic uplift consistency constraint.

        O = u·dh/dx + v·dh/dy  (wind-aligned terrain gradient)

        Two variants:
        1. Correlation-based (default):
           L_oro = -corr(P_hat, ReLU(O))
           Encourages positive correlation between precipitation and uplift.

        2. Threshold-based:
           Penalizes low precipitation where uplift is high.
        """
        if self.orographic_corr_weight:
            # Flatten spatial dims
            B = P_hat.shape[0]
            P_flat = P_hat.reshape(B, -1)         # [B, H*W]
            O_flat = F.relu(oro_lift).reshape(B, -1)  # [B, H*W]

            P_centered = P_flat - P_flat.mean(dim=1, keepdim=True)
            O_centered = O_flat - O_flat.mean(dim=1, keepdim=True)

            cov = (P_centered * O_centered).sum(dim=1)
            var_p = (P_centered * P_centered).sum(dim=1)
            var_o = (O_centered * O_centered).sum(dim=1)

            corr = cov / (torch.sqrt(var_p * var_o) + 1e-8)
            return -corr.mean()  # Minimize negative correlation
        else:
            # Threshold: penalize no-rain where uplift is high
            pos_uplift = F.relu(oro_lift)
            flat = pos_uplift.flatten()
            threshold_val = torch.quantile(
                flat[flat > 0], 0.8
            ) if (flat > 0).any() else 1.0

            mask = (pos_uplift > threshold_val).float()
            return (mask * F.relu(0.1 - P_hat).pow(2)).mean()

    def _smoothness_loss(self, P_hat: torch.Tensor,
                         P_prev: torch.Tensor) -> torch.Tensor:
        """Weak spatial + temporal smoothness constraint.

        Small weight is critical — too large and peak precipitation
        gets smoothed out.
        """
        # Spatial TV (L1 gradient)
        dy = (P_hat[:, :, 1:, :] - P_hat[:, :, :-1, :]).abs().mean()
        dx = (P_hat[:, :, :, 1:] - P_hat[:, :, :, :-1]).abs().mean()
        L_spatial = dy + dx

        # Temporal change from previous frame
        L_temporal = (P_hat - P_prev).abs().mean()

        return L_spatial + L_temporal

    def _extreme_loss(self, P_hat: torch.Tensor,
                      P_true: torch.Tensor) -> torch.Tensor:
        """Extra focus on extreme precipitation regions.

        Computes MSE only over pixels where P_true > extreme_threshold.
        """
        mask = (P_true > self.extreme_threshold).float()
        if mask.sum() < 1:
            return torch.tensor(0.0, device=P_hat.device)

        sq_error = (P_hat - P_true).pow(2)
        return (mask * sq_error).sum() / (mask.sum() + 1e-8)
