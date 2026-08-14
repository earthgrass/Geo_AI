"""Physics-informed loss functions for typhoon precipitation prediction.

Composite loss (revised for scientific cleanliness):

    L_total = L_rain + λ_smooth·L_smooth + λ_extreme·L_extreme  (+ λ_oro·L_oro, opt-in)

where:
    L_rain:    standard (unweighted) MSE over the full precipitation field
    L_smooth:  weak spatial + temporal smoothness
    L_extreme: MSE restricted to extreme-precipitation pixels (P_true > threshold)
    L_oro:     orographic-uplift consistency (OPT-IN; requires explicitly
               configured environmental wind channels)

Removed on purpose (2026-08-14):
    * L_nonneg — the model output is already P_hat = ReLU(P_t + ΔP) >= 0, so
      ReLU(-P_hat)^2 is identically zero and carries no signal.
    * Heavy-rain weighting inside L_rain — extreme weighting now lives ONLY in
      L_extreme, so that "base MSE" and "extreme focus" ablate cleanly.

Orographic constraint semantics:
    u_move / v_move (storm translation velocities) are NOT environmental wind
    components and MUST NOT be used as the wind terms of an orographic uplift
    constraint. The orographic term is therefore OFF by default and can only be
    enabled by configuring real atmospheric wind channels (see `oro_config`).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List


class PhysicsInformedLoss(nn.Module):
    """Composite physics-informed loss for precipitation prediction.

    Args:
        lambda_smooth: Weight for spatiotemporal smoothness.
        lambda_extreme: Weight for extreme precipitation focus.
        lambda_oro: Weight for orographic uplift constraint (only used when the
            orographic term is explicitly enabled).
        extreme_threshold: Precipitation threshold (mm/h) defining extreme pixels.
        orographic_corr_weight: Use correlation-based oro loss (True) vs
            threshold-based (False).
        components: List of loss component names to enable. Defaults to
            ["rain", "smooth", "extreme"] (orographic is opt-in).
        oro_config: Dict configuring the orographic term, e.g.
            {
                "enabled": False,
                "u_channel": None,       # real environmental zonal wind channel
                "v_channel": None,       # real environmental meridional wind channel
                "dh_dx_channel": 9,
                "dh_dy_channel": 10,
            }
            When enabled=True, u_channel and v_channel must be set to valid
            channel indices, otherwise a RuntimeError is raised (fail-safe).
    """

    def __init__(
        self,
        lambda_smooth: float = 0.01,
        lambda_extreme: float = 0.5,
        lambda_oro: float = 0.1,
        extreme_threshold: float = 10.0,
        orographic_corr_weight: bool = True,
        components: Optional[List[str]] = None,
        oro_config: Optional[Dict] = None,
    ):
        super().__init__()
        self.lambda_smooth = lambda_smooth
        self.lambda_extreme = lambda_extreme
        self.lambda_oro = lambda_oro
        self.extreme_threshold = extreme_threshold
        self.orographic_corr_weight = orographic_corr_weight

        self.oro_config = dict(oro_config or {})
        oro_enabled = bool(self.oro_config.get("enabled", False))

        if oro_enabled:
            self._validate_oro_config(self.oro_config)

        if components is None:
            components = ["rain", "smooth", "extreme"]
            if oro_enabled:
                components.append("oro")
        self.components = list(components)

    @staticmethod
    def _validate_oro_config(oro_config: Dict) -> None:
        """Fail fast if the orographic term is enabled without valid wind channels."""
        u_channel = oro_config.get("u_channel", None)
        v_channel = oro_config.get("v_channel", None)
        if u_channel is None or v_channel is None:
            raise RuntimeError(
                "Orographic uplift loss is enabled (orographic.enabled=True) but "
                "u_channel/v_channel are not configured. An orographic constraint "
                "requires REAL environmental atmospheric wind channels; storm "
                "translation velocities (u_move/v_move) are NOT valid wind terms. "
                "Set explicit u_channel and v_channel indices, or disable the "
                "orographic term."
            )
        for name, val in (("u_channel", u_channel), ("v_channel", v_channel)):
            if not isinstance(val, int) or val < 0:
                raise RuntimeError(
                    f"orographic {name} must be a non-negative integer channel "
                    f"index, got {val!r}."
                )

    def forward(
        self,
        P_hat: torch.Tensor,
        P_true: torch.Tensor,
        aux: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute the physics-informed loss.

        Args:
            P_hat: [B, 1, H, W] predicted precipitation.
            P_true: [B, 1, H, W] ground truth precipitation.
            aux: Auxiliary tensors. Expected keys:
                - 'oro_lift': [B, 1, H, W] orographic uplift O = u·∇h (only when
                  the orographic term is enabled)
                - 'P_prev':  [B, 1, H, W] previous-timestep precipitation

        Returns:
            Dict mapping component name -> scalar loss; always contains 'total'.
        """
        if aux is None:
            aux = {}

        losses = {}

        # 1. Standard (unweighted) MSE.
        if "rain" in self.components:
            losses["rain"] = (P_hat - P_true).pow(2).mean()

        # 2. Orographic uplift consistency (opt-in).
        if "oro" in self.components and "oro_lift" in aux:
            losses["oro"] = self._oro_loss(P_hat, aux["oro_lift"])

        # 3. Spatiotemporal smoothness.
        if "smooth" in self.components and "P_prev" in aux:
            losses["smooth"] = self._smoothness_loss(P_hat, aux["P_prev"])

        # 4. Extreme precipitation focus.
        if "extreme" in self.components:
            losses["extreme"] = self._extreme_loss(P_hat, P_true)

        # Combine.
        total = torch.tensor(0.0, device=P_hat.device)
        if "rain" in losses:
            total = total + losses["rain"]
        if "oro" in losses:
            total = total + self.lambda_oro * losses["oro"]
        if "smooth" in losses:
            total = total + self.lambda_smooth * losses["smooth"]
        if "extreme" in losses:
            total = total + self.lambda_extreme * losses["extreme"]

        losses["total"] = total
        return losses

    # ---- Component implementations ----

    def _oro_loss(self, P_hat: torch.Tensor, oro_lift: torch.Tensor) -> torch.Tensor:
        """Orographic uplift consistency constraint.

        O = u·dh/dx + v·dh/dy  (environmental-wind-aligned terrain gradient).

        Only reachable when the orographic term is explicitly enabled with
        configured environmental wind channels.
        """
        if self.orographic_corr_weight:
            B = P_hat.shape[0]
            P_flat = P_hat.reshape(B, -1)
            O_flat = F.relu(oro_lift).reshape(B, -1)

            P_centered = P_flat - P_flat.mean(dim=1, keepdim=True)
            O_centered = O_flat - O_flat.mean(dim=1, keepdim=True)

            cov = (P_centered * O_centered).sum(dim=1)
            var_p = (P_centered * P_centered).sum(dim=1)
            var_o = (O_centered * O_centered).sum(dim=1)

            corr = cov / (torch.sqrt(var_p * var_o) + 1e-8)
            return -corr.mean()
        else:
            pos_uplift = F.relu(oro_lift)
            flat = pos_uplift.flatten()
            threshold_val = torch.quantile(
                flat[flat > 0], 0.8
            ) if (flat > 0).any() else 1.0

            mask = (pos_uplift > threshold_val).float()
            return (mask * F.relu(0.1 - P_hat).pow(2)).mean()

    def _smoothness_loss(self, P_hat: torch.Tensor, P_prev: torch.Tensor) -> torch.Tensor:
        """Weak spatial + temporal smoothness constraint."""
        dy = (P_hat[:, :, 1:, :] - P_hat[:, :, :-1, :]).abs().mean()
        dx = (P_hat[:, :, :, 1:] - P_hat[:, :, :, :-1]).abs().mean()
        L_spatial = dy + dx
        L_temporal = (P_hat - P_prev).abs().mean()
        return L_spatial + L_temporal

    def _extreme_loss(self, P_hat: torch.Tensor, P_true: torch.Tensor) -> torch.Tensor:
        """MSE restricted to extreme-precipitation pixels (P_true > threshold)."""
        mask = (P_true > self.extreme_threshold).float()
        if mask.sum() < 1:
            return torch.tensor(0.0, device=P_hat.device)

        sq_error = (P_hat - P_true).pow(2)
        return (mask * sq_error).sum() / (mask.sum() + 1e-8)
