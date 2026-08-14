"""Data normalization and augmentation transforms for typhoon data."""

import torch
from typing import Dict, Tuple


class MinMaxNormalize:
    """Min-max normalization: x -> (x - vmin) / (vmax - vmin).

    Args:
        vmin: Minimum value for normalization.
        vmax: Maximum value for normalization.
        channel_idx: Channel index to normalize. If None, normalizes all.
    """

    def __init__(self, vmin: float = 0.0, vmax: float = 100.0,
                 channel_idx: int = 0):
        self.vmin = vmin
        self.vmax = vmax
        self.range = max(vmax - vmin, 1e-8)
        self.channel_idx = channel_idx

    def __call__(self, X: torch.Tensor, Y: torch.Tensor,
                 meta: Dict) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        if self.channel_idx is not None:
            X[:, self.channel_idx:self.channel_idx + 1] = (
                (X[:, self.channel_idx:self.channel_idx + 1] - self.vmin)
                / self.range
            )
        else:
            X = (X - self.vmin) / self.range

        Y = (Y - self.vmin) / self.range
        meta['P_prev'] = (meta['P_prev'] - self.vmin) / self.range
        return X, Y, meta

    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reverse normalization."""
        return tensor * self.range + self.vmin


class LogTransform:
    """Logarithmic transform: x -> log(1 + x).

    Compresses the dynamic range, helpful for precipitation data
    where values span several orders of magnitude.
    """

    def __call__(self, X: torch.Tensor, Y: torch.Tensor,
                 meta: Dict) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        X = torch.log1p(X)
        Y = torch.log1p(Y)
        meta['P_prev'] = torch.log1p(meta['P_prev'])
        return X, Y, meta

    @staticmethod
    def inverse(tensor: torch.Tensor) -> torch.Tensor:
        """Reverse log transform: x -> exp(x) - 1."""
        return torch.expm1(tensor)


class RandomRotation:
    """Random 90-degree rotation augmentation (preserves grid alignment).

    .. warning::
        UNSAFE for the vector-valued channels of the paper schema. Rotating the
        spatial grid does NOT correctly rotate u_move/v_move (translation vector),
        dh_dx/dh_dy (terrain gradient vector), or dx/dy (coordinate axes). A
        fully vector-aware rotation is NOT implemented, so this transform is
        DISABLED for paper experiments. Use only scalar-field transforms.

    Args:
        p: Probability of applying rotation.
    """

    def __init__(self, p: float = 0.5):
        import warnings

        warnings.warn(
            "RandomRotation is unsafe for vector-valued channels "
            "(u/v, dh_dx/dh_dy, dx/dy) and is disabled for paper experiments. "
            "A vector-aware rotation is not implemented.",
            RuntimeWarning,
        )
        self.p = p

    def __call__(self, X: torch.Tensor, Y: torch.Tensor,
                 meta: Dict) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        if torch.rand(1).item() > self.p:
            return X, Y, meta

        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            X = torch.rot90(X, k, dims=[-2, -1])
            Y = torch.rot90(Y, k, dims=[-2, -1])
            meta['P_prev'] = torch.rot90(meta['P_prev'], k, dims=[-2, -1])
        return X, Y, meta


class Compose:
    """Compose multiple transforms sequentially.

    Args:
        transforms: List of callable transforms.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, X: torch.Tensor, Y: torch.Tensor,
                 meta: Dict) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        for t in self.transforms:
            X, Y, meta = t(X, Y, meta)
        return X, Y, meta


class ChannelNormalize:
    """Per-channel normalization keyed by CANONICAL channel index.

    Applies the frozen train-only statistics from ``configs/normalization_v1.json``:
      - precipitation (0): min-max to [0, 1] using vmax.
      - track features (1,2,6,7): z-score.
      - terrain (8,9,10): z-score.
      - normalized geometry (3,4,5) and land_mask (11): unchanged.

    Args:
        stats: dict loaded from configs/normalization_v1.json.
        channel_indices: list of canonical indices present in X (in order).
        precip_vmax: precipitation min-max upper bound (mm/h).
    """

    def __init__(self, stats: Dict, channel_indices=None, precip_vmax: float = 100.0):
        self.channel_indices = channel_indices if channel_indices is not None else list(range(12))
        self.precip_vmax = precip_vmax
        self.track_stats = stats.get("track_features", {})
        self.terrain_stats = stats.get("terrain", {})

        # Map canonical index -> (name, kind). kind in {precip, zscore_track, zscore_terrain, none}.
        self._rules = {}
        self._rules[0] = ("precip",)
        self._rules[1] = ("zscore_track", "center_wind_speed")
        self._rules[2] = ("zscore_track", "center_pressure")
        self._rules[6] = ("zscore_track", "u_move")
        self._rules[7] = ("zscore_track", "v_move")
        self._rules[8] = ("zscore_terrain", "dem")
        self._rules[9] = ("zscore_terrain", "dh_dx")
        self._rules[10] = ("zscore_terrain", "dh_dy")
        # 3, 4, 5 (r_norm, dx_norm, dy_norm) and 11 (land_mask): unchanged.

    def __call__(self, X: torch.Tensor, Y: torch.Tensor,
                 meta: Dict) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        for pos, canonical_idx in enumerate(self.channel_indices):
            if canonical_idx not in self._rules:
                continue
            rule = self._rules[canonical_idx]
            if rule[0] == "precip":
                X[:, pos] = X[:, pos] / self.precip_vmax
            elif rule[0] == "zscore_track":
                s = self.track_stats.get(rule[1], {})
                mean, std = s.get("mean", 0.0), s.get("std", 1.0)
                X[:, pos] = (X[:, pos] - mean) / (std if std > 1e-8 else 1.0)
            elif rule[0] == "zscore_terrain":
                s = self.terrain_stats.get(rule[1], {})
                mean, std = s.get("mean", 0.0), s.get("std", 1.0)
                X[:, pos] = (X[:, pos] - mean) / (std if std > 1e-8 else 1.0)

        # Target precipitation is min-max normalized too.
        Y = Y / self.precip_vmax
        meta["P_prev"] = meta["P_prev"] / self.precip_vmax
        return X, Y, meta
