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
