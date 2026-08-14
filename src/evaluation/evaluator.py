"""Unified evaluator with overall + per-event (typhoon-level) aggregation.

All categorical metrics are computed at physically meaningful rainfall-rate
thresholds (mm/h). Metrics are aggregated BOTH overall and per typhoon event so
that paired event comparisons (e.g. E4 - E3) are directly possible, avoiding
pseudoreplication from temporally-correlated sliding windows.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .metrics import compute_all_metrics

# Primary rainfall-rate thresholds (mm/h) for categorical metrics.
DEFAULT_THRESHOLDS = [5.0, 10.0, 20.0, 30.0]


def _per_sample_metrics(P_pred_mmh: np.ndarray, P_true_mmh: np.ndarray,
                        thresholds: List[float]) -> Dict[str, float]:
    """Compute metrics for a single [H, W] frame in mm/h."""
    return compute_all_metrics(P_pred_mmh, P_true_mmh, thresholds=thresholds)


@torch.no_grad()
def evaluate_model(
    model,
    loader: DataLoader,
    device: torch.device,
    precip_vmax: float = 100.0,
    thresholds: Optional[List[float]] = None,
    channel_indices: Optional[List[int]] = None,
) -> Dict:
    """Evaluate a model on a loader.

    The model is expected to produce normalized precipitation; results are
    denormalized by `precip_vmax` before thresholding (thresholds are mm/h).

    Returns a dict:
        overall: {metric: value}  (averaged over all windows)
        per_event: {typhoon_id: {metric: value}}
        per_window: list of {typhoon_id, metric: value}
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    model.eval()
    per_window = []

    for X, Y, meta in loader:
        X = X.to(device)
        Y = Y.to(device)

        if channel_indices is not None:
            X = X[:, :, channel_indices, :, :]

        if hasattr(model, "compute_prediction"):
            P_hat = model.compute_prediction(X)
        elif isinstance(model, torch.nn.Module):
            # PlainConvLSTM / TrajGRU take precipitation-only input.
            if hasattr(model, "forward"):
                try:
                    P_hat = model(X)
                except TypeError:
                    P_hat = model(X[:, :, 0:1, :, :])
            else:
                P_hat = model(X)
        else:
            # Persistence-style callable.
            P_hat = model(X)

        P_hat = P_hat.detach().cpu().numpy()[:, 0]
        Y_np = Y.detach().cpu().numpy()[:, 0]

        # Denormalize (dataset uses [0, precip_vmax] min-max on precipitation).
        P_hat = P_hat * precip_vmax
        Y_np = Y_np * precip_vmax

        tids = meta.get("typhoon_id")
        for i in range(P_hat.shape[0]):
            m = _per_sample_metrics(P_hat[i], Y_np[i], thresholds)
            tid = int(tids[i]) if tids is not None else -1
            per_window.append({"typhoon_id": tid, **m})

    overall = _aggregate(per_window)
    per_event = _aggregate_by_event(per_window)
    return {"overall": overall, "per_event": per_event, "per_window": per_window}


def _aggregate(rows: List[Dict]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = [k for k in rows[0] if k != "typhoon_id"]
    out = {}
    for k in keys:
        vals = [r[k] for r in rows if np.isfinite(r[k])]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out


def _aggregate_by_event(rows: List[Dict]) -> Dict[int, Dict[str, float]]:
    events: Dict[int, List[Dict]] = {}
    for r in rows:
        events.setdefault(r["typhoon_id"], []).append(r)
    return {tid: _aggregate(ev_rows) for tid, ev_rows in events.items()}
