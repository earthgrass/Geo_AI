"""Evaluation Protocol V2 metrics (frozen: docs/EVALUATION_PROTOCOL_V2.md).

Design rules enforced here:

1. **Continuous metrics** are computed from sufficient statistics
   (``sum_abs``, ``sum_sq``, pixel count ``n``). Global / event aggregation
   pools these statistics FIRST and then computes MAE / RMSE once. Averaging
   per-window ratios is never a primary score.

2. **Categorical metrics** pool integer contingency counts (``a,b,c,d``) at the
   relevant aggregation level and then compute CSI / POD / FAR / HSS / ACC /
   BIAS once from the pooled counts. ``d`` (correct negatives) is always
   returned so HSS and ACC are reconstructable.

3. **Zero denominators return NaN** — never ``max(denominator, eps)`` forcing a
   finite value.

4. **SSIM** uses the frozen fixed ``data_range = 100 mm/h`` with a 7x7 uniform
   window (``skimage.metrics.structural_similarity``). It is computed per window
   and aggregated as an arithmetic mean; there is no pooled-image SSIM.

5. Legacy per-window-range NRMSE and peak-relative error are NOT emitted by v2.
   Only absolute peak error (mm/h) is retained as a diagnostic.
"""

import numpy as np
from typing import Dict, List, Optional

from skimage.metrics import structural_similarity as ssim

# ---------------------------------------------------------------------------
# Frozen constants
# ---------------------------------------------------------------------------

# Primary categorical thresholds (mm/h), frozen before any test access.
DEFAULT_THRESHOLDS = [5.0, 10.0, 20.0, 30.0]

# Frozen SSIM dynamic range (mm/h) — MUST NOT depend on per-window maxima.
SSIM_DATA_RANGE = 100.0

# Predictions below this (in normalized units) are treated as rounding noise.
NEGATIVE_TOLERANCE = -1e-7


# ---------------------------------------------------------------------------
# Key formatting
# ---------------------------------------------------------------------------

def threshold_key(tau: float) -> str:
    """Canonical threshold suffix, e.g. ``5.0 -> '5mmh'``, ``10.0 -> '10mmh'``.

    Single source of truth for threshold-key formatting so the report writer
    and the metric emitter can never disagree (e.g. ``CSI_10.0mmh`` vs
    ``CSI_10mmh``).
    """
    if float(tau).is_integer():
        return f"{int(tau)}mmh"
    return f"{tau}mmh"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def validate_finite(P_hat: np.ndarray, P_true: np.ndarray) -> None:
    """Raise on non-finite prediction/target arrays (evaluator error)."""
    if not np.isfinite(P_hat).all():
        raise ValueError("Non-finite prediction array encountered in evaluation.")
    if not np.isfinite(P_true).all():
        raise ValueError("Non-finite target array encountered in evaluation.")


def clamp_negative_tiny(P_hat: np.ndarray) -> int:
    """Clamp values in ``[NEGATIVE_TOLERANCE, 0)`` to zero (roundoff only).

    Returns the number of clamped pixels so the caller can log it. Values below
    ``NEGATIVE_TOLERANCE`` are NOT touched here; the evaluator fails on them
    because the ReLU architecture should never produce them.
    """
    if not np.isfinite(P_hat).all():
        raise ValueError("Non-finite prediction array encountered in evaluation.")
    mask = (P_hat < 0) & (P_hat >= NEGATIVE_TOLERANCE)
    n_clamped = int(mask.sum())
    P_hat[mask] = 0.0
    return n_clamped


# ---------------------------------------------------------------------------
# Continuous metrics (sufficient statistics)
# ---------------------------------------------------------------------------

def compute_continuous_suff_stats(
    P_hat: np.ndarray,
    P_true: np.ndarray,
) -> Dict[str, float]:
    """Return continuous sufficient statistics for one window (mm/h).

    Args:
        P_hat: [H, W] predicted precipitation in mm/h.
        P_true: [H, W] observed precipitation in mm/h.

    Returns:
        ``{"sum_abs": float, "sum_sq": float, "n": int}``.
    """
    validate_finite(P_hat, P_true)
    diff = P_hat.astype(np.float64) - P_true.astype(np.float64)
    return {
        "sum_abs": float(np.abs(diff).sum()),
        "sum_sq": float((diff * diff).sum()),
        "n": int(P_hat.size),
    }


def continuous_from_suff(stats: Dict[str, float]) -> Dict[str, float]:
    """Compute MAE / RMSE from pooled sufficient statistics."""
    n = stats["n"]
    if n <= 0:
        return {"MAE": float("nan"), "RMSE": float("nan")}
    mae = stats["sum_abs"] / n
    rmse = float(np.sqrt(stats["sum_sq"] / n))
    return {"MAE": float(mae), "RMSE": rmse}


def nrmse_fixed100(rmse_global: float, scale: float = SSIM_DATA_RANGE) -> float:
    """Optional secondary diagnostic: ``RMSE_global / 100 mm/h``.

    Protocol v2 §14: this is the ONLY permitted NRMSE form. It uses the frozen
    normalization scale (100 mm/h) — never the per-window observed range — and
    is never part of the primary v2 table.
    """
    return float(rmse_global) / float(scale)


# ---------------------------------------------------------------------------
# SSIM (fixed data range)
# ---------------------------------------------------------------------------

def compute_window_ssim(
    P_hat: np.ndarray,
    P_true: np.ndarray,
    data_range: float = SSIM_DATA_RANGE,
) -> float:
    """Per-window SSIM with the frozen fixed range ``100 mm/h``.

    Args:
        P_hat: [H, W] predicted precipitation in mm/h.
        P_true: [H, W] observed precipitation in mm/h.
        data_range: fixed dynamic range (mm/h). MUST be 100.0 for v2.

    Returns:
        Float SSIM value.
    """
    validate_finite(P_hat, P_true)
    if data_range != SSIM_DATA_RANGE:
        raise ValueError(
            f"SSIM data_range must be fixed at {SSIM_DATA_RANGE} mm/h for v2, "
            f"got {data_range}."
        )
    return float(ssim(
        P_true, P_hat,
        data_range=data_range,
        win_size=7,
        gaussian_weights=False,
        use_sample_covariance=True,
        channel_axis=None,
        full=False,
        K1=0.01,
        K2=0.03,
    ))


# ---------------------------------------------------------------------------
# Categorical metrics (contingency counts first)
# ---------------------------------------------------------------------------

def compute_contingency_counts(
    P_hat: np.ndarray,
    P_true: np.ndarray,
    threshold: float,
) -> Dict[str, int]:
    """Return integer contingency counts at one threshold (mm/h).

    A threshold event is present when the value is >= `threshold`.

    Returns:
        ``{"a": hits, "b": false_alarms, "c": misses, "d": correct_negatives}``
        all as Python ints.
    """
    validate_finite(P_hat, P_true)
    pred_yes = P_hat >= threshold
    obs_yes = P_true >= threshold
    return {
        "a": int((pred_yes & obs_yes).sum()),
        "b": int((pred_yes & ~obs_yes).sum()),
        "c": int((~pred_yes & obs_yes).sum()),
        "d": int((~pred_yes & ~obs_yes).sum()),
    }


def categorical_from_counts(
    a: int, b: int, c: int, d: int,
) -> Dict[str, float]:
    """Compute categorical scores from POOLED contingency counts.

    Undefined ratios (zero denominator) return NaN. Counts are always included
    so pooled HSS / ACC are auditable and reconstructable.
    """
    n = a + b + c + d
    denom_csi = a + b + c
    denom_pos = a + c              # observed positives
    denom_far = a + b              # forecast positives
    denom_hss = (a + c) * (c + d) + (a + b) * (b + d)

    return {
        "CSI": _safe_ratio(a, denom_csi),
        "POD": _safe_ratio(a, denom_pos),
        "FAR": _safe_ratio(b, denom_far),
        "HSS": _safe_ratio(2.0 * (a * d - b * c), denom_hss),
        "ACC": _safe_ratio(a + d, n),
        "BIAS": _safe_ratio(a + b, denom_pos),
        "a_hits": int(a),
        "b_false_alarms": int(b),
        "c_misses": int(c),
        "d_correct_negatives": int(d),
        "n_total": int(n),
    }


def _safe_ratio(num: float, den: float) -> float:
    """Return num/den, or NaN when den == 0 (never a forced epsilon)."""
    if den == 0:
        return float("nan")
    return float(num / den)


# ---------------------------------------------------------------------------
# Per-window convenience
# ---------------------------------------------------------------------------

def compute_window_diagnostics(
    P_hat: np.ndarray,
    P_true: np.ndarray,
    thresholds: Optional[List[float]] = None,
) -> Dict[str, object]:
    """Compute the v2 per-window diagnostic record for one field (mm/h).

    Returns a dict with continuous values, SSIM, absolute peak error, and the
    raw contingency counts per threshold. This is the raw material the
    evaluator accumulates; it does NOT aggregate anything.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS
    cont = compute_continuous_suff_stats(P_hat, P_true)
    out: Dict[str, object] = dict(cont)
    out["SSIM"] = compute_window_ssim(P_hat, P_true)
    out["peak_error"] = float(
        np.abs(float(P_hat.max()) - float(P_true.max()))
    )
    # Field maxima in mm/h — needed for event/split-level absolute peak error
    # ``|max_{w,p} f - max_{w,p} y|`` (protocol v2 §15).
    out["f_max"] = float(P_hat.max())
    out["y_max"] = float(P_true.max())
    counts = {
        threshold_key(tau): compute_contingency_counts(P_hat, P_true, tau)
        for tau in thresholds
    }
    out["counts"] = counts
    return out
