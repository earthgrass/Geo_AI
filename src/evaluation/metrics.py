"""Meteorological evaluation metrics for precipitation nowcasting.

Implements the full suite recommended by the PI-ResConvLSTM blueprint:
    Continuous:  MAE, RMSE, SSIM, Peak Error, Storm Area Error, Center Displacement
    Categorical: CSI, POD, FAR, HSS (at multiple precipitation thresholds)
    Per-category: Accuracy by CMA precipitation intensity category

Reference:
    Forecast verification methods: Wilks (2011) Statistical Methods in
    the Atmospheric Sciences, 3rd ed.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from skimage.metrics import structural_similarity as ssim


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CMA_CATEGORIES = {
    'light':       (0.1, 2.0),
    'moderate':    (2.0, 5.0),
    'heavy':       (5.0, 10.0),
    'torrential':  (10.0, 20.0),
    'extreme':     (20.0, float('inf')),
}

DEFAULT_THRESHOLDS = [1.0, 5.0, 10.0, 20.0, 50.0]  # mm/h


# ---------------------------------------------------------------------------
# Continuous metrics
# ---------------------------------------------------------------------------

def compute_continuous_metrics(
    P_hat: np.ndarray,
    P_true: np.ndarray,
    pixel_area_km2: float = 100.0,
    resolution_km: float = 10.0,
) -> Dict[str, float]:
    """Compute continuous evaluation metrics.

    Args:
        P_hat: [H, W] predicted precipitation (mm/h).
        P_true: [H, W] ground truth precipitation (mm/h).
        pixel_area_km2: Area per pixel in km^2.
        resolution_km: Grid resolution in km.

    Returns:
        Dict of metric name -> value.
    """
    metrics = {}

    # MAE and RMSE
    diff = P_hat - P_true
    metrics['MAE'] = float(np.mean(np.abs(diff)))
    metrics['RMSE'] = float(np.sqrt(np.mean(diff ** 2)))

    # Normalized RMSE
    p_range = max(P_true.max() - P_true.min(), 1e-6)
    metrics['NRMSE'] = metrics['RMSE'] / p_range

    # SSIM
    data_max = max(P_true.max(), P_hat.max(), 1e-6)
    win_size = min(7, min(P_true.shape) // 2 * 2 + 1)
    if win_size >= 3:
        metrics['SSIM'] = float(ssim(
            P_true / data_max, P_hat / data_max,
            data_range=1.0, win_size=win_size,
        ))
    else:
        metrics['SSIM'] = float('nan')

    # Peak precipitation error
    metrics['peak_error'] = float(np.abs(P_hat.max() - P_true.max()))
    metrics['peak_rel_error'] = float(
        metrics['peak_error'] / max(P_true.max(), 1e-6)
    )

    # Storm area error (area with precipitation > 1 mm/h)
    storm_true = (P_true > 1.0).sum() * pixel_area_km2
    storm_pred = (P_hat > 1.0).sum() * pixel_area_km2
    metrics['storm_area_true_km2'] = float(storm_true)
    metrics['storm_area_pred_km2'] = float(storm_pred)
    metrics['storm_area_error_km2'] = float(np.abs(storm_pred - storm_true))
    metrics['storm_area_rel_error'] = float(
        np.abs(storm_pred - storm_true) / max(storm_true, 1.0)
    )

    # Center displacement
    metrics['center_displacement_km'] = float(
        _center_displacement(P_hat, P_true, resolution_km)
    )

    return metrics


# ---------------------------------------------------------------------------
# Categorical (dichotomous) metrics
# ---------------------------------------------------------------------------

def compute_categorical_metrics(
    P_hat: np.ndarray,
    P_true: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """Compute categorical metrics at a given precipitation threshold.

    Contingency table (2x2):
                   Predicted YES   Predicted NO
        Observed YES       a              c
        Observed NO        b              d

    where:
        a = hits (correct yes)
        b = false alarms
        c = misses
        d = correct negatives

    Metrics:
        CSI = a / (a + b + c)           Threat Score / Critical Success Index
        POD = a / (a + c)               Probability of Detection (hit rate)
        FAR = b / (a + b)               False Alarm Ratio
        HSS = 2(ad - bc) / [(a+c)(c+d) + (a+b)(b+d)]  Heidke Skill Score
        ACC = (a + d) / n               Accuracy
        BIAS = (a + b) / (a + c)        Frequency Bias

    Args:
        P_hat: [H, W] predicted precipitation.
        P_true: [H, W] ground truth.
        threshold: Precipitation threshold for yes/no.

    Returns:
        Dict of metric name -> value.
    """
    pred_yes = P_hat >= threshold
    obs_yes = P_true >= threshold

    a = float((pred_yes & obs_yes).sum())       # hits
    b = float((pred_yes & ~obs_yes).sum())      # false alarms
    c = float((~pred_yes & obs_yes).sum())      # misses
    d = float((~pred_yes & ~obs_yes).sum())     # correct negatives
    n = a + b + c + d

    return {
        'CSI': a / max(a + b + c, 1.0),
        'POD': a / max(a + c, 1.0),
        'FAR': b / max(a + b, 1.0),
        'HSS': (2.0 * (a * d - b * c)) / max(
            (a + c) * (c + d) + (a + b) * (b + d), 1.0
        ),
        'ACC': (a + d) / max(n, 1.0),
        'BIAS': (a + b) / max(a + c, 1.0),
        'a_hits': a,
        'b_false_alarms': b,
        'c_misses': c,
    }


# ---------------------------------------------------------------------------
# Per-category metrics
# ---------------------------------------------------------------------------

def compute_category_accuracy(
    P_hat: np.ndarray,
    P_true: np.ndarray,
    categories: Dict[str, Tuple[float, float]] = None,
) -> Dict[str, float]:
    """Compute classification accuracy per CMA precipitation category.

    For each category, what fraction of true pixels in that category
    are correctly predicted to be in that category?

    Args:
        P_hat: [H, W] predicted precipitation.
        P_true: [H, W] ground truth.
        categories: Dict of name -> (lo, hi) mm/h thresholds.

    Returns:
        Dict of 'acc_{name}' -> value.
    """
    if categories is None:
        categories = CMA_CATEGORIES

    acc = {}
    for name, (lo, hi) in categories.items():
        mask_true = (P_true >= lo) & (P_true < hi)
        mask_pred = (P_hat >= lo) & (P_hat < hi)
        n_true = mask_true.sum()
        if n_true > 0:
            correct = (mask_true & mask_pred).sum()
            acc[f'acc_{name}'] = float(correct / n_true)
        else:
            acc[f'acc_{name}'] = float('nan')

    return acc


# ---------------------------------------------------------------------------
# Combined: all metrics
# ---------------------------------------------------------------------------

def compute_all_metrics(
    P_hat: np.ndarray,
    P_true: np.ndarray,
    thresholds: List[float] = None,
    pixel_area_km2: float = 100.0,
    resolution_km: float = 10.0,
) -> Dict[str, float]:
    """Compute all evaluation metrics at once.

    Args:
        P_hat: [H, W] or [B, H, W] predicted precipitation.
        P_true: [H, W] or [B, H, W] ground truth.
        thresholds: Precipitation thresholds for categorical metrics.
        pixel_area_km2: Area per pixel.
        resolution_km: Grid resolution.

    Returns:
        Dict of all metric names -> values.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    # Handle batched input
    if P_hat.ndim == 3:
        results = {}
        all_metrics = [
            compute_all_metrics(
                P_hat[i], P_true[i], thresholds,
                pixel_area_km2, resolution_km,
            )
            for i in range(P_hat.shape[0])
        ]
        for key in all_metrics[0]:
            vals = [m[key] for m in all_metrics if not np.isnan(m.get(key, float('nan')))]
            results[key] = float(np.mean(vals)) if vals else float('nan')
        return results

    # Single frame
    metrics = {}

    # Continuous
    metrics.update(
        compute_continuous_metrics(P_hat, P_true, pixel_area_km2, resolution_km)
    )

    # Categorical (multiple thresholds)
    for thresh in thresholds:
        cat = compute_categorical_metrics(P_hat, P_true, thresh)
        for k, v in cat.items():
            metrics[f'{k}_{thresh:.0f}mmh'] = v

    # Per-category accuracy
    metrics.update(compute_category_accuracy(P_hat, P_true))

    return metrics


# ---------------------------------------------------------------------------
# Helper: center displacement
# ---------------------------------------------------------------------------

def _center_displacement(
    P_hat: np.ndarray,
    P_true: np.ndarray,
    resolution_km: float,
) -> float:
    """Compute distance between precipitation centroids."""
    h, w = P_hat.shape
    y, x = np.mgrid[0:h, 0:w]

    mass_true = P_true.sum()
    mass_pred = P_hat.sum()

    if mass_true < 1e-6 or mass_pred < 1e-6:
        return float('nan')

    cx_true = (x * P_true).sum() / mass_true
    cy_true = (y * P_true).sum() / mass_true
    cx_pred = (x * P_hat).sum() / mass_pred
    cy_pred = (y * P_hat).sum() / mass_pred

    dist_pix = np.sqrt((cx_pred - cx_true) ** 2 + (cy_pred - cy_true) ** 2)
    return dist_pix * resolution_km
