"""Evaluation protocol v2 — pooled, level-qualified metrics."""

from .metrics import (
    DEFAULT_THRESHOLDS,
    SSIM_DATA_RANGE,
    NEGATIVE_TOLERANCE,
    categorical_from_counts,
    compute_contingency_counts,
    compute_continuous_suff_stats,
    compute_window_diagnostics,
    compute_window_ssim,
    continuous_from_suff,
    threshold_key,
    validate_finite,
)
from .evaluator import (
    PROTOCOL_ID,
    evaluate_model_v2,
    paired_event_differences,
    predict_absolute,
)
from .reporting import write_v2_json, write_v2_csv, write_v2_markdown

__all__ = [
    'DEFAULT_THRESHOLDS',
    'SSIM_DATA_RANGE',
    'NEGATIVE_TOLERANCE',
    'categorical_from_counts',
    'compute_contingency_counts',
    'compute_continuous_suff_stats',
    'compute_window_diagnostics',
    'compute_window_ssim',
    'continuous_from_suff',
    'threshold_key',
    'validate_finite',
    'PROTOCOL_ID',
    'evaluate_model_v2',
    'paired_event_differences',
    'predict_absolute',
    'write_v2_json',
    'write_v2_csv',
    'write_v2_markdown',
]
