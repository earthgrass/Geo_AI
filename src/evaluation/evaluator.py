"""Evaluator V2 — pooled, level-qualified aggregation.

Implements ``docs/EVALUATION_PROTOCOL_V2.md`` exactly:

- emits four named levels: ``per_window``, ``per_event``, ``overall_global``,
  and ``overall_window_mean`` (plus the optional ``overall_event_macro``);
- pools continuous sufficient statistics and integer contingency counts
  ``a,b,c,d`` BEFORE computing any ratio (never the mean of window ratios);
- fixed-range SSIM (``data_range = 100 mm/h``), global RMSE, NaN zero
  denominators;
- channel-subset safety: asserts ``X.shape[2] == len(channel_indices)`` and
  treats precipitation as subset position 0 — it never re-applies canonical
  channel indices to the already-subset tensor;
- fails on non-finite predictions/targets and on negative predictions below
  tolerance ``-1e-7`` (ReLU architecture should produce none);
- records ``protocol_id=evaluation_v2``, split, test status, and
  undefined-value counts on every output.

Test-set policy: this module has no knowledge of test IDs and no flag to
enable test evaluation. Ordinary entry points (run_experiment.py,
evaluate_checkpoint.py) refuse ``--split test`` / any test path while the seal
is active.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

from .metrics import (
    DEFAULT_THRESHOLDS,
    NEGATIVE_TOLERANCE,
    categorical_from_counts,
    clamp_negative_tiny,
    compute_window_diagnostics,
    continuous_from_suff,
    threshold_key,
    validate_finite,
)

PROTOCOL_ID = "evaluation_v2"


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_absolute(model, X: torch.Tensor) -> torch.Tensor:
    """Return the absolute precipitation prediction ``[B, 1, H, W]``.

    Channel semantics: the DataLoader already returns the configured channel
    subset (``TyphoonDataset`` builds the canonical 12 channels and selects
    ``channel_indices``). Precipitation is ALWAYS subset position 0 for every
    frozen channel list, so the evaluator never re-applies canonical indices.

    - ResConvLSTM / PI-ResConvLSTM expose ``compute_prediction`` (absolute).
    - PlainConvLSTM / TrajGRU / Persistence are precipitation-only absolute
      models; precipitation is taken from subset position 0.
    """
    if hasattr(model, "compute_prediction"):
        return model.compute_prediction(X)
    if X.shape[2] > 1:
        X = X[:, :, 0:1, :, :]
    return model(X)


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_model_v2(
    model,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    precip_vmax: float = 100.0,
    thresholds: Optional[List[float]] = None,
    channel_indices: Optional[List[int]] = None,
    split: str = "val",
    test_status: str = "SEALED",
) -> Dict:
    """Evaluate a model on a loader with the frozen v2 protocol.

    Args:
        model: trained model (absolute precipitation contract).
        loader: DataLoader yielding ``(X, Y, meta)`` where X has already been
            channel-subset by the dataset.
        device: torch device.
        precip_vmax: precipitation min-max scale (mm/h) used for denormalization.
        thresholds: categorical thresholds (mm/h); defaults to frozen [5,10,20,30].
        channel_indices: canonical channel list used to build the loader; used
            ONLY to assert the loader produced that many channels. Never re-sliced.
        split: split name recorded in the output ("val").
        test_status: recorded as SEALED while the final-test phase is closed.

    Returns:
        Level-qualified result dict (see module docstring).
    """
    if thresholds is None:
        thresholds = list(DEFAULT_THRESHOLDS)
    if channel_indices is not None and len(channel_indices) == 0:
        raise ValueError("channel_indices must be a non-empty canonical list.")

    if isinstance(model, torch.nn.Module):
        model.eval()
    per_window: List[Dict] = []
    n_clamped_total = 0

    for X, Y, meta in loader:
        X = X.to(device)
        Y = Y.to(device)

        if channel_indices is not None:
            if X.shape[2] != len(channel_indices):
                raise ValueError(
                    f"Channel-subset mismatch: loader produced {X.shape[2]} "
                    f"channels but config declares {len(channel_indices)} "
                    f"({channel_indices}). The evaluator must never re-slice a "
                    "subset tensor with canonical indices."
                )

        P_hat = predict_absolute(model, X)
        P_hat_np = P_hat.detach().cpu().numpy()[:, 0]      # [B, H, W]
        Y_np = Y.detach().cpu().numpy()[:, 0]

        P_hat_np = P_hat_np * precip_vmax
        Y_np = Y_np * precip_vmax

        validate_finite(P_hat_np, Y_np)
        n_neg = clamp_negative_tiny(P_hat_np)
        n_clamped_total += n_neg
        if (P_hat_np < NEGATIVE_TOLERANCE).any():
            raise ValueError(
                "Negative prediction below tolerance "
                f"({NEGATIVE_TOLERANCE}) encountered. The ReLU architecture "
                "should never produce these."
            )

        tids = meta.get("typhoon_id")
        for i in range(P_hat_np.shape[0]):
            tid = int(tids[i]) if tids is not None else -1
            diag = compute_window_diagnostics(P_hat_np[i], Y_np[i], thresholds)
            per_window.append({"typhoon_id": tid, **diag})

    result = aggregate_v2(
        per_window, thresholds,
        split=split, test_status=test_status,
    )
    result["n_negative_roundoff_clamped"] = n_clamped_total
    return result


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_v2(
    per_window: List[Dict],
    thresholds: Optional[List[float]] = None,
    split: str = "val",
    test_status: str = "SEALED",
    protocol_id: str = PROTOCOL_ID,
) -> Dict:
    """Aggregate per-window records into the four v2 levels."""
    if thresholds is None:
        thresholds = list(DEFAULT_THRESHOLDS)

    # ---- pooled sufficient statistics ----
    g_cont = {"sum_abs": 0.0, "sum_sq": 0.0, "n": 0}
    g_cat = {tau: {"a": 0, "b": 0, "c": 0, "d": 0} for tau in thresholds}
    ssim_vals: List[float] = []
    mae_window_vals: List[float] = []
    rmse_window_vals: List[float] = []
    peak_window_vals: List[float] = []
    max_f_overall: float = -np.inf
    max_y_overall: float = -np.inf

    events: Dict[int, Dict] = {}

    for w in per_window:
        tid = w["typhoon_id"]
        g_cont["sum_abs"] += w["sum_abs"]
        g_cont["sum_sq"] += w["sum_sq"]
        g_cont["n"] += w["n"]
        for tau in thresholds:
            c = w["counts"][threshold_key(tau)]
            g_cat[tau]["a"] += c["a"]
            g_cat[tau]["b"] += c["b"]
            g_cat[tau]["c"] += c["c"]
            g_cat[tau]["d"] += c["d"]

        ssim_vals.append(w["SSIM"])
        mae_window_vals.append(w["sum_abs"] / w["n"])
        rmse_window_vals.append(float(np.sqrt(w["sum_sq"] / w["n"])))
        peak_window_vals.append(w["peak_error"])
        max_f_overall = max(max_f_overall, w["f_max"])
        max_y_overall = max(max_y_overall, w["y_max"])

        ev = events.setdefault(tid, {
            "cont": {"sum_abs": 0.0, "sum_sq": 0.0, "n": 0},
            "cat": {tau: {"a": 0, "b": 0, "c": 0, "d": 0} for tau in thresholds},
            "ssim": [], "peak": [],
            "f_max": -np.inf, "y_max": -np.inf,
            "n_windows": 0,
        })
        ev["cont"]["sum_abs"] += w["sum_abs"]
        ev["cont"]["sum_sq"] += w["sum_sq"]
        ev["cont"]["n"] += w["n"]
        for tau in thresholds:
            c = w["counts"][threshold_key(tau)]
            for k in ("a", "b", "c", "d"):
                ev["cat"][tau][k] += c[k]
        ev["ssim"].append(w["SSIM"])
        ev["peak"].append(w["peak_error"])
        ev["f_max"] = max(ev["f_max"], w["f_max"])
        ev["y_max"] = max(ev["y_max"], w["y_max"])
        ev["n_windows"] += 1

    n_windows = len(per_window)
    n_pixels = g_cont["n"]

    # ---- overall_global (primary pooled scores) ----
    g_continuous = continuous_from_suff(g_cont)
    overall_global = {
        "MAE_global": g_continuous["MAE"],
        "RMSE_global": g_continuous["RMSE"],
        "categorical": {
            threshold_key(tau): _categorical_level(g_cat[tau])
            for tau in thresholds
        },
        "peak_error_global": _abs_peak(max_f_overall, max_y_overall),
    }

    # ---- overall_window_mean (diagnostics, never primary categorical) ----
    overall_window_mean = {
        "MAE_window_mean": _mean(mae_window_vals),
        "RMSE_window_mean": _mean(rmse_window_vals),
        "SSIM_window_mean": _mean(ssim_vals),
        "peak_error_window_mean": _mean(peak_window_vals),
        "n_defined_windows": n_windows,
    }

    # ---- per_event ----
    per_event: Dict[str, Dict] = {}
    for tid in sorted(events):
        ev = events[tid]
        ev_cont = continuous_from_suff(ev["cont"])
        per_event[str(tid)] = {
            "MAE_event": ev_cont["MAE"],
            "RMSE_event": ev_cont["RMSE"],
            "SSIM_event_mean": _mean(ev["ssim"]),
            "categorical": {
                threshold_key(tau): _categorical_level(ev["cat"][tau])
                for tau in thresholds
            },
            "peak_error_event": _abs_peak(ev["f_max"], ev["y_max"]),
            "n_windows": ev["n_windows"],
        }

    # ---- overall_event_macro (interpretation only; never the pooled score) ----
    overall_event_macro = _event_macro(per_event, thresholds)

    return {
        "protocol_id": protocol_id,
        "split": split,
        "test_status": test_status,
        "thresholds": [float(t) for t in thresholds],
        "n_events": len(per_event),
        "n_windows": n_windows,
        "n_pixels": n_pixels,
        "overall_global": overall_global,
        "overall_window_mean": overall_window_mean,
        "overall_event_macro": overall_event_macro,
        "per_event": per_event,
        "per_window": per_window,
        "undefined_value_counts": _undefined_counts(overall_global,
                                                    overall_window_mean,
                                                    per_event, thresholds),
    }


def _categorical_level(counts: Dict[str, int]) -> Dict:
    """Scores + counts from pooled a,b,c,d (undefined ratios become NaN)."""
    return categorical_from_counts(counts["a"], counts["b"], counts["c"], counts["d"])


def _abs_peak(f_max: float, y_max: float) -> float:
    return float(np.abs(f_max - y_max))


def _mean(vals: List[float]) -> float:
    finite = [v for v in vals if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


def _event_macro(per_event: Dict[str, Dict], thresholds: List[float]) -> Dict:
    """Equal-event mean of defined per-event scores (interpretation only)."""
    out: Dict[str, float] = {}
    for metric in ("MAE_event", "RMSE_event", "SSIM_event_mean"):
        out[metric] = _mean([ev[metric] for ev in per_event.values()])
    for tau in thresholds:
        key = threshold_key(tau)
        for metric in ("CSI", "POD", "FAR", "HSS", "ACC", "BIAS"):
            out[f"{metric}_{key}_event_macro"] = _mean(
                [ev["categorical"][key][metric] for ev in per_event.values()]
            )
    out["n_events"] = len(per_event)
    return out


def _undefined_counts(overall_global: Dict, overall_window_mean: Dict,
                      per_event: Dict[str, Dict],
                      thresholds: List[float]) -> Dict[str, int]:
    """Count undefined (NaN) values per metric across the emitted levels."""
    counts: Dict[str, int] = {}

    def _bump(key: str, value: float) -> None:
        if not np.isfinite(value):
            counts[key] = counts.get(key, 0) + 1

    for key in ("MAE_global", "RMSE_global", "peak_error_global"):
        _bump(key, overall_global[key])
    for key in ("MAE_window_mean", "RMSE_window_mean", "SSIM_window_mean"):
        _bump(key, overall_window_mean[key])
    for tau in thresholds:
        k = threshold_key(tau)
        for metric in ("CSI", "POD", "FAR", "HSS", "ACC", "BIAS"):
            _bump(f"{metric}_{k}_global", overall_global["categorical"][k][metric])
    for tid, ev in per_event.items():
        for metric in ("MAE_event", "RMSE_event", "SSIM_event_mean"):
            _bump(f"{metric}_event[{tid}]", ev[metric])
        for tau in thresholds:
            k = threshold_key(tau)
            for metric in ("CSI", "POD", "FAR", "HSS", "ACC", "BIAS"):
                _bump(f"{metric}_{k}_event[{tid}]", ev["categorical"][k][metric])
    return counts


# ---------------------------------------------------------------------------
# Paired event analysis (frozen protocol v2 §17)
# ---------------------------------------------------------------------------

LOWER_IS_BETTER = {"MAE_event", "RMSE_event", "FAR"}
HIGHER_IS_BETTER = {"SSIM_event_mean", "CSI", "POD", "HSS", "ACC"}


def extract_per_event_values(result: Dict, metric: str,
                             threshold: Optional[float] = None) -> Dict[str, float]:
    """Return ``{typhoon_id: value}`` for one metric at the event level."""
    out: Dict[str, float] = {}
    for tid, ev in result["per_event"].items():
        if threshold is not None:
            out[tid] = ev["categorical"][threshold_key(threshold)][metric]
        elif metric in ev:
            out[tid] = ev[metric]
        else:
            raise KeyError(f"unknown per-event metric {metric!r}")
    return out


def paired_event_differences(
    baseline: Dict,
    candidate: Dict,
    metric: str,
    threshold: Optional[float] = None,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> Dict:
    """Paired event differences for one preregistered contrast.

    Positive differences always mean improvement:
      - lower-is-better (MAE, RMSE, FAR): ``baseline - candidate``;
      - higher-is-better (SSIM, CSI, POD, HSS, ACC): ``candidate - baseline``;
      - BIAS: ``|BIAS_b - 1| - |BIAS_c - 1|`` (both raw values also reported).

    Returns the per-event differences, equal-event summaries, and a 95%
    bootstrap CI over paired event resamples (bootstrap seed 42). Windows are
    never resampled as independent cases.
    """
    b = extract_per_event_values(baseline, metric, threshold)
    c = extract_per_event_values(candidate, metric, threshold)
    common = sorted(set(b) & set(c))

    diffs = []
    raw_pairs = []
    for tid in common:
        bv, cv = b[tid], c[tid]
        if not (np.isfinite(bv) and np.isfinite(cv)):
            continue
        if metric == "BIAS":
            raw_pairs.append((tid, bv, cv))
            diffs.append((tid, abs(bv - 1.0) - abs(cv - 1.0)))
        elif metric in LOWER_IS_BETTER:
            diffs.append((tid, bv - cv))
        elif metric in HIGHER_IS_BETTER:
            diffs.append((tid, cv - bv))
        else:
            raise KeyError(f"unknown direction for metric {metric!r}")

    diff_vals = np.array([d[1] for d in diffs], dtype=float)
    if diff_vals.size == 0:
        return {"metric": metric, "n_pairs": 0, "per_event": {},
                "mean": float("nan"), "median": float("nan"),
                "iqr": float("nan"), "ci95": (float("nan"), float("nan")),
                "raw_biases": {}}

    rng = np.random.RandomState(seed)
    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(diff_vals, size=diff_vals.size, replace=True)
        boot_means[i] = sample.mean()
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

    return {
        "metric": metric,
        "threshold_mmh": threshold,
        "n_pairs": int(diff_vals.size),
        "per_event": {str(t): float(v) for t, v in diffs},
        "mean": float(diff_vals.mean()),
        "median": float(np.median(diff_vals)),
        "iqr": float(np.percentile(diff_vals, 75) - np.percentile(diff_vals, 25)),
        "ci95": (float(ci_lo), float(ci_hi)),
        "raw_biases": {str(t): (float(bv), float(cv)) for t, bv, cv in raw_pairs},
    }
