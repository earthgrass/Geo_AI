"""V2 result writers (JSON / CSV / Markdown).

The machine-readable JSON is authoritative. CSV and Markdown are derived views
that iterate the ACTUAL v2 result fields — they never hand-construct metric
keys (e.g. they can never emit ``CSI_10.0mmh`` while the metrics emit
``CSI_10mmh``).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Optional

from .metrics import threshold_key


def write_v2_json(result: Dict, out_path: str, model_name: str = "",
                  extra: Optional[Dict] = None) -> Path:
    """Write the authoritative machine-readable JSON result."""
    payload = {"model": model_name, **(extra or {}), "result": result}
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False,
                   allow_nan=True, default=str),
        encoding="utf-8",
    )
    return path


def write_v2_csv(result: Dict, out_path: str, model_name: str = "") -> Path:
    """Long-format CSV: one row per (model, level, metric, value)."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "level", "metric", "value"])
        for level_name, level_dict in _flatten_levels(result):
            if isinstance(level_dict, dict):
                for key, value in _sorted_flat(level_dict):
                    writer.writerow([model_name, level_name, key,
                                     _fmt(value)])
    return path


def write_v2_markdown(result: Dict, out_path: str, model_name: str = "",
                      header: Optional[Dict] = None) -> Path:
    """Markdown report iterating actual v2 fields."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = [f"## Experiment: {model_name}"]
    rows.append("")
    rows.append(f"- protocol_id: {result.get('protocol_id')}")
    rows.append(f"- split: {result.get('split')}")
    rows.append(f"- test_status: {result.get('test_status')}")
    rows.append(f"- n_events: {result.get('n_events')}")
    rows.append(f"- n_windows: {result.get('n_windows')}")
    rows.append(f"- n_pixels: {result.get('n_pixels')}")
    for k, v in (header or {}).items():
        rows.append(f"- {k}: {v}")
    rows.append("")

    # --- Overall (pooled) ---
    rows.append("### Overall validation metrics (pooled / protocol v2)")
    rows.append("")
    rows.append("| Metric | Value |")
    rows.append("|---|---|")
    og = result.get("overall_global", {})
    for key in ("MAE_global", "RMSE_global", "peak_error_global"):
        if key in og:
            rows.append(f"| {key} | {_fmt(og[key])} |")
    for key, value in _sorted_flat(og.get("categorical", {})):
        rows.append(f"| {key} | {_fmt(value)} |")
    rows.append("")

    # --- Per-event ---
    rows.append("### Per-event validation metrics (pooled within event)")
    rows.append("")
    per_event = result.get("per_event", {})
    if per_event:
        sample = next(iter(per_event.values()))
        cat_cols = []
        for tau in result.get("thresholds", []):
            k = threshold_key(tau)
            for metric in ("CSI", "POD", "FAR", "HSS", "ACC", "BIAS"):
                cat_cols.append(f"{metric}_{k}")
        cols = ["typhoon_id", "MAE_event", "RMSE_event", "SSIM_event_mean"] + cat_cols
        rows.append("| " + " | ".join(cols) + " |")
        rows.append("|" + "---|" * len(cols))
        for tid in sorted(per_event, key=int):
            ev = per_event[tid]
            vals = [tid,
                    _fmt(ev.get("MAE_event")),
                    _fmt(ev.get("RMSE_event")),
                    _fmt(ev.get("SSIM_event_mean"))]
            for tau in result.get("thresholds", []):
                k = threshold_key(tau)
                cat = ev.get("categorical", {}).get(k, {})
                for metric in ("CSI", "POD", "FAR", "HSS", "ACC", "BIAS"):
                    vals.append(_fmt(cat.get(metric)))
            rows.append("| " + " | ".join(vals) + " |")
        rows.append("")

    # --- Window diagnostics ---
    owm = result.get("overall_window_mean", {})
    if owm:
        rows.append("### Window diagnostics (mean of per-window values)")
        rows.append("")
        rows.append("| Metric | Value |")
        rows.append("|---|---|")
        for key in ("MAE_window_mean", "RMSE_window_mean",
                    "SSIM_window_mean", "peak_error_window_mean",
                    "n_defined_windows"):
            if key in owm:
                rows.append(f"| {key} | {_fmt(owm[key])} |")
        rows.append("")

    rows.append(f"- n_negative_roundoff_clamped: "
                f"{result.get('n_negative_roundoff_clamped', 0)}")

    path.write_text("\n".join(rows), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flatten_levels(result: Dict):
    """Yield (level_name, dict) pairs for the level-qualified sections."""
    levels = []
    for key in ("overall_global", "overall_window_mean",
                "overall_event_macro", "per_event"):
        if key in result:
            levels.append((key, result[key]))
    return levels


def _sorted_flat(d: Dict) -> list:
    """Flatten nested dicts to (dotted_key, value) pairs, sorted by key."""
    out: list = []

    def walk(prefix: str, obj):
        if isinstance(obj, dict):
            for k in sorted(obj):
                walk(f"{prefix}.{k}" if prefix else k, obj[k])
        else:
            out.append((prefix, obj))

    walk("", d)
    return out


def _fmt(value) -> str:
    if value is None or (isinstance(value, float) and value != value):  # NaN
        return "nan"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)
