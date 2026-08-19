"""Consistency tests pinning numeric interpretation + canonical source.

These tests guard two correctness invariants that the analyzer must satisfy
on the real GPU artifacts:

1. **Improvement-delta direction**: ``paired_event_differences`` returns
   positive diff = candidate is better than baseline, regardless of
   whether the metric is lower-is-better (MAE/RMSE/FAR) or
   higher-is-better (SSIM/CSI/POD/HSS). Verified by asserting that the
   sign of the event-mean diff on the canonical P0/P1/P2/P3 contrasts
   matches the user's pre-registered interpretation
   (P1 better, P2 worse, P3 worse).

2. **Source-of-truth = canonical result_v2.json**: the
   ``experiment_summary.csv`` MAE_event_mean / RMSE_event_mean /
   SSIM_event_mean_mean columns must equal the equal-event mean of
   per_event[<tid>][metric] in the canonical
   ``results/<exp>/result_v2.json`` (after SHA256-verified archival). The
   script must never report numbers from scratch dirs, legacy CSV
   dumps, or hand-edited values.

The tests do NOT modify evaluator semantics. They only verify the
contract: the analyzer's number = (per-event equal mean of canonical
result_v2.json per_event), with the documented sign convention.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
TABLES_DIR = REPO_ROOT / "tables" / "ablation_analysis"


def _load_canonical_result_v2(canonical_dir: Path) -> dict:
    """Read result_v2.json (wrapper {"model": str, "result": inner}) from a
    canonical results/<exp> dir."""
    path = canonical_dir / "result_v2.json"
    assert path.exists(), f"missing canonical {path}"
    wrapper = json.loads(path.read_text(encoding="utf-8"))
    assert "result" in wrapper, (
        f"{path}: wrapper format violated (no 'result' key). "
        f"Wrapper must be {{'model': ..., 'result': ...}}."
    )
    return wrapper["result"]


def _per_event_equal_mean(inner: dict, metric: str) -> float:
    """Equal-event mean of per_event[<tid>][metric], NaN-skipping."""
    pe = inner["per_event"]
    vals = [v[metric] for v in pe.values() if v.get(metric) is not None]
    vals = [v for v in vals if v == v]  # NaN skip
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def _load_summary_row(experiment_id: str) -> dict | None:
    """Read one row from tables/ablation_analysis/experiment_summary.csv."""
    summary_path = TABLES_DIR / "experiment_summary.csv"
    if not summary_path.exists():
        return None
    with summary_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["experiment_id"] == experiment_id:
                return row
    return None


def _load_long_contrast(axis: str, baseline: str, candidate: str,
                        metric: str, threshold_mmh: str) -> dict | None:
    """Read one row from tables/ablation_analysis/contrasts_long.csv."""
    long_path = TABLES_DIR / "contrasts_long.csv"
    if not long_path.exists():
        return None
    with long_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row["axis"] == axis
                and row["baseline"] == baseline
                and row["candidate"] == candidate
                and row["metric"] == metric
                and row["threshold_mmh"] == threshold_mmh):
                return row
    return None


# ---------------------------------------------------------------------------
# Skip unless both canonical + tables exist (so the test suite is still
# runnable in environments where one side has not been generated yet).
# ---------------------------------------------------------------------------

requires_canonical = pytest.mark.skipif(
    not (RESULTS_DIR / "I5_terrain_geometry_seed42" / "result_v2.json").exists()
    or not (TABLES_DIR / "experiment_summary.csv").exists(),
    reason="canonical result_v2.json or summary CSV not present; "
           "run archive_validation_results.py + analyze_ablation_results.py first",
)


# ===========================================================================
# 1. Source-of-truth consistency: experiment_summary == canonical per_event
# ===========================================================================

@requires_canonical
def test_b1_canonical_event_means_match_summary():
    """B1 row in experiment_summary.csv must equal canonical per_event means."""
    inner = _load_canonical_result_v2(RESULTS_DIR / "B1_trajgru_seed42")
    row = _load_summary_row("B1_trajgru")
    assert row is not None, "B1_trajgru row missing from experiment_summary.csv"
    for col, metric in (
        ("MAE_event_mean", "MAE_event"),
        ("RMSE_event_mean", "RMSE_event"),
        ("SSIM_event_mean_mean", "SSIM_event_mean"),
    ):
        canonical = _per_event_equal_mean(inner, metric)
        reported = float(row[col])
        assert abs(canonical - reported) < 1e-9, (
            f"{col}: canonical={canonical!r} differs from "
            f"summary={reported!r}. The summary must be generated from "
            f"canonical result_v2.json per_event, not from any other source."
        )


@requires_canonical
def test_p0_i5_canonical_event_means_match_summary():
    """P0 ≡ I5 (single artifact) canonical per_event must equal summary row."""
    inner = _load_canonical_result_v2(RESULTS_DIR / "I5_terrain_geometry_seed42")
    row = _load_summary_row("E5_terrain_geometry")
    assert row is not None, "E5_terrain_geometry row missing"
    for col, metric in (
        ("MAE_event_mean", "MAE_event"),
        ("RMSE_event_mean", "RMSE_event"),
        ("SSIM_event_mean_mean", "SSIM_event_mean"),
    ):
        canonical = _per_event_equal_mean(inner, metric)
        reported = float(row[col])
        assert abs(canonical - reported) < 1e-9, (
            f"{col}: canonical={canonical!r} differs from "
            f"summary={reported!r}."
        )


@requires_canonical
@pytest.mark.parametrize("exp_dir,exp_id", [
    ("P1_smooth_seed42", "P1_resconvlstm_smooth"),
    ("P2_extreme_seed42", "E6_terrain_extreme"),
    ("P3_smooth_extreme_seed42", "P3_resconvlstm_smooth_extreme"),
    ("I0_persistence_seed42", "E0_persistence"),
    ("I1_plain_convlstm_seed42", "E1_plain_convlstm"),
    ("I2_resconvlstm_seed42_v2", "E2_resconvlstm"),
    ("I3_resconvlstm_cma_seed42", "E3_resconvlstm_cma"),
    ("I4_static_terrain_seed42", "E4_static_terrain"),
])
def test_all_canonical_event_means_match_summary(exp_dir, exp_id):
    """Every canonical row in experiment_summary.csv must equal canonical
    per_event means — guards against stale/legacy/scratch contamination."""
    inner = _load_canonical_result_v2(RESULTS_DIR / exp_dir)
    row = _load_summary_row(exp_id)
    assert row is not None, f"{exp_id} row missing from experiment_summary.csv"
    for col, metric in (
        ("MAE_event_mean", "MAE_event"),
        ("RMSE_event_mean", "RMSE_event"),
        ("SSIM_event_mean_mean", "SSIM_event_mean"),
    ):
        canonical = _per_event_equal_mean(inner, metric)
        reported = float(row[col])
        assert abs(canonical - reported) < 1e-9, (
            f"{exp_id}.{col}: canonical={canonical!r} != summary={reported!r}"
        )


# ===========================================================================
# 2. Improvement-delta direction: positive diff = candidate better
# ===========================================================================

@requires_canonical
def test_p1_p0_mae_improvement_delta_is_positive():
    """P1 MAE < P0 MAE ⇒ improvement_delta > 0 (P1 is BETTER on MAE).

    This is the canonical interpretation; the audit prose must match.
    """
    row = _load_long_contrast("AxisII", "P0", "P1", "MAE_event", "")
    assert row is not None, "P1-P0 MAE_event contrast missing from contrasts_long.csv"
    diff = float(row["mean_diff"])
    assert diff > 0, (
        f"P1 - P0 MAE_event mean_diff must be > 0 (P1 is better), "
        f"got {diff}. The improvement_delta convention is "
        f"baseline - candidate for lower-is-better metrics; a positive "
        f"value means the candidate is better. See "
        f"src/evaluation/evaluator.py::paired_event_differences docstring."
    )


@requires_canonical
def test_p1_p0_rmse_improvement_delta_is_positive():
    """P1 RMSE < P0 RMSE ⇒ improvement_delta > 0 (P1 is BETTER on RMSE)."""
    row = _load_long_contrast("AxisII", "P0", "P1", "RMSE_event", "")
    assert row is not None
    diff = float(row["mean_diff"])
    assert diff > 0, f"P1 - P0 RMSE_event mean_diff must be > 0, got {diff}"


@requires_canonical
def test_p1_p0_ssim_improvement_delta_is_positive():
    """P1 SSIM > P0 SSIM ⇒ improvement_delta > 0 (P1 is BETTER on SSIM)."""
    row = _load_long_contrast("AxisII", "P0", "P1", "SSIM_event_mean", "")
    assert row is not None
    diff = float(row["mean_diff"])
    assert diff > 0, f"P1 - P0 SSIM_event_mean mean_diff must be > 0, got {diff}"


@requires_canonical
def test_p2_p0_mae_improvement_delta_is_negative():
    """P2 MAE > P0 MAE ⇒ improvement_delta < 0 (P2 is WORSE on MAE)."""
    row = _load_long_contrast("AxisII", "P0", "P2", "MAE_event", "")
    assert row is not None
    diff = float(row["mean_diff"])
    assert diff < 0, (
        f"P2 - P0 MAE_event mean_diff must be < 0 (P2 is worse), "
        f"got {diff}. improvement_delta is baseline - candidate for "
        f"lower-is-better metrics."
    )


@requires_canonical
def test_p2_p0_rmse_improvement_delta_is_negative():
    """P2 RMSE > P0 RMSE ⇒ improvement_delta < 0 (P2 is WORSE on RMSE)."""
    row = _load_long_contrast("AxisII", "P0", "P2", "RMSE_event", "")
    assert row is not None
    diff = float(row["mean_diff"])
    assert diff < 0, f"P2 - P0 RMSE_event mean_diff must be < 0, got {diff}"


@requires_canonical
def test_p2_p0_ssim_improvement_delta_is_negative():
    """P2 SSIM < P0 SSIM ⇒ improvement_delta < 0 (P2 is WORSE on SSIM)."""
    row = _load_long_contrast("AxisII", "P0", "P2", "SSIM_event_mean", "")
    assert row is not None
    diff = float(row["mean_diff"])
    assert diff < 0, f"P2 - P0 SSIM_event_mean mean_diff must be < 0, got {diff}"


@requires_canonical
def test_p3_p0_mae_improvement_delta_is_negative():
    """P3 MAE > P0 MAE ⇒ improvement_delta < 0 (P3 is WORSE on MAE)."""
    row = _load_long_contrast("AxisII", "P0", "P3", "MAE_event", "")
    assert row is not None
    diff = float(row["mean_diff"])
    assert diff < 0, f"P3 - P0 MAE_event mean_diff must be < 0, got {diff}"


# ===========================================================================
# 3. Interpretation-generation contract: positive diff = "BETTER" prose
# ===========================================================================

@requires_canonical
def test_directional_prose_p1_is_better_on_continuous():
    """All three P1 continuous diffs > 0 ⇒ generated interpretation is
    P1 is BETTER on every continuous metric.

    This test does NOT compare hard-coded prose; it asserts the SIGN of
    every continuous diff, so any downstream text generator that maps
    sign→BETTER/WORSE must produce BETTER for P1.
    """
    for metric in ("MAE_event", "RMSE_event", "SSIM_event_mean"):
        row = _load_long_contrast("AxisII", "P0", "P1", metric, "")
        assert row is not None, f"P1-P0 {metric} contrast missing"
        diff = float(row["mean_diff"])
        assert diff > 0, (
            f"P1 - P0 {metric} must have mean_diff > 0 for P1 to be "
            f"interpreted as BETTER. Got {diff}. If this test fails, "
            f"either the canonical artifact changed or the diff "
            f"convention was inverted. STOP and investigate."
        )


@requires_canonical
def test_directional_prose_p2_is_worse_on_continuous():
    """All three P2 continuous diffs < 0 ⇒ generated interpretation is
    P2 is WORSE on every continuous metric.

    Trade-off note: P2 may increase POD on extreme-rain detection; that
    is a separate categorical@τ statement, NOT a continuous-metric
    BETTER claim.
    """
    for metric in ("MAE_event", "RMSE_event", "SSIM_event_mean"):
        row = _load_long_contrast("AxisII", "P0", "P2", metric, "")
        assert row is not None, f"P2-P0 {metric} contrast missing"
        diff = float(row["mean_diff"])
        assert diff < 0, (
            f"P2 - P0 {metric} must have mean_diff < 0 for P2 to be "
            f"interpreted as WORSE on continuous metrics. Got {diff}. "
            f"Trade-off: P2 may simultaneously increase extreme-rain "
            f"POD — that is a separate categorical@τ statement."
        )


# ===========================================================================
# 4. Canonical B1 sanity (regression guard against hand-edited numbers)
# ===========================================================================

CANONICAL_B1_VALUES = {
    # csv col in experiment_summary.csv → metric key in result_v2 per_event
    "MAE_event_mean": ("MAE_event", 0.2332007103049345),
    "RMSE_event_mean": ("RMSE_event", 0.8935367958375073),
    "SSIM_event_mean_mean": ("SSIM_event_mean", 0.9380130764634814),
}


@requires_canonical
def test_b1_canonical_values_match_known_truth():
    """B1 canonical MAE / RMSE / SSIM event means must equal the values
    computed directly from ``results/B1_trajgru_seed42/result_v2.json``
    per_event. Guards against silent substitution of stale or hand-edited
    numbers in the audit/manuscript tables.
    """
    inner = _load_canonical_result_v2(RESULTS_DIR / "B1_trajgru_seed42")
    for col, (metric, truth) in CANONICAL_B1_VALUES.items():
        canonical = _per_event_equal_mean(inner, metric)
        assert abs(canonical - truth) < 1e-9, (
            f"B1.{col} canonical value {canonical!r} differs from "
            f"pinned truth {truth!r}. Either the source artifact "
            f"changed (re-run archive) or a stale value has been "
            f"substituted. STOP and re-run from "
            f"Geo_AI_validation_results_ONLY_20260819.tar.gz."
        )


# ===========================================================================
# 5. n_events / n_windows sanity (regression guard)
# ===========================================================================

@requires_canonical
@pytest.mark.parametrize("exp_dir,n_events,n_windows,split", [
    ("I0_persistence_seed42", 7, 1266, "val"),
    ("I2_resconvlstm_seed42_v2", 7, 1266, "val"),
    ("I5_terrain_geometry_seed42", 7, 1266, "val"),
    ("P1_smooth_seed42", 7, 1266, "val"),
    ("P2_extreme_seed42", 7, 1266, "val"),
    ("B1_trajgru_seed42", 7, 1266, "val"),
])
def test_canonical_n_events_n_windows_split(exp_dir, n_events, n_windows, split):
    inner = _load_canonical_result_v2(RESULTS_DIR / exp_dir)
    assert inner["n_events"] == n_events
    assert inner["n_windows"] == n_windows
    assert inner["split"] == split
    assert inner["test_status"] == "SEALED"
    assert inner["protocol_id"] == "evaluation_v2"