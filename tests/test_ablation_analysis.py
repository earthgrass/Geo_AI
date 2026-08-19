"""Frozen-contract tests for scripts/analyze_ablation_results.py.

Each test pins one item from the analyze script's contract:
  - metric direction (lower-is-better vs higher-is-better, BIAS=1)
  - event-level pairing + NaN handling
  - n_pairs >= 4 inferential block
  - deterministic bootstrap (seed=42)
  - exact two-sided sign-flip p-value
  - Holm correction
  - fingerprint contract (test_status=SEALED, split=val,
    protocol=evaluation_v2, smoke=false)
  - I5 == P0 same-artifact identity
  - no window-level significance
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_ablation_results import (  # noqa: E402
    exact_two_sided_sign_flip,
    holm_correction,
    analyze,
    enforce_i5_p0_identity,
    load_results,
    ContractViolation,
    _categorical_event_value,
    _emit_contrast,
    BACKBONE_SANITY_PAIRS,
    CONTINUOUS_METRICS,
    EXPECTED_N_EVENTS,
    EXPECTED_N_WINDOWS,
    EXPECTED_PROTOCOL_ID,
    EXPECTED_SMOKE,
    EXPECTED_SPLIT,
    EXPECTED_TEST_STATUS,
    HOLM_FAMILY_MIN_SIZE,
    HOLM_N_PAIRS_MIN,
    PROTOCOL_ID,
)
from src.evaluation.evaluator import (  # noqa: E402
    paired_event_differences,
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic manifest + result_v2 in a temp dir
# ---------------------------------------------------------------------------

def _make_manifest(exp_id: str = "I2_resconvlstm_seed42",
                   aliases: list | None = None,
                   **overrides) -> dict:
    m = {
        "experiment_id": exp_id,
        "alias_ids": aliases if aliases is not None else [exp_id.split("_seed")[0]],
        "seed": 42,
        "n_events_expected": EXPECTED_N_EVENTS,
        "n_windows_expected": EXPECTED_N_WINDOWS,
        "git_commit": "deadbeef" * 4,
        "config_sha256": "ab" * 32,
        "dataset_sha256": "cd" * 32,
        "split_sha256": "ef" * 32,
        "normalization_sha256": "12" * 32,
        "checkpoint_sha256": "34" * 32,
        "protocol_id": EXPECTED_PROTOCOL_ID,
        "test_status": EXPECTED_TEST_STATUS,
        "split": EXPECTED_SPLIT,
        "smoke": EXPECTED_SMOKE,
        "epochs": 20,
        "best_epoch": 17,
    }
    m.update(overrides)
    return m


def _make_result_v2(per_event_metrics: dict[str, dict],
                    n_events: int = EXPECTED_N_EVENTS,
                    n_windows: int = EXPECTED_N_WINDOWS,
                    model: str = "I2_resconvlstm_seed42",
                    flat: bool = False) -> dict:
    """Build the wrapper format produced by ``src.evaluation.reporting.write_v2_json``.

    per_event_metrics: {typhoon_id: {metric: float, categorical: {key: {m: v}, ...}, ...}}

    Returns ``{"model": str, "result": {...evaluator dict...}}``.

    If ``flat=True``, the legacy un-wrapped format is emitted instead — used
    only by negative tests that assert the analyzer rejects it.
    """
    inner = {
        "protocol_id": PROTOCOL_ID,
        "split": EXPECTED_SPLIT,
        "test_status": EXPECTED_TEST_STATUS,
        "smoke": EXPECTED_SMOKE,
        "n_events": n_events,
        "n_windows": n_windows,
        "thresholds": [5.0, 10.0, 20.0, 30.0],
        "per_event": {tid: m for tid, m in per_event_metrics.items()},
    }
    if flat:
        return inner
    return {"model": model, "result": inner}


def _populate_experiment(
    results_dir: Path,
    alias: str,
    per_event_metrics: dict[str, dict],
    *,
    alias_ids: list | None = None,
    manifest_overrides: dict | None = None,
    n_events: int = EXPECTED_N_EVENTS,
    n_windows: int = EXPECTED_N_WINDOWS,
    exp_id: str | None = None,
) -> Path:
    """Create results/<exp_id>/{manifest.json, result_v2.json}."""
    exp_id = exp_id or f"{alias}_seed42"
    alias_ids = alias_ids or [alias]
    d = results_dir / exp_id
    d.mkdir(parents=True, exist_ok=True)
    manifest = _make_manifest(exp_id=exp_id, aliases=alias_ids,
                              **(manifest_overrides or {}))
    (d / "manifest.json").write_text(
        _to_json(manifest), encoding="utf-8")
    (d / "result_v2.json").write_text(
        _to_json(_make_result_v2(per_event_metrics,
                                  n_events=n_events,
                                  n_windows=n_windows)),
        encoding="utf-8")
    return d


def _to_json(obj) -> str:
    import json
    return json.dumps(obj, indent=2, ensure_ascii=False)


def _synth_per_event(values: list[dict]) -> dict[str, dict]:
    """{tid: continuous_metrics + categorical block} for typhoons 1..len."""
    out: dict[str, dict] = {}
    for i, vals in enumerate(values, start=1):
        tid = str(i)
        out[tid] = {
            "MAE_event": vals["mae"],
            "RMSE_event": vals["rmse"],
            "SSIM_event_mean": vals["ssim"],
            "categorical": vals.get("categorical", {}),
        }
    return out


def _cat_buckets(csi, pod, far, hss, bias, acc, tau: float = 5.0) -> dict:
    # Canonical key matches src.evaluation.metrics.threshold_key(tau).
    if float(tau).is_integer():
        key = f"{int(tau)}mmh"
    else:
        key = f"{tau}mmh"
    return {
        key: {"CSI": csi, "POD": pod, "FAR": far, "HSS": hss,
              "BIAS": bias, "ACC": acc},
    }


def _cat_buckets_all(csi, pod, far, hss, bias, acc,
                     taus=(5.0, 10.0, 20.0, 30.0)) -> dict:
    """Return categorical buckets populated for every frozen threshold.

    The analyze script iterates over every (metric, threshold) pair in
    FROZEN_THRESHOLDS_MMH; the test fixtures MUST populate each bucket.
    """
    out: dict = {}
    for tau in taus:
        out.update(_cat_buckets(csi, pod, far, hss, bias, acc, tau))
    return out


# ---------------------------------------------------------------------------
# 1. Metric direction
# ---------------------------------------------------------------------------

def test_metric_direction_lower_is_better(tmp_path):
    """MAE / RMSE / FAR: positive diff = candidate improvement."""
    results_dir = tmp_path / "results"
    # baseline 7 events: MAE=10 each; candidate 7 events: MAE=5 each.
    base = _synth_per_event([{"mae": 10.0, "rmse": 10.0, "ssim": 0.5,
                              "categorical": _cat_buckets(
                                  0.2, 0.4, 0.6, 0.1, 1.5, 0.6, 5.0)}] * 7)
    cand = _synth_per_event([{"mae": 5.0, "rmse": 5.0, "ssim": 0.8,
                              "categorical": _cat_buckets(
                                  0.4, 0.5, 0.2, 0.2, 1.0, 0.7, 5.0)}] * 7)
    _populate_experiment(results_dir, "I2", base)
    _populate_experiment(results_dir, "I3", cand)
    alias_to_result, alias_to_manifest = load_results(results_dir)
    pa = paired_event_differences(
        alias_to_result["I2"], alias_to_result["I3"],
        metric="MAE_event", threshold=None,
        n_bootstrap=2000, seed=42,
    )
    # baseline - candidate = 10 - 5 = +5
    assert pa["mean"] == pytest.approx(5.0)
    assert pa["median"] == pytest.approx(5.0)
    # lower-is-better: CI should sit strictly above 0
    assert pa["ci95"][0] > 0.0


def test_metric_direction_higher_is_better():
    """SSIM / CSI / POD / HSS / ACC: positive diff = candidate improvement."""
    base = _make_result_v2(_synth_per_event([
        {"mae": 10.0, "rmse": 10.0, "ssim": 0.5,
         "categorical": _cat_buckets(0.2, 0.4, 0.6, 0.1, 1.5, 0.6, 5.0)}
    ] * 7), flat=True)
    cand = _make_result_v2(_synth_per_event([
        {"mae": 5.0, "rmse": 5.0, "ssim": 0.8,
         "categorical": _cat_buckets(0.4, 0.5, 0.2, 0.2, 1.0, 0.7, 5.0)}
    ] * 7), flat=True)
    pa_ssim = paired_event_differences(base, cand, metric="SSIM_event_mean",
                                        threshold=None, n_bootstrap=2000, seed=42)
    # candidate - baseline = 0.8 - 0.5 = +0.3
    assert pa_ssim["mean"] == pytest.approx(0.3)
    assert pa_ssim["ci95"][0] > 0.0
    pa_csi = paired_event_differences(base, cand, metric="CSI",
                                       threshold=5.0, n_bootstrap=2000, seed=42)
    assert pa_csi["mean"] == pytest.approx(0.2)


def test_metric_direction_bias_min_distance_from_one():
    """BIAS: sign is |BIAS_b - 1| - |BIAS_c - 1|; improvement iff candidate closer to 1."""
    base = _make_result_v2(_synth_per_event([
        {"mae": 10.0, "rmse": 10.0, "ssim": 0.5,
         "categorical": _cat_buckets(0.2, 0.4, 0.6, 0.1, 1.5, 0.6, 5.0)}
    ] * 7), flat=True)
    cand = _make_result_v2(_synth_per_event([
        {"mae": 10.0, "rmse": 10.0, "ssim": 0.5,
         "categorical": _cat_buckets(0.2, 0.4, 0.6, 0.1, 1.0, 0.6, 5.0)}
    ] * 7), flat=True)
    pa = paired_event_differences(base, cand, metric="BIAS",
                                  threshold=5.0, n_bootstrap=2000, seed=42)
    # |1.5 - 1| - |1.0 - 1| = 0.5 - 0.0 = 0.5
    assert pa["mean"] == pytest.approx(0.5)
    # raw_biases is keyed by typhoon_id; each entry is (baseline, candidate).
    assert pa["raw_biases"]["1"] == pytest.approx((1.5, 1.0))


# ---------------------------------------------------------------------------
# 2. Event-level pairing
# ---------------------------------------------------------------------------

def test_event_level_pairing_only_shared_events(tmp_path):
    results_dir = tmp_path / "results"
    # Baseline has 7 events; candidate has only 5 of them (tid=1..5).
    base = _synth_per_event([{"mae": 10.0, "rmse": 10.0, "ssim": 0.5,
                              "categorical": _cat_buckets(
                                  0.2, 0.4, 0.6, 0.1, 1.5, 0.6, 5.0)}] * 7)
    cand = _synth_per_event([{"mae": 5.0, "rmse": 5.0, "ssim": 0.8,
                              "categorical": _cat_buckets(
                                  0.4, 0.5, 0.2, 0.2, 1.0, 0.7, 5.0)}] * 5)
    _populate_experiment(results_dir, "I2", base)
    _populate_experiment(results_dir, "I3", cand)
    alias_to_result, alias_to_manifest = load_results(results_dir)
    pa = paired_event_differences(
        alias_to_result["I2"], alias_to_result["I3"],
        metric="MAE_event", threshold=None,
        n_bootstrap=2000, seed=42,
    )
    # only 5 events are shared; n_pairs = 5, not 7
    assert pa["n_pairs"] == 5
    assert set(pa["per_event"]) == {"1", "2", "3", "4", "5"}


# ---------------------------------------------------------------------------
# 3. NaN event handling
# ---------------------------------------------------------------------------

def test_nan_event_in_candidate_dropped_from_pair(tmp_path):
    results_dir = tmp_path / "results"
    base = _synth_per_event([{"mae": 10.0, "rmse": 10.0, "ssim": 0.5,
                              "categorical": _cat_buckets(
                                  0.2, 0.4, 0.6, 0.1, 1.5, 0.6, 5.0)}] * 7)
    cand = _synth_per_event([{"mae": 5.0, "rmse": 5.0, "ssim": 0.8,
                              "categorical": _cat_buckets(
                                  0.4, 0.5, 0.2, 0.2, 1.0, 0.7, 5.0)}] * 7)
    # Replace event 7's MAE with NaN to test paired_event_differences handling.
    cand["7"]["MAE_event"] = float("nan")
    _populate_experiment(results_dir, "I2", base)
    _populate_experiment(results_dir, "I3", cand)
    alias_to_result, alias_to_manifest = load_results(results_dir)
    pa = paired_event_differences(
        alias_to_result["I2"], alias_to_result["I3"],
        metric="MAE_event", threshold=None,
        n_bootstrap=2000, seed=42,
    )
    assert pa["n_pairs"] == 6
    assert "7" not in pa["per_event"]
    # remaining 6 events still produce mean_diff = 5
    assert pa["mean"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# 4. n_pairs < 4 ⇒ inferential=no
# ---------------------------------------------------------------------------

def test_n_lt_4_inferential_blocked(tmp_path):
    results_dir = tmp_path / "results"
    # Base has 7 events; candidate has only 3 events (IDs 1..3).
    base = _synth_per_event([{"mae": 10.0, "rmse": 10.0, "ssim": 0.5,
                              "categorical": _cat_buckets_all(
                                  0.2, 0.4, 0.6, 0.1, 1.5, 0.6)}] * 7)
    cand_full = _synth_per_event([{"mae": 5.0, "rmse": 5.0, "ssim": 0.8,
                                    "categorical": _cat_buckets_all(
                                        0.4, 0.5, 0.2, 0.2, 1.0, 0.7)}] * 7)
    # Trim cand down to events 1..3 only.
    cand_trim = {k: v for k, v in cand_full.items() if int(k) <= 3}
    # Populate every formal-contrast alias so the analyzer doesn't short-
    # circuit on missing artifacts; only I3 has trimmed per_event to
    # exercise n_pairs<4.
    _populate_experiment(results_dir, "I2", base)
    _populate_experiment(results_dir, "I3", cand_trim)
    _populate_experiment(results_dir, "I4", base)
    _populate_experiment(results_dir, "I5", base)
    _populate_experiment(results_dir, "P0", base)
    _populate_experiment(results_dir, "P1", base)
    _populate_experiment(results_dir, "P2", base)
    _populate_experiment(results_dir, "P3", base)
    alias_to_result, alias_to_manifest = load_results(results_dir)
    payload = analyze(alias_to_result, alias_to_manifest,
                      n_bootstrap=2000, bootstrap_seed=42,
                      include_backbone_sanity=False)
    mae_rows = [r for r in payload["long_rows"]
                if r["axis"] == "AxisI" and r["metric"] == "MAE_event"]
    # I3 has only 3 events; I2/I4/I5 have 7. Contrasts that involve I3
    # produce n_pairs=3 (descriptive-only); contrast (I4-I5) is n_pairs=7.
    low_n = [r for r in mae_rows if r["n_pairs"] < 4]
    high_n = [r for r in mae_rows if r["n_pairs"] >= 4]
    assert low_n, "expected at least one low-n MAE_event AxisI row"
    assert high_n, "expected at least one high-n MAE_event AxisI row"
    for r in low_n:
        assert r["n_pairs"] == 3
        assert r["inferential"] == "no"
    for r in high_n:
        assert r["n_pairs"] == 7
        assert r["inferential"] == "yes"


# ---------------------------------------------------------------------------
# 5. Deterministic bootstrap (seed=42)
# ---------------------------------------------------------------------------

def test_deterministic_bootstrap_seed_42(tmp_path):
    """Same seed MUST give identical CI; we do NOT assert the converse."""
    results_dir = tmp_path / "results"
    base_values = [
        {"mae": x, "rmse": x, "ssim": 0.5,
         "categorical": _cat_buckets(0.2, 0.4, 0.6, 0.1, 1.0, 0.6, 5.0)}
        for x in (10, 9, 11, 8, 12, 7, 9)
    ]
    cand_values = [
        {"mae": y, "rmse": y, "ssim": 0.5,
         "categorical": _cat_buckets(0.3, 0.4, 0.5, 0.2, 1.0, 0.6, 5.0)}
        for y in (5, 6, 4, 7, 3, 8, 4)
    ]
    _populate_experiment(results_dir, "I2", _synth_per_event(base_values))
    _populate_experiment(results_dir, "I3", _synth_per_event(cand_values))
    alias_to_result, _ = load_results(results_dir)
    pa1 = paired_event_differences(
        alias_to_result["I2"], alias_to_result["I3"],
        metric="MAE_event", threshold=None,
        n_bootstrap=10000, seed=42,
    )
    pa2 = paired_event_differences(
        alias_to_result["I2"], alias_to_result["I3"],
        metric="MAE_event", threshold=None,
        n_bootstrap=10000, seed=42,
    )
    # Same seed → identical resample sequence → identical CI.
    assert pa1["ci95"] == pa2["ci95"]
    # CI width must also be sane: hi > lo.
    assert pa1["ci95"][1] > pa1["ci95"][0]


# ---------------------------------------------------------------------------
# 6. Exact two-sided sign-flip
# ---------------------------------------------------------------------------

def test_exact_signflip_two_of_two():
    # Two positives: P(#positives >= 2) under H0 = 1/4; two-sided p = 1/2.
    p, n = exact_two_sided_sign_flip([+1.0, +1.0])
    assert (p, n) == (0.5, 2)


def test_exact_signflip_perfectly_balanced():
    p, n = exact_two_sided_sign_flip([+1.0, -1.0, +1.0, -1.0])
    assert (p, n) == (1.0, 4)


def test_exact_signflip_three_of_three():
    # Three positives: P(#positives >= 3) = 1/8; two-sided p = 1/4.
    p, n = exact_two_sided_sign_flip([+1.0, +1.0, +1.0])
    assert (p, n) == (0.25, 3)


def test_exact_signflip_drops_nans():
    p, n = exact_two_sided_sign_flip([+1.0, float("nan"), -1.0, +1.0])
    assert (p, n) == (1.0, 3)


def test_exact_signflip_empty_returns_nan():
    p, n = exact_two_sided_sign_flip([])
    assert n == 0
    assert p != p  # NaN


# ---------------------------------------------------------------------------
# 7. Holm correction
# ---------------------------------------------------------------------------

def test_holm_correction_classic_example():
    raw = {"a": 0.04, "b": 0.01, "c": 0.20}
    adj = holm_correction(raw)
    # After sorting by raw ascending: b(0.01) -> 3*0.01=0.03;
    # a(0.04) -> max(0.03, 2*0.04)=0.08; c(0.20) -> max(0.08, 0.20)=0.20.
    assert adj["b"] == pytest.approx(0.03)
    assert adj["a"] == pytest.approx(0.08)
    assert adj["c"] == pytest.approx(0.20)


def test_holm_correction_monotone_non_increasing_adjusted():
    raw = {"a": 0.5, "b": 0.4, "c": 0.3, "d": 0.2, "e": 0.1}
    adj = holm_correction(raw)
    keys_by_raw = sorted(raw, key=lambda k: raw[k])
    vals = [adj[k] for k in keys_by_raw]
    for x, y in zip(vals, vals[1:]):
        assert x <= y


# ---------------------------------------------------------------------------
# 8-11. Fingerprint contract failures
# ---------------------------------------------------------------------------

def test_test_status_not_sealed_fails(tmp_path):
    results_dir = tmp_path / "results"
    base = _synth_per_event([{"mae": 1.0, "rmse": 1.0, "ssim": 0.5,
                              "categorical": _cat_buckets(
                                  0.2, 0.4, 0.6, 0.1, 1.0, 0.6, 5.0)}] * 7)
    _populate_experiment(results_dir, "I2", base,
                        manifest_overrides={"test_status": "UNSEALED"})
    _populate_experiment(results_dir, "I3",
                         _synth_per_event([{"mae": 1.0, "rmse": 1.0,
                                            "ssim": 0.5,
                                            "categorical": _cat_buckets(
                                                0.2, 0.4, 0.6, 0.1, 1.0,
                                                0.6, 5.0)}] * 7))
    with pytest.raises(ContractViolation, match="test_status"):
        load_results(results_dir)


def test_split_test_fails(tmp_path):
    results_dir = tmp_path / "results"
    base = _synth_per_event([{"mae": 1.0, "rmse": 1.0, "ssim": 0.5,
                              "categorical": _cat_buckets(
                                  0.2, 0.4, 0.6, 0.1, 1.0, 0.6, 5.0)}] * 7)
    _populate_experiment(results_dir, "I2", base,
                        manifest_overrides={"split": "test"})
    _populate_experiment(results_dir, "I3",
                         _synth_per_event([{"mae": 1.0, "rmse": 1.0,
                                            "ssim": 0.5,
                                            "categorical": _cat_buckets(
                                                0.2, 0.4, 0.6, 0.1, 1.0,
                                                0.6, 5.0)}] * 7))
    with pytest.raises(ContractViolation, match="split"):
        load_results(results_dir)


def test_protocol_wrong_fails(tmp_path):
    results_dir = tmp_path / "results"
    base = _synth_per_event([{"mae": 1.0, "rmse": 1.0, "ssim": 0.5,
                              "categorical": _cat_buckets(
                                  0.2, 0.4, 0.6, 0.1, 1.0, 0.6, 5.0)}] * 7)
    _populate_experiment(results_dir, "I2", base,
                        manifest_overrides={"protocol_id": "evaluation_v1"})
    _populate_experiment(results_dir, "I3",
                         _synth_per_event([{"mae": 1.0, "rmse": 1.0,
                                            "ssim": 0.5,
                                            "categorical": _cat_buckets(
                                                0.2, 0.4, 0.6, 0.1, 1.0,
                                                0.6, 5.0)}] * 7))
    with pytest.raises(ContractViolation, match="protocol_id"):
        load_results(results_dir)


def test_smoke_true_fails(tmp_path):
    results_dir = tmp_path / "results"
    base = _synth_per_event([{"mae": 1.0, "rmse": 1.0, "ssim": 0.5,
                              "categorical": _cat_buckets(
                                  0.2, 0.4, 0.6, 0.1, 1.0, 0.6, 5.0)}] * 7)
    _populate_experiment(results_dir, "I2", base,
                        manifest_overrides={"smoke": True})
    _populate_experiment(results_dir, "I3",
                         _synth_per_event([{"mae": 1.0, "rmse": 1.0,
                                            "ssim": 0.5,
                                            "categorical": _cat_buckets(
                                                0.2, 0.4, 0.6, 0.1, 1.0,
                                                0.6, 5.0)}] * 7))
    with pytest.raises(ContractViolation, match="smoke"):
        load_results(results_dir)


def test_n_events_mismatch_fails(tmp_path):
    results_dir = tmp_path / "results"
    base = _synth_per_event([{"mae": 1.0, "rmse": 1.0, "ssim": 0.5,
                              "categorical": _cat_buckets(
                                  0.2, 0.4, 0.6, 0.1, 1.0, 0.6, 5.0)}] * 7)
    _populate_experiment(results_dir, "I2", base)
    # result_v2.n_events says 5, contract says 7 -> fail.
    _populate_experiment(results_dir, "I3",
                         _synth_per_event([{"mae": 1.0, "rmse": 1.0,
                                            "ssim": 0.5,
                                            "categorical": _cat_buckets(
                                                0.2, 0.4, 0.6, 0.1, 1.0,
                                                0.6, 5.0)}] * 5),
                         n_events=5)
    with pytest.raises(ContractViolation, match="n_events"):
        load_results(results_dir)


# ---------------------------------------------------------------------------
# 12. I5 == P0 same-artifact identity
# ---------------------------------------------------------------------------

def test_i5_p0_identity_match_passes(tmp_path):
    results_dir = tmp_path / "results"
    base = _synth_per_event([{"mae": 1.0, "rmse": 1.0, "ssim": 0.5,
                              "categorical": _cat_buckets(
                                  0.2, 0.4, 0.6, 0.1, 1.0, 0.6, 5.0)}] * 7)
    common_fp = {
        "checkpoint_sha256": "ab" * 32,
        "config_sha256": "cd" * 32,
        "dataset_sha256": "ef" * 32,
        "split_sha256": "12" * 32,
        "normalization_sha256": "34" * 32,
        "git_commit": "f0" * 16,
        "epochs": 20,
        "best_epoch": 17,
    }
    _populate_experiment(results_dir, "I5", base,
                         exp_id="E5_terrain_geometry_seed42",
                         alias_ids=["I5", "P0"],
                         manifest_overrides=common_fp)
    # NO second directory for P0: the alias mapping comes from a single
    # manifest's alias_ids list. The script's analyze step consumes it via
    # the alias_to_manifest map; identity is verified post-load.
    alias_to_result, alias_to_manifest = load_results(results_dir)
    identity = enforce_i5_p0_identity(alias_to_manifest)
    assert identity is not None
    assert identity["experiment_id"] == "E5_terrain_geometry_seed42"


def test_i5_p0_identity_mismatch_fails(tmp_path):
    results_dir = tmp_path / "results"
    base = _synth_per_event([{"mae": 1.0, "rmse": 1.0, "ssim": 0.5,
                              "categorical": _cat_buckets(
                                  0.2, 0.4, 0.6, 0.1, 1.0, 0.6, 5.0)}] * 7)
    # Two distinct canonical directories; each declares its own alias.
    _populate_experiment(results_dir, "I5", base,
                         exp_id="I5_terrain_geometry_seed42",
                         alias_ids=["I5"],
                         manifest_overrides={"best_epoch": 17})
    _populate_experiment(results_dir, "P0", base,
                         exp_id="P1_resconvlstm_smooth_seed42",
                         alias_ids=["P0"],
                         manifest_overrides={"best_epoch": 12})
    alias_to_result, alias_to_manifest = load_results(results_dir)
    with pytest.raises(ContractViolation, match="artifact identity"):
        enforce_i5_p0_identity(alias_to_manifest)


# ---------------------------------------------------------------------------
# 13. No window-level significance (sanity guard)
# ---------------------------------------------------------------------------

def test_window_level_significance_forbidden(tmp_path):
    """If a contrast emits n_pairs > n_events, the analyzer MUST refuse."""
    results_dir = tmp_path / "results"
    base = _synth_per_event([{"mae": 1.0, "rmse": 1.0, "ssim": 0.5,
                              "categorical": _cat_buckets(
                                  0.2, 0.4, 0.6, 0.1, 1.0, 0.6, 5.0)}] * 7)
    cand = _synth_per_event([{"mae": 1.0, "rmse": 1.0, "ssim": 0.5,
                              "categorical": _cat_buckets(
                                  0.2, 0.4, 0.6, 0.1, 1.0, 0.6, 5.0)}] * 7)
    _populate_experiment(results_dir, "I2", base)
    _populate_experiment(results_dir, "I3", cand)
    # Patch paired_event_differences to misreport n_pairs.
    import scripts.analyze_ablation_results as A
    real_ped = A.paired_event_differences
    def fake_ped(b, c, metric, threshold, n_bootstrap, seed):
        r = real_ped(b, c, metric, threshold, n_bootstrap, seed)
        r["n_pairs"] = 8  # > n_events=7
        return r
    A.paired_event_differences = fake_ped
    try:
        alias_to_result, alias_to_manifest = load_results(results_dir)
        with pytest.raises(ContractViolation, match="n_pairs"):
            analyze(alias_to_result, alias_to_manifest,
                    n_bootstrap=1000, bootstrap_seed=42,
                    include_backbone_sanity=False)
    finally:
        A.paired_event_differences = real_ped


# ---------------------------------------------------------------------------
# 14b. Wrapper-format integration (real artifact schema)
# ---------------------------------------------------------------------------

def test_wrapper_format_missing_result_key_fails(tmp_path):
    """A result_v2.json written without the ``{'model': ..., 'result': ...}``
    wrapper (the actual schema produced by ``src.evaluation.reporting.
    write_v2_json``) MUST be rejected with a precise error."""
    results_dir = tmp_path / "results"
    pe = {"1": {"MAE_event": 1.0, "RMSE_event": 1.0, "SSIM_event_mean": 0.5,
                "categorical": _cat_buckets_all(0.2, 0.4, 0.6, 0.1, 1.0, 0.6)}}
    d = results_dir / "I2_resconvlstm_seed42"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(_to_json(_make_manifest(
        exp_id="I2_resconvlstm_seed42", aliases=["I2"])), encoding="utf-8")
    # Legacy / un-wrapped format on disk:
    (d / "result_v2.json").write_text(_to_json(pe), encoding="utf-8")
    with pytest.raises(ContractViolation,
                       match="missing required wrapper key 'result'"):
        load_results(results_dir)


def test_wrapper_format_accepted(tmp_path):
    """The canonical wrapper format ``{model, result}`` is accepted and the
    inner evaluator result is what gets threaded into the analyzer."""
    results_dir = tmp_path / "results"
    _populate_experiment(results_dir, "I2",
                         _synth_per_event([
                             {"mae": 1.0, "rmse": 1.0, "ssim": 0.5,
                              "categorical": _cat_buckets_all(
                                  0.2, 0.4, 0.6, 0.1, 1.0, 0.6)}
                         ] * 7))
    alias_to_result, alias_to_manifest = load_results(results_dir)
    inner = alias_to_result["I2"]
    # Inner result has the v2 evaluator fields at the top level.
    assert inner["protocol_id"] == EXPECTED_PROTOCOL_ID
    assert inner["n_events"] == EXPECTED_N_EVENTS
    assert "per_event" in inner
    assert "5mmh" in inner["per_event"]["1"]["categorical"]


# ---------------------------------------------------------------------------
# 14. Backbone sanity block: I1 - I2 is descriptive-only
# ---------------------------------------------------------------------------

def test_backbone_sanity_excluded_from_holm_correction(tmp_path):
    """Including the backbone block MUST NOT inject rows into the
    statistical_summary Holm families."""
    results_dir = tmp_path / "results"
    # Populate every formal Axis I / Axis II alias so analyze() doesn't
    # short-circuit on missing artifacts, plus the backbone aliases.
    canon = ["I2", "I3", "I4", "I5", "P0", "P1", "P2", "P3",
             "I0", "I1", "B1"]
    synth = _synth_per_event([{"mae": 1.0, "rmse": 1.0, "ssim": 0.5,
                                "categorical": _cat_buckets_all(
                                    0.2, 0.4, 0.6, 0.1, 1.0, 0.6)}] * 7)
    for alias in canon:
        _populate_experiment(results_dir, alias, synth)
    alias_to_result, alias_to_manifest = load_results(results_dir)
    payload = analyze(alias_to_result, alias_to_manifest,
                      n_bootstrap=2000, bootstrap_seed=42,
                      include_backbone_sanity=True)
    # No backbone row should appear in statistical_summary_rows.
    for r in payload["statistical_summary_rows"]:
        assert "BackboneSanity" not in r["contrast"]
    # But backbone rows DO appear in backbone_long_rows.
    assert payload["backbone_long_rows"], \
        "backbone-sanity rows must be emitted when --include-backbone-sanity"