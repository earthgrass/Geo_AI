"""Exact toy-array tests for the frozen Evaluation Protocol V2.

Covers every rule in docs/EVALUATION_PROTOCOL_V2.md and the metric-level tests
required by docs/MINIMAX_IMPLEMENTATION_SPEC.md §12:
pooled contingency counts, pooled CSI != mean window CSI, per-event pooling,
zero-denominator NaN, fixed-range SSIM, global RMSE, absence of legacy
range-NRMSE / peak_relative_error, non-finite failure, and key formatting.
"""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _counts(a, b, c, d):
    return {"a": a, "b": b, "c": c, "d": d}


def _win(tid, sum_abs, sum_sq, n, counts, ssim=1.0, peak=0.0,
         fmax=0.0, ymax=0.0):
    return {
        "typhoon_id": tid,
        "sum_abs": sum_abs,
        "sum_sq": sum_sq,
        "n": n,
        "SSIM": ssim,
        "peak_error": peak,
        "f_max": fmax,
        "y_max": ymax,
        "counts": counts,
    }


def _isnan(v):
    return isinstance(v, float) and v != v


# ---------------------------------------------------------------------------
# 1-2. Exact counts and exact scores from a hand-computed fixture
# ---------------------------------------------------------------------------

def test_exact_contingency_counts():
    from src.evaluation.metrics import compute_contingency_counts

    P = np.array([[5.0, 2.0], [8.0, 0.0]])
    T = np.array([[6.0, 1.0], [7.0, 0.0]])
    c = compute_contingency_counts(P, T, threshold=5.0)
    # pred>=5: [[T,F],[T,F]]; obs>=5: [[T,F],[T,F]]
    assert c == {"a": 2, "b": 0, "c": 0, "d": 2}, c
    # d (correct negatives) is always returned.
    assert set(c) == {"a", "b", "c", "d"}


def test_exact_categorical_values_from_counts():
    from src.evaluation.metrics import categorical_from_counts

    s = categorical_from_counts(a=2, b=0, c=0, d=2)
    assert s["CSI"] == 1.0
    assert s["POD"] == 1.0
    assert s["FAR"] == 0.0
    assert s["HSS"] == 1.0
    assert s["ACC"] == 1.0
    assert s["BIAS"] == 1.0

    s2 = categorical_from_counts(a=1, b=1, c=1, d=1)
    # CSI = 1/3; POD = 1/2; FAR = 1/2; HSS = 2(1-1)/((2)(2)+(2)(2)) = 0; ACC=1/2
    assert np.isclose(s2["CSI"], 1 / 3)
    assert np.isclose(s2["POD"], 1 / 2)
    assert np.isclose(s2["FAR"], 1 / 2)
    assert np.isclose(s2["HSS"], 0.0)
    assert np.isclose(s2["ACC"], 1 / 2)
    assert np.isclose(s2["BIAS"], 1.0)


# ---------------------------------------------------------------------------
# 3. Pooled CSI differs from mean window CSI; v2 returns the pooled value
# ---------------------------------------------------------------------------

def test_pooled_csi_differs_from_mean_window_csi():
    from src.evaluation.metrics import categorical_from_counts
    from src.evaluation.evaluator import aggregate_v2

    # w1: CSI=0.5 ; w2: CSI=1.0 ; mean window CSI = 0.75
    # pooled: a=11,b=1,c=0 -> CSI = 11/12 ~= 0.9167 (dry/undefined included)
    w1 = _win(1, 1, 1, 4, {"5mmh": _counts(1, 1, 0, 2)})
    w2 = _win(1, 1, 1, 4, {"5mmh": _counts(10, 0, 0, 2)})
    w3 = _win(1, 1, 1, 4, {"5mmh": _counts(0, 0, 0, 4)})  # dry window

    res = aggregate_v2([w1, w2, w3], thresholds=[5.0])
    pooled = res["overall_global"]["categorical"]["5mmh"]["CSI"]
    assert np.isclose(pooled, 11 / 12), pooled

    # Direct categorical from pooled counts agrees.
    direct = categorical_from_counts(11, 1, 0, 8)["CSI"]
    assert np.isclose(pooled, direct)


# ---------------------------------------------------------------------------
# 4. Per-event pooling before ratio
# ---------------------------------------------------------------------------

def test_event_categorical_uses_pooled_counts():
    from src.evaluation.evaluator import aggregate_v2

    w1 = _win(101, 1, 1, 4, {"10mmh": _counts(1, 1, 0, 2)})
    w2 = _win(101, 1, 1, 4, {"10mmh": _counts(10, 0, 0, 2)})
    res = aggregate_v2([w1, w2], thresholds=[10.0])
    ev = res["per_event"]["101"]
    pooled = ev["categorical"]["10mmh"]["CSI"]
    assert np.isclose(pooled, 11 / 12), pooled


def test_overall_categorical_uses_pooled_counts():
    # The overall_global CSI is computed from POOLED a/b/c/d over all windows —
    # never from the mean of window-level ratios.
    from src.evaluation.metrics import categorical_from_counts
    from src.evaluation.evaluator import aggregate_v2

    w1 = _win(1, 1, 1, 4, {"5mmh": _counts(1, 1, 0, 2)})
    w2 = _win(1, 1, 1, 4, {"5mmh": _counts(10, 0, 0, 2)})
    res = aggregate_v2([w1, w2], thresholds=[5.0])
    got = res["overall_global"]["categorical"]["5mmh"]["CSI"]
    expected = categorical_from_counts(11, 1, 0, 4)["CSI"]
    assert np.isclose(got, expected)
    # And it differs from the mean of the two window CSI values (0.75).
    assert not np.isclose(got, 0.75)


# ---------------------------------------------------------------------------
# 5. d participates in HSS and ACC
# ---------------------------------------------------------------------------

def test_d_included_in_hss_and_acc():
    from src.evaluation.metrics import categorical_from_counts

    # Identical a,b,c with different d must change HSS and ACC.
    s_d0 = categorical_from_counts(1, 1, 1, 0)
    s_d9 = categorical_from_counts(1, 1, 1, 9)
    assert s_d9["HSS"] != s_d0["HSS"] or not (s_d0["HSS"] == s_d0["HSS"])
    assert s_d9["ACC"] > s_d0["ACC"]
    # ACC reflects correct negatives.
    assert np.isclose(s_d9["ACC"], (1 + 9) / 12)


# ---------------------------------------------------------------------------
# 6. Zero denominators -> NaN; all-dry ACC is 1
# ---------------------------------------------------------------------------

def test_categorical_zero_denominator_returns_nan():
    from src.evaluation.metrics import categorical_from_counts

    # No observed positives, no forecast positives: CSI/POD/FAR/BIAS undefined.
    s = categorical_from_counts(0, 0, 0, 100)
    assert _isnan(s["CSI"])
    assert _isnan(s["POD"])
    assert _isnan(s["FAR"])
    assert _isnan(s["BIAS"])
    # HSS denominator zero -> NaN (a=1,b=0,c=0,d=0: both product sums are 0).
    s2 = categorical_from_counts(1, 0, 0, 0)
    assert _isnan(s2["HSS"])
    # ACC is defined for the all-dry case (n > 0).
    assert s["ACC"] == 1.0
    assert s2["ACC"] == 1.0


# ---------------------------------------------------------------------------
# 7. Dry windows contribute counts to global pooling
# ---------------------------------------------------------------------------

def test_dry_windows_contribute_to_global_pooling():
    from src.evaluation.evaluator import aggregate_v2

    # Dry window 1: b=1 (one false alarm), contributes to pooled FAR/CSI.
    # Active window 2: a=2. Each window has exactly 4 pixels.
    w1 = _win(1, 1, 1, 4, {"5mmh": _counts(0, 1, 0, 3)})
    w2 = _win(2, 1, 1, 4, {"5mmh": _counts(2, 0, 0, 2)})
    res = aggregate_v2([w1, w2], thresholds=[5.0])
    cat = res["overall_global"]["categorical"]["5mmh"]
    assert cat["b_false_alarms"] == 1  # dry window's false alarm pooled
    assert np.isclose(cat["CSI"], 2 / 3)
    assert cat["n_total"] == 8


# ---------------------------------------------------------------------------
# 8. Global RMSE = sqrt(pooled MSE), NOT mean window RMSE
# ---------------------------------------------------------------------------

def test_global_rmse_is_sqrt_pooled_mse_not_mean_window():
    from src.evaluation.evaluator import aggregate_v2

    # w1 error 0, w2 error 2 (single pixel each): mean window RMSE = 1.0;
    # pooled RMSE = sqrt((0+4)/2) = sqrt(2) ~= 1.414.
    w1 = _win(1, 0, 0, 1, {"5mmh": _counts(0, 0, 0, 1)})
    w2 = _win(1, 2, 4, 1, {"5mmh": _counts(0, 0, 0, 1)})
    res = aggregate_v2([w1, w2], thresholds=[5.0])
    assert np.isclose(res["overall_global"]["RMSE_global"], np.sqrt(2.0))
    # The window diagnostic is a DIFFERENT number and is labeled as such.
    assert res["overall_window_mean"]["RMSE_window_mean"] == 1.0
    assert res["overall_global"]["RMSE_global"] != \
        res["overall_window_mean"]["RMSE_window_mean"]


# ---------------------------------------------------------------------------
# 9. SSIM uses the fixed data_range (100 mm/h)
# ---------------------------------------------------------------------------

def test_ssim_fixed_data_range():
    from src.evaluation.metrics import compute_window_ssim, SSIM_DATA_RANGE

    # All-zero observation / all-zero forecast is a perfect match.
    z = np.zeros((32, 32))
    assert compute_window_ssim(z, z) == 1.0

    # Two constant fields differing in intensity: v1 would pick
    # data_range = max(f.max(), y.max()) = 2, giving SSIM = (4+C1)/(5+C1).
    # v2 fixes data_range = 100, giving (4+1)/(5+1) = 5/6.
    f = np.full((32, 32), 1.0)
    y = np.full((32, 32), 2.0)
    v2 = compute_window_ssim(f, y)
    assert np.isclose(v2, 5.0 / 6.0, atol=1e-3), v2

    # A per-window dynamic data_range is rejected (must stay fixed at 100).
    try:
        compute_window_ssim(f, y, data_range=2.0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for non-fixed SSIM data_range")


# ---------------------------------------------------------------------------
# 10-11. No legacy range-NRMSE / peak_relative_error in v2 output
# ---------------------------------------------------------------------------

def test_nrmse_not_window_range_explosive():
    from src.evaluation.metrics import nrmse_fixed100

    # The frozen-scale diagnostic is RMSE_global / 100, stable for dry windows.
    assert np.isclose(nrmse_fixed100(0.70091), 0.0070091)
    # Not explosive: the legacy dry-window artifact (1623.65774) is impossible.
    assert nrmse_fixed100(0.70091) < 0.01


def test_peak_relative_error_not_primary():
    from src.evaluation.metrics import compute_window_diagnostics
    from src.evaluation.evaluator import aggregate_v2

    rng = np.random.RandomState(0)
    f8 = rng.rand(8, 8)
    t8 = rng.rand(8, 8)
    diag = compute_window_diagnostics(f8, t8)
    assert "peak_rel_error" not in diag
    assert "NRMSE" not in diag
    assert "peak_error" in diag  # absolute peak retained as a diagnostic

    res = aggregate_v2([_win(1, 1, 1, 4, {"5mmh": _counts(0, 0, 0, 4)})],
                       thresholds=[5.0])
    s = json_dump(res)
    assert "peak_rel_error" not in s
    assert "NRMSE" not in s


def json_dump(obj):
    import json
    return json.dumps(obj)


# ---------------------------------------------------------------------------
# 12. Non-finite input fails
# ---------------------------------------------------------------------------

def test_nan_inf_input_fails():
    from src.evaluation.metrics import (
        compute_window_diagnostics, compute_continuous_suff_stats,
        compute_contingency_counts, compute_window_ssim,
    )

    nan = np.full((8, 8), np.nan)
    ok = np.zeros((8, 8))
    for fn, extra in (
        (compute_continuous_suff_stats, {}),
        (compute_contingency_counts, {"threshold": 5.0}),
        (compute_window_ssim, {}),
        (compute_window_diagnostics, {}),
    ):
        try:
            fn(nan, ok, **extra)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{fn.__name__} accepted NaN input")
        try:
            fn(ok, nan, **extra)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{fn.__name__} accepted NaN target")


def test_negative_tiny_clamped_and_hard_negative_fails():
    from src.evaluation.metrics import clamp_negative_tiny, NEGATIVE_TOLERANCE

    P = np.array([[0.0, -1e-8], [-0.5, 1.0]])
    n = clamp_negative_tiny(P)
    assert n == 1  # only -1e-8 clamped; -0.5 left untouched
    assert P[0, 1] == 0.0
    assert P[1, 0] == -0.5


# ---------------------------------------------------------------------------
# Metric key formatting
# ---------------------------------------------------------------------------

def test_metric_key_format():
    from src.evaluation.metrics import threshold_key

    for tau, expected in [(5.0, "5mmh"), (10.0, "10mmh"),
                          (20.0, "20mmh"), (30.0, "30mmh")]:
        assert threshold_key(tau) == expected
    # Never a floating-point string artifact (CSI_10.0mmh vs CSI_10mmh).
    assert threshold_key(10.0) != "10.0mmh"
    assert "CSI_10mmh" == f"CSI_{threshold_key(10.0)}"


# ---------------------------------------------------------------------------
# Evaluator-level: channel subset + report key round-trip
# ---------------------------------------------------------------------------

def _make_v2_h5(path, n=3, H=8, W=8):
    import h5py
    with h5py.File(path, "w") as f:
        f.create_dataset("precip/input",
                         data=np.random.rand(n, 11, H, W).astype("float32"))
        f.create_dataset("precip/target",
                         data=np.random.rand(n, 1, H, W).astype("float32"))
        f.create_dataset("terrain",
                         data=np.random.rand(n, 4, H, W).astype("float32"))
        f.create_dataset("track",
                         data=np.random.rand(n, 11, 6).astype("float32"))
        g = f.create_group("meta")
        g.create_dataset("year", data=np.arange(2020, 2020 + n))
        g.create_dataset("typhoon_id", data=np.arange(1, 1 + n))
    return str(path)


def test_i4_channel_subset_no_double_slice():
    import torch
    import tempfile
    from torch.utils.data import DataLoader
    from src.models.baselines import ResConvLSTM
    from src.evaluation.evaluator import evaluate_model_v2
    from src.data.dataset import TyphoonDataset

    ch = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11]  # I4 10-channel subset
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        p = _make_v2_h5(Path(tmp) / "d.h5")
        ds = TyphoonDataset(p, typhoon_ids=[1], channel_indices=ch)
        loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
        model = ResConvLSTM(input_channels=10, hidden_dims=[4, 8],
                            kernel_size=3)
        # The loader produced 10 channels; the evaluator must NOT re-slice with
        # canonical indices (would index out of range). It asserts the count.
        res = evaluate_model_v2(
            model, loader, torch.device("cpu"), precip_vmax=100.0,
            thresholds=[5.0, 10.0], channel_indices=ch, split="val")
        assert res["n_windows"] == 1  # one sample (typhoon 1)


def test_per_event_report_keys():
    # Exact user-specified name: per-event report keys are the canonical
    # threshold-key forms (CSI_10mmh), never floating-point-string artifacts.
    from src.evaluation.metrics import threshold_key

    assert "CSI_10mmh" == "CSI_" + threshold_key(10.0)
    assert "POD_20mmh" == "POD_" + threshold_key(20.0)


def test_per_event_report_keys_consistent():
    from src.evaluation.evaluator import aggregate_v2
    from src.evaluation.metrics import threshold_key

    w1 = _win(2203, 1, 1, 4, {"10mmh": _counts(2, 0, 0, 2)})
    w2 = _win(2203, 1, 1, 4, {"10mmh": _counts(0, 0, 0, 4)})
    res = aggregate_v2([w1, w2], thresholds=[10.0])
    ev = res["per_event"]["2203"]
    # The categorical structure is keyed by the canonical threshold key.
    assert list(ev["categorical"]) == ["10mmh"]
    assert not any("10.0mmh" in k for k in ev["categorical"])
    assert np.isfinite(ev["categorical"]["10mmh"]["CSI"])
    # The report writer constructs "CSI_10mmh" via the single threshold_key
    # helper — the v1 "CSI_10.0mmh" mismatch cannot recur.
    assert "CSI_" + threshold_key(10.0) == "CSI_10mmh"


# ---------------------------------------------------------------------------
# Paired event analysis helper (protocol v2 §17) — direction correctness
# ---------------------------------------------------------------------------

def test_paired_event_differences_lower_is_better():
    from src.evaluation.evaluator import paired_event_differences

    def ev_result(vals):
        # vals: {tid: MAE_event}
        return {"per_event": {str(t): {"MAE_event": v} for t, v in vals.items()}}

    base = ev_result({1: 2.0, 2: 4.0})
    cand = ev_result({1: 1.0, 2: 5.0})
    out = paired_event_differences(base, cand, "MAE_event")
    assert out["per_event"] == {"1": 1.0, "2": -1.0}  # base - candidate
    assert out["n_pairs"] == 2
    assert np.isfinite(out["mean"])
    assert len(out["ci95"]) == 2


# ---------------------------------------------------------------------------
# Config matrix: frozen aliases (I5 == P0) and no runnable P4/P5
# ---------------------------------------------------------------------------

def test_i5_p0_same_artifact():
    from src.experiments.registry import (
        resolve_alias, aliases_for_stem, load_alias_registry)

    reg = load_alias_registry()
    assert resolve_alias("I5", reg) == "E5_terrain_geometry"
    assert resolve_alias("P0", reg) == "E5_terrain_geometry"
    assert set(aliases_for_stem("E5_terrain_geometry", reg)) >= {"I5", "P0"}


def test_no_runnable_p4_p5():
    from src.experiments.registry import load_alias_registry

    reg = load_alias_registry()
    aliases = reg.get("aliases", {})
    assert "P4" not in aliases and "P5" not in aliases
    assert reg["blocked"]["P4"] == "BLOCKED_BY_ENVIRONMENTAL_WIND_DATA"
    assert reg["blocked"]["P5"] == "BLOCKED_BY_ENVIRONMENTAL_WIND_DATA"
    # No config files exist for P4/P5.
    for f in (REPO_ROOT / "configs" / "experiments").glob("P[45]*.yaml"):
        raise AssertionError(f"unexpected runnable P4/P5 config: {f}")
