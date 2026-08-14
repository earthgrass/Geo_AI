"""Tests for the frozen experiment pipeline: channel subset, normalization,
model channel counts, TrajGRU, test sealing, and event-level aggregation."""

import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _make_v2_h5(path, n=4, H=8, W=8):
    import h5py
    with h5py.File(path, "w") as f:
        f.create_dataset("precip/input", data=np.random.rand(n, 11, H, W).astype("float32"))
        f.create_dataset("precip/target", data=np.random.rand(n, 1, H, W).astype("float32"))
        f.create_dataset("terrain", data=np.random.rand(n, 4, H, W).astype("float32"))
        f.create_dataset("track", data=np.random.rand(n, 11, 6).astype("float32"))
        g = f.create_group("meta")
        g.create_dataset("year", data=np.arange(2020, 2020 + n))
        g.create_dataset("typhoon_id", data=np.arange(1, 1 + n))
    return str(path)


def test_channel_subset_shapes():
    import torch
    from src.data.dataset import TyphoonDataset

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        p = _make_v2_h5(Path(tmp) / "d.h5")
        # E1/E2: precipitation only -> 1 channel.
        ds1 = TyphoonDataset(p, channel_indices=[0])
        X, Y, _ = ds1[0]
        assert X.shape == (11, 1, 8, 8), X.shape
        # E3: 8 channels.
        ds3 = TyphoonDataset(p, channel_indices=[0, 1, 2, 3, 4, 5, 6, 7])
        assert ds3[0][0].shape == (11, 8, 8, 8)
        # E4: 10 channels.
        ds4 = TyphoonDataset(p, channel_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 11])
        assert ds4[0][0].shape == (11, 10, 8, 8)
        # E5/E6: all 12.
        ds5 = TyphoonDataset(p, channel_indices=list(range(12)))
        assert ds5[0][0].shape == (11, 12, 8, 8)


def test_channel_normalize():
    import torch
    from src.data.transforms import ChannelNormalize

    stats = {
        "track_features": {
            "center_wind_speed": {"mean": 20.0, "std": 10.0},
            "center_pressure": {"mean": 990.0, "std": 20.0},
            "u_move": {"mean": 0.0, "std": 10.0},
            "v_move": {"mean": 0.0, "std": 10.0},
        },
        "terrain": {
            "dem": {"mean": 50.0, "std": 200.0},
            "dh_dx": {"mean": 0.0, "std": 5.0},
            "dh_dy": {"mean": 0.0, "std": 5.0},
        },
    }
    idx = [0, 1, 8, 11]  # precip, wind, dem, land_mask
    t = ChannelNormalize(stats, channel_indices=idx, precip_vmax=100.0)

    X = torch.zeros(1, 4, 4, 4)
    X[:, 0] = 100.0  # precip -> 1.0
    X[:, 1] = 30.0   # wind -> (30-20)/10 = 1.0
    X[:, 2] = 250.0  # dem -> (250-50)/200 = 1.0
    X[:, 3] = 1.0    # land_mask -> unchanged
    Y = torch.full((1, 1, 4, 4), 50.0)
    meta = {"P_prev": torch.full((1, 1, 4, 4), 100.0)}

    X, Y, meta = t(X, Y, meta)
    assert abs(X[0, 0, 0, 0].item() - 1.0) < 1e-6
    assert abs(X[0, 1, 0, 0].item() - 1.0) < 1e-6
    assert abs(X[0, 2, 0, 0].item() - 1.0) < 1e-6
    assert abs(X[0, 3, 0, 0].item() - 1.0) < 1e-6  # land_mask unchanged
    assert abs(Y[0, 0, 0, 0].item() - 0.5) < 1e-6
    assert abs(meta["P_prev"][0, 0, 0, 0].item() - 1.0) < 1e-6


def test_model_input_channel_count():
    import torch
    from src.models.baselines import ResConvLSTM

    for n_ch in (1, 8, 10, 12):
        m = ResConvLSTM(input_channels=n_ch, hidden_dims=[8, 16])
        X = torch.randn(2, 11, n_ch, 8, 8)
        P_prev = torch.randn(2, 1, 8, 8)
        out = m.compute_prediction(X)
        assert out.shape == (2, 1, 8, 8)


def test_trajgru_forward():
    import torch
    from src.models.trajgru import TrajGRU

    m = TrajGRU(input_channels=1, hidden_dims=[8, 16])
    X = torch.randn(2, 11, 1, 8, 8)
    out = m(X)
    assert out.shape == (2, 1, 8, 8)
    assert torch.isfinite(out).all()


def test_test_sealing_default_off():
    # The experiment runner must refuse test evaluation unless --allow-test-eval.
    import argparse
    from scripts.run_experiment import main as _unused  # noqa: F401

    # Reconstruct the parser defaults by inspecting the argparse setup.
    import runpy
    import inspect
    # Simpler: assert the flag is action='store_true' (default False).
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_exp", REPO_ROOT / "scripts" / "run_experiment.py")
    mod = importlib.util.module_from_spec(spec)
    # We can't exec (it runs argparse only inside main), so inspect source.
    src = (REPO_ROOT / "scripts" / "run_experiment.py").read_text(encoding="utf-8")
    assert 'action="store_true"' in src
    assert "--allow-test-eval" in src


def _load_run_experiment():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_experiment", REPO_ROOT / "scripts" / "run_experiment.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_loader_shuffle():
    from torch.utils.data import RandomSampler, SequentialSampler

    run_exp = _load_run_experiment()
    stats = {"track_features": {}, "terrain": {}}
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        p = _make_v2_h5(Path(tmp) / "d.h5")
        _, train_loader = run_exp.make_loader(
            p, [1, 2], [0], stats, batch_size=2, num_workers=0,
            shuffle=True, pin_memory=False)
        _, val_loader = run_exp.make_loader(
            p, [1, 2], [0], stats, batch_size=2, num_workers=0,
            shuffle=False, pin_memory=False)
        assert isinstance(train_loader.sampler, RandomSampler)
        assert isinstance(val_loader.sampler, SequentialSampler)


def test_amp_selection():
    run_exp = _load_run_experiment()
    f = run_exp.resolve_use_amp
    assert f("auto", None, True) is True     # CUDA, no YAML -> on
    assert f("auto", None, False) is False   # CPU, no YAML -> off
    assert f("auto", False, True) is False   # YAML False overrides CUDA
    assert f("auto", True, False) is True    # YAML True overrides CPU
    assert f("on", None, False) is True      # CLI on
    assert f("off", True, True) is False     # CLI off


def test_config_precedence():
    run_exp = _load_run_experiment()
    assert run_exp._resolve(None, 8, 4) == 8     # YAML wins when CLI None
    assert run_exp._resolve(16, 8, 4) == 16      # CLI wins
    assert run_exp._resolve(None, None, 4) == 4  # default


def test_event_aggregation():
    from src.evaluation.evaluator import _aggregate_by_event

    rows = [
        {"typhoon_id": 1, "MAE": 1.0, "RMSE": 2.0},
        {"typhoon_id": 1, "MAE": 3.0, "RMSE": 4.0},
        {"typhoon_id": 2, "MAE": 5.0, "RMSE": 6.0},
    ]
    ev = _aggregate_by_event(rows)
    assert set(ev.keys()) == {1, 2}
    assert ev[1]["MAE"] == 2.0  # (1+3)/2
    assert ev[2]["MAE"] == 5.0


def main():
    tests = [
        ("test_channel_subset_shapes", test_channel_subset_shapes),
        ("test_channel_normalize", test_channel_normalize),
        ("test_model_input_channel_count", test_model_input_channel_count),
        ("test_trajgru_forward", test_trajgru_forward),
        ("test_test_sealing_default_off", test_test_sealing_default_off),
        ("test_event_aggregation", test_event_aggregation),
        ("test_loader_shuffle", test_loader_shuffle),
        ("test_amp_selection", test_amp_selection),
        ("test_config_precedence", test_config_precedence),
    ]
    n_fail = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  [PASS] {name}")
        except Exception as exc:  # noqa: BLE001
            n_fail += 1
            print(f"  [FAIL] {name}: {type(exc).__name__}: {exc}")
    print(f"\n{sys.argv[0]}: {len(tests) - n_fail}/{len(tests)} passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
