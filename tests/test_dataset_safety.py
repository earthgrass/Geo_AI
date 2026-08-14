"""Smoke tests for dataset leakage safety and physics-loss cleanup.

Requires torch / numpy / h5py (install with ``pip install -r requirements.txt``
plus pytest). Run from the repo root:

    python -m pytest tests/test_dataset_safety.py -v
    # or
    python tests/test_dataset_safety.py

The tests use a tiny synthetic HDF5 fixture (4x4 grid) so they run in seconds
without the multi-GB production dataset.
"""

import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _make_h5(path: Path, n: int, with_meta: bool = True,
             years=None, tids=None) -> str:
    import numpy as np
    import h5py

    H = W = 4
    data = np.random.rand(n, 12, 12, H, W).astype("float32")
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=data)
        if with_meta:
            g = f.create_group("meta")
            g.create_dataset("year", data=np.asarray(years or [2014] * n))
            g.create_dataset("typhoon_id", data=np.asarray(tids or list(range(n))))
    return str(path)


def test_metadata_filtering_works():
    import numpy as np
    from src.data.dataset import TyphoonDataset

    years = np.array([2014] * 5 + [2023] * 10 + [2024] * 5)
    tids = np.array([1] * 5 + [2] * 10 + [3] * 5)
    with tempfile.TemporaryDirectory() as tmp:
        path = _make_h5(Path(tmp) / "d.h5", n=20, years=years, tids=tids)
        ds = TyphoonDataset(path, split_years=(2023, 2023))
        assert len(ds) == 10, f"expected 10, got {len(ds)}"
        assert set(ds.get_typhoon_ids()) == {2}


def test_missing_metadata_raises():
    from src.data.dataset import TyphoonDataset

    with tempfile.TemporaryDirectory() as tmp:
        path = _make_h5(Path(tmp) / "d.h5", n=20, with_meta=False)
        try:
            TyphoonDataset(path, split_years=(2023, 2023))
        except RuntimeError:
            return
        raise AssertionError("expected RuntimeError when metadata is missing")


def test_event_split_overlap_raises():
    from src.data.splits import assert_disjoint_event_split

    try:
        assert_disjoint_event_split({1, 2}, {2, 3}, {4})
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError for overlapping event split")


def test_physics_oro_requires_wind_channels():
    from src.training.physics_loss import PhysicsInformedLoss

    # Enabled but no wind channels -> must raise.
    try:
        PhysicsInformedLoss(oro_config={"enabled": True,
                                        "u_channel": None,
                                        "v_channel": None})
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError for enabled oro without wind channels")

    # Enabled with valid channels -> constructs fine.
    PhysicsInformedLoss(oro_config={"enabled": True,
                                    "u_channel": 0,
                                    "v_channel": 1,
                                    "dh_dx_channel": 9,
                                    "dh_dy_channel": 10})


def test_nonneg_removed():
    from src.training.physics_loss import PhysicsInformedLoss

    components = PhysicsInformedLoss().components
    assert "nonneg" not in components, f"nonneg still in {components}"
    assert "rain" in components and "extreme" in components


def test_mse_and_extreme_independent():
    import torch
    from src.training.physics_loss import PhysicsInformedLoss

    P_hat = torch.tensor([[[[0.0, 5.0], [20.0, 0.0]]]], dtype=torch.float32)
    P_true = torch.tensor([[[[0.0, 4.0], [30.0, 0.0]]]], dtype=torch.float32)

    # Base (rain) only -> unweighted MSE.
    loss_rain = PhysicsInformedLoss(components=["rain"])
    out_rain = loss_rain(P_hat, P_true, {"P_prev": P_true})
    expected_mse = ((P_hat - P_true) ** 2).mean()
    assert torch.allclose(out_rain["rain"], expected_mse, atol=1e-6)
    assert set(out_rain.keys()) == {"rain", "total"}

    # Extreme only -> restricted to pixels > threshold (10 mm/h: only the 30 and 20
    # pixels qualify, so exactly two pixels contribute).
    loss_ext = PhysicsInformedLoss(components=["extreme"], extreme_threshold=10.0)
    out_ext = loss_ext(P_hat, P_true, {"P_prev": P_true})
    # Mask covers P_true > 10: positions (0,0) 0 (no), (0,1) 4 (no),
    # (1,0) 30 (yes), (1,1) 0 (no). Only the 30 pixel; but P_hat=20 there too.
    # Expected: mean over that one pixel = (20-30)^2 = 100.
    assert torch.allclose(out_ext["extreme"], torch.tensor(100.0), atol=1e-4)
    assert set(out_ext.keys()) == {"extreme", "total"}


def main() -> None:
    try:
        import numpy  # noqa: F401
        import h5py  # noqa: F401
        import torch  # noqa: F401
    except ImportError as exc:
        print(f"SKIP: required dependency not installed ({exc}). "
              f"Run on a machine with torch/numpy/h5py.")
        sys.exit(0)

    tests = [
        ("test_metadata_filtering_works", test_metadata_filtering_works),
        ("test_missing_metadata_raises", test_missing_metadata_raises),
        ("test_event_split_overlap_raises", test_event_split_overlap_raises),
        ("test_physics_oro_requires_wind_channels", test_physics_oro_requires_wind_channels),
        ("test_nonneg_removed", test_nonneg_removed),
        ("test_mse_and_extreme_independent", test_mse_and_extreme_independent),
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
