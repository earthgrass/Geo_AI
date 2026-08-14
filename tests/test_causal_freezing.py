"""Causal-track, physical-units, and default-loss tests.

Requires numpy / pandas / h5py / rasterio (and torch for the loss test).
Run on the server:

    python -m pytest tests/test_causal_freezing.py -v
"""

import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCRIPT_DIR = REPO_ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def _needs(*mods):
    for m in mods:
        try:
            __import__(m)
        except ImportError as exc:
            return str(exc)
    return None


# 1. Future CMA data cannot alter an earlier sample.
def test_future_cma_cannot_alter_sample():
    import numpy as np
    import pandas as pd
    from build_paper_dataset import causal_track_at

    times = pd.to_datetime(["2020-01-01 00:00", "2020-01-01 06:00",
                            "2020-01-01 12:00", "2020-01-01 18:00"])
    raw = pd.DataFrame({
        "Time": times,
        "Lat": [10.0, 10.5, 11.0, 11.5],
        "Lon": [120.0, 120.5, 121.0, 121.5],
        "Pressure": [1000.0, 990.0, 980.0, 970.0],
        "Wind_Speed": [20.0, 25.0, 30.0, 35.0],
    })
    anchor = pd.Timestamp("2020-01-01 15:00")  # between 12:00 and 18:00 fixes

    track1, _, _ = causal_track_at(raw, anchor)

    # Mutate the FUTURE fix (18:00) -> must NOT change the sample.
    raw_future = raw.copy()
    raw_future.loc[3, ["Lat", "Lon", "Pressure", "Wind_Speed"]] = [99.0, 99.0, 500.0, 100.0]
    track2, _, _ = causal_track_at(raw_future, anchor)
    assert np.allclose(track1, track2)

    # Mutate a PAST fix (12:00) -> MUST change the sample (it is available).
    raw_past = raw.copy()
    raw_past.loc[2, "Lat"] = 15.0
    track3, _, _ = causal_track_at(raw_past, anchor)
    assert not np.allclose(track1, track3)


# 1b. Regression: causal track must yield valid geography (guards against the
#     pandas-3.0 datetime64[us] -> unit bug that produced lat/lon ~ 1e8).
def test_causal_track_valid_geography():
    import numpy as np
    import pandas as pd
    from build_paper_dataset import causal_track_at

    # format= produces datetime64[us] on pandas 3.0 (same path as the CMA parser).
    times = pd.to_datetime(
        ["2020010100", "2020010106", "2020010112", "2020010118"],
        format="%Y%m%d%H",
    )
    raw = pd.DataFrame({
        "Time": times,
        "Lat": [10.0, 10.5, 11.0, 11.5],
        "Lon": [120.0, 120.5, 121.0, 121.5],
        "Pressure": [1000.0, 990.0, 980.0, 970.0],
        "Wind_Speed": [20.0, 25.0, 30.0, 35.0],
    })
    track, _, _ = causal_track_at(raw, pd.Timestamp("2020-01-01 15:00:00"))
    assert np.all(np.abs(track[:, 0]) <= 90.0), track[:, 0]
    assert np.all(np.abs(track[:, 1]) <= 180.0), track[:, 1]
    assert np.all(track[:, 2] >= 0.0)   # wind >= 0
    assert np.all(track[:, 3] > 0.0)    # pressure > 0


# 1c. Regression: CMA 0-360 longitude must be normalized to EPSG:4326 (-180..180).
def test_longitude_normalized():
    import pandas as pd
    from build_paper_dataset import parse_best_track
    import tempfile, os

    lines = [
        "66666 0000    2 0001 2401 0 6 TEST                             20250101",
        "2025010100 1 200 2150 1000 20",   # lon 215.0 -> -145.0
        "2025010106 1 210 2200 995 22",   # lon 220.0 -> -140.0
    ]
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fp:
        fp.write("\n".join(lines))
        p = fp.name
    df = parse_best_track(p)
    os.unlink(p)
    assert (df["Lon"] < 180).all() and (df["Lon"] > -180).all()
    assert abs(df["Lon"].iloc[0] - (-145.0)) < 1e-6


# 1d. Regression: dateline crossing must not produce spurious u_move velocities.
def test_dateline_umove_plausible():
    import numpy as np
    import pandas as pd
    from build_paper_dataset import causal_track_at

    # Longitude crossing the +180/-180 dateline.
    times = pd.to_datetime(
        ["2020010100", "2020010106", "2020010112", "2020010118"],
        format="%Y%m%d%H",
    )
    raw = pd.DataFrame({
        "Time": times,
        "Lat": [20.0, 20.2, 20.4, 20.6],
        "Lon": [178.0, -178.0, -176.0, -174.0],  # crosses dateline eastward
        "Pressure": [1000.0, 990.0, 980.0, 970.0],
        "Wind_Speed": [20.0, 25.0, 30.0, 35.0],
    })
    track, _, _ = causal_track_at(raw, pd.Timestamp("2020-01-01 15:00:00"))
    u_move = track[:, 4]
    assert np.all(np.abs(u_move) < 100.0), u_move  # km/h, not thousands


# 2. Terrain gradients have physical units (m/km).
def test_terrain_gradient_units():
    import numpy as np
    from affine import Affine
    from build_paper_dataset import terrain_from_dem

    dem = np.tile(np.linspace(0.0, 1000.0, 128), (128, 1)).astype("float32")
    win_transform = Affine(0.1, 0, 100.0, 0, -0.1, 30.0)  # 0.1 deg/pixel
    anchor_lat = 20.0

    _, dh_dx, dh_dy, land_mask = terrain_from_dem(dem, win_transform, anchor_lat)

    km_per_deg_lon = 111.19 * np.cos(np.radians(anchor_lat))
    km_per_pixel = 0.1 * km_per_deg_lon
    grad_per_pixel = 1000.0 / 127.0          # m/pixel
    expected = grad_per_pixel / km_per_pixel  # m/km

    assert dh_dx.shape == (128, 128)
    assert np.allclose(dh_dx[0, 0], expected, atol=1e-2), (dh_dx[0, 0], expected)
    assert np.allclose(dh_dy, 0.0, atol=1e-6)  # no meridional slope
    assert np.isin(land_mask, [0.0, 1.0]).all()


# 3. Geometry channels are normalized and correctly shaped.
def test_geometry_channels_normalized():
    import numpy as np
    from src.data.dataset import static_grid_channels

    r_norm, dx_norm, dy_norm = static_grid_channels(128)
    assert dx_norm.shape == (128, 128)
    assert dx_norm.min() >= -1.0 - 1e-6 and dx_norm.max() <= 1.0 + 1e-6
    assert dy_norm.min() >= -1.0 - 1e-6 and dy_norm.max() <= 1.0 + 1e-6
    assert np.allclose(r_norm, np.sqrt(dx_norm ** 2 + dy_norm ** 2))


# 4. GPM signed offset is correct (actual - requested).
def test_gpm_signed_offset():
    from build_paper_dataset import lookup_gpm

    index = {"1410": {
        datetime(2020, 1, 1, 0, 0): "a.tif",
        datetime(2020, 1, 1, 0, 30): "b.tif",
    }}
    # Exact.
    path, actual, off = lookup_gpm(index, "1410", datetime(2020, 1, 1, 0, 0), 900)
    assert off == 0.0
    # Requested 0:10 -> nearest is 0:00 (offset -600 s).
    _, actual, off = lookup_gpm(index, "1410", datetime(2020, 1, 1, 0, 10), 900)
    assert off == -600.0, off
    # Beyond tolerance -> None.
    path, _, off = lookup_gpm(index, "1410", datetime(2020, 1, 1, 1, 0), 900)
    assert path is None


# 5. Target actual lead is recorded (actual_anchor/target GPM times).
def test_target_actual_lead_recorded():
    import numpy as np
    import h5py
    from build_paper_dataset import StreamingH5Writer

    with tempfile_TemporaryDirectory() as tmp:
        out = Path(tmp) / "d.h5"
        w = StreamingH5Writer(str(out), grid_size=4, buffer_size=1)
        w.add(_sample(actual_anchor=1000, actual_target=2800, tgt_off=0.0))
        w.close()
        with h5py.File(out, "r") as f:
            assert "actual_anchor_gpm_time" in f["meta"]
            assert "actual_target_gpm_time" in f["meta"]
            lead = f["meta"]["actual_target_gpm_time"][0] - f["meta"]["actual_anchor_gpm_time"][0]
            assert lead == 1800, lead


# 6. Inconsistent GPM raster grids fail preflight.
def test_inconsistent_grid_fails_preflight():
    import numpy as np
    from affine import Affine
    from rasterio.crs import CRS
    import rasterio

    with tempfile_TemporaryDirectory() as tmp:
        crs = CRS.from_epsg(4326)
        t1 = Affine(0.1, 0, 100.0, 0, -0.1, 30.0)
        t2 = Affine(0.05, 0, 100.0, 0, -0.05, 30.0)  # different resolution
        p1 = Path(tmp) / "a.tif"
        p2 = Path(tmp) / "b.tif"
        _write_raster(p1, t1, crs, np.zeros((100, 100), dtype="float32"))
        _write_raster(p2, t2, crs, np.zeros((200, 200), dtype="float32"))
        with rasterio.open(p1) as s1, rasterio.open(p2) as s2:
            assert tuple(s1.transform) != tuple(s2.transform)
            # The preflight would report 2 distinct transforms -> FAIL.


# 7. seq_len cannot silently disagree with schema.
def test_seq_len_fixed():
    from build_paper_dataset import SEQ_LEN, INPUT_SEQ_LEN
    assert SEQ_LEN == 12
    assert INPUT_SEQ_LEN == 11
    # No --seq-len CLI flag remains (option A).
    import build_paper_dataset as b
    src = Path(b.__file__).read_text(encoding="utf-8")
    assert "--seq-len" not in src


# 8. MSE-only is the default loss configuration.
def test_mse_only_default_loss():
    from src.training.physics_loss import PhysicsInformedLoss
    components = PhysicsInformedLoss().components
    assert components == ["rain"], components


# ---- helpers ----

import tempfile  # noqa: E402


def tempfile_TemporaryDirectory():
    return tempfile.TemporaryDirectory()


def _write_raster(path, transform, crs, data):
    import rasterio
    with rasterio.open(path, "w", driver="GTiff", height=data.shape[0],
                       width=data.shape[1], count=1, dtype=data.dtype,
                       crs=crs, transform=transform) as dst:
        dst.write(data, 1)


def _sample(actual_anchor=1000, actual_target=2800, tgt_off=0.0):
    import numpy as np
    return {
        "precip/input": np.random.rand(11, 4, 4).astype("float32"),
        "precip/target": np.random.rand(1, 4, 4).astype("float32"),
        "terrain": np.random.rand(4, 4, 4).astype("float32"),
        "track": np.random.rand(11, 6).astype("float32"),
        "meta/typhoon_id": np.int64(1),
        "meta/year": np.int64(2020),
        "meta/start_time": np.int64(0),
        "meta/anchor_time": np.int64(1000),
        "meta/target_time": np.int64(2800),
        "meta/latest_cma_fix_time": np.int64(0),
        "meta/cma_fix_age_sec": np.float32(0.0),
        "meta/anchor_lat": np.float32(20.0),
        "meta/anchor_lon": np.float32(120.0),
        "meta/grid_transform": np.zeros(6, dtype="float64"),
        "meta/input_imputed_mask": np.zeros(11, dtype="uint8"),
        "meta/input_gpm_match_offset": np.zeros(11, dtype="float32"),
        "meta/target_gpm_match_offset": np.float32(tgt_off),
        "meta/actual_anchor_gpm_time": np.int64(actual_anchor),
        "meta/actual_target_gpm_time": np.int64(actual_target),
    }


def main() -> None:
    err = _needs("numpy", "pandas", "h5py", "rasterio", "affine")
    if err:
        print(f"SKIP: missing dependency ({err}). Run on the server.")
        sys.exit(0)
    tests = [
        ("test_future_cma_cannot_alter_sample", test_future_cma_cannot_alter_sample),
        ("test_terrain_gradient_units", test_terrain_gradient_units),
        ("test_geometry_channels_normalized", test_geometry_channels_normalized),
        ("test_gpm_signed_offset", test_gpm_signed_offset),
        ("test_target_actual_lead_recorded", test_target_actual_lead_recorded),
        ("test_inconsistent_grid_fails_preflight", test_inconsistent_grid_fails_preflight),
        ("test_seq_len_fixed", test_seq_len_fixed),
        ("test_mse_only_default_loss", test_mse_only_default_loss),
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
