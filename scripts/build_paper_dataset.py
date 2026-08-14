"""Build the schema-v2 paper dataset with a FIXED ANCHOR GRID (no oracle info).

Key corrections over the v1 builder:

1. Fixed anchor grid (P0):
   Every sample uses ONE 128x128 geographic grid anchored at the LAST INPUT
   timestep's typhoon center. All 11 input frames AND the target frame are
   cropped onto this same grid. The target never uses its own future center,
   so no future-track oracle information leaks into the prediction.

2. DEM aligned to GPM (P0):
   DEM is reprojected/resampled with rasterio onto exactly the GPM anchor grid
   (same CRS / bounds / transform / size). Terrain gradients are computed AFTER
   resampling. No assumption that one DEM pixel == 10 km.

3. Missing-value policy (P0):
   Only INPUT precipitation frames may be temporally imputed. The TARGET is
   NEVER imputed — a missing target drops the sample. An input-imputation mask
   and GPM match offset are recorded per sample.

4. Streaming writes (P0):
   Samples are buffered and flushed incrementally into extendable HDF5
   datasets; the full dataset is never held in RAM.

HDF5 schema v2:
    /precip/input   [N, 11, H, W]   float32
    /precip/target  [N, 1,  H, W]   float32
    /terrain        [N, 4,  H, W]   float32  (dem, dh_dx, dh_dy, land_mask)
    /track          [N, 11, 6]      float32  (lat, lon, wind, pressure, u_move, v_move)
    /meta/{typhoon_id, year, start_time, anchor_time, target_time,
           anchor_lat, anchor_lon, grid_transform, input_imputed_mask,
           gpm_match_offset}

Run from repo root:
    python scripts/build_paper_dataset.py \
        --cma-dir CMABSTdata --tif-dir TIFdata --dem Global_DEM.tif \
        --out ConvLSTM_Dataset_128.h5 --buffer-size 256
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import rasterio
from rasterio.windows import Window
from rasterio.warp import reproject, Resampling
from scipy.interpolate import CubicSpline

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import (  # noqa: E402
    CHANNEL_NAMES,
    TRACK_FEATURE_NAMES,
    TERRAIN_CHANNEL_NAMES,
    SCHEMA_VERSION,
    GRID_SIZE,
)

SEQ_LEN = 12          # total frames per sample (11 input + 1 target)
INPUT_SEQ_LEN = 11
N_TRACK = len(TRACK_FEATURE_NAMES)   # 6
PRECIP_UNITS = "mm/h"

_KM_PER_DEG_LAT = 111.19
_TIF_TIME_RE = re.compile(r"(\d{8}-S\d{6})")


# ---------------------------------------------------------------------------
# CMA best-track parsing + interpolation
# ---------------------------------------------------------------------------

def parse_best_track(txt_path: str) -> pd.DataFrame:
    rows = []
    current_id = None
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "66666":
            current_id = parts[4] if len(parts) > 4 else None
            continue
        if current_id is None:
            continue
        try:
            time_str = parts[0]
            lat = float(parts[2]) / 10.0
            lon = float(parts[3]) / 10.0
            pres = float(parts[4])
            wind = float(parts[5])
        except (ValueError, IndexError):
            continue
        rows.append({
            "Typhoon_ID": str(current_id).zfill(4),
            "Time": pd.to_datetime(time_str, format="%Y%m%d%H"),
            "Lat": lat, "Lon": lon, "Pressure": pres, "Wind_Speed": wind,
        })
    return pd.DataFrame(rows)


def interpolate_track(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("Time").reset_index(drop=True)
    if len(group) < 3:
        return pd.DataFrame()

    start = group["Time"].iloc[0]
    hours = (group["Time"] - start).dt.total_seconds().to_numpy() / 3600.0
    new_hours = np.arange(0.0, hours[-1] + 0.5, 0.5)
    new_times = start + pd.to_timedelta(new_hours, unit="h")

    cs = {col: CubicSpline(hours, group[col].to_numpy())
          for col in ("Lat", "Lon", "Pressure", "Wind_Speed")}

    out = pd.DataFrame({
        "Time": new_times,
        "Lat": cs["Lat"](new_hours),
        "Lon": cs["Lon"](new_hours),
        "Pressure": cs["Pressure"](new_hours),
        "Wind_Speed": np.clip(cs["Wind_Speed"](new_hours), a_min=0.0, a_max=None),
    })

    dlat = np.diff(out["Lat"].to_numpy(), prepend=out["Lat"].iloc[0])
    dlon = np.diff(out["Lon"].to_numpy(), prepend=out["Lon"].iloc[0])
    dt_h = 0.5
    out["v_move"] = dlat * _KM_PER_DEG_LAT / dt_h
    out["u_move"] = dlon * _KM_PER_DEG_LAT * np.cos(np.radians(out["Lat"])) / dt_h
    return out


# ---------------------------------------------------------------------------
# GPM index + lookup (pre-indexed, exact-first with nearest fallback)
# ---------------------------------------------------------------------------

def build_tif_index(tif_dir: str) -> dict:
    """Return {typhoon_id: {datetime: tif_path}}."""
    index: dict = {}
    for tif_path in Path(tif_dir).rglob("*.tif"):
        m = _TIF_TIME_RE.search(tif_path.name)
        if not m:
            continue
        ts = datetime.strptime(m.group(1), "%Y%m%d-S%H%M%S")
        tid = tif_path.parent.name[2:6].zfill(4)
        index.setdefault(tid, {})[ts] = str(tif_path)
    return index


def lookup_gpm(index: dict, typhoon_id: str, time: datetime, tolerance_sec: int):
    """Exact-first GPM lookup; nearest fallback within tolerance.

    Returns (path, offset_seconds). offset_seconds == 0 for an exact match,
    >0 for a nearest match, or None if no frame within tolerance.
    """
    frames = index.get(typhoon_id, {})
    if not frames:
        return None, None
    if time in frames:
        return frames[time], 0.0

    best_ts, best_dt = None, None
    for ts in frames:
        dt = abs((ts - time).total_seconds())
        if dt <= tolerance_sec and (best_dt is None or dt < best_dt):
            best_dt, best_ts = dt, ts
    if best_ts is None:
        return None, None
    return frames[best_ts], float(best_dt)


# ---------------------------------------------------------------------------
# Anchor grid + DEM resampling (geospatial alignment)
# ---------------------------------------------------------------------------

def compute_anchor_grid(gpm_transform, gpm_crs, anchor_lat: float, anchor_lon: float,
                        grid_size: int = GRID_SIZE):
    """Return the fixed anchor window + its transform, centered on the anchor."""
    col, row = ~gpm_transform * (anchor_lon, anchor_lat)
    col, row = int(np.round(col)), int(np.round(row))
    window = Window(col - grid_size // 2, row - grid_size // 2, grid_size, grid_size)
    win_transform = rasterio.windows.transform(window, gpm_transform)
    return window, win_transform, gpm_crs


def read_precip_window(tif_path: str, window: Window) -> np.ndarray | None:
    try:
        with rasterio.open(tif_path) as src:
            crop = src.read(1, window=window, boundless=True, fill_value=0.0)
    except Exception:
        return None
    crop = np.where(np.isnan(crop), 0.0, crop)
    crop = np.where(crop < 0, 0.0, crop)
    return crop.astype("float32")


def resample_dem_to_grid(dem_path: str, gpm_crs, win_transform,
                         grid_size: int = GRID_SIZE) -> np.ndarray | None:
    """Resample DEM onto exactly the GPM anchor grid (CRS/bounds/transform/size)."""
    if not Path(dem_path).exists():
        return None
    dest = np.zeros((grid_size, grid_size), dtype="float32")
    try:
        with rasterio.open(dem_path) as src:
            reproject(
                source=rasterio.band(src, 1),
                destination=dest,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=win_transform,
                dst_crs=gpm_crs,
                resampling=Resampling.bilinear,
            )
    except Exception:
        return None
    return dest


def terrain_from_dem(dem: np.ndarray) -> tuple:
    dem = np.clip(dem, 0.0, None).astype("float32")
    land_mask = (dem > 0.0).astype("float32")
    dh_dy, dh_dx = np.gradient(dem)
    return dem, dh_dx.astype("float32"), dh_dy.astype("float32"), land_mask


# ---------------------------------------------------------------------------
# Static grid channels (constant, pixel units)
# ---------------------------------------------------------------------------

def static_grid_channels(grid_size: int = GRID_SIZE):
    """Return distance_center, dx, dy, each [H, W] (explicitly broadcast)."""
    y, x = np.meshgrid(np.arange(grid_size), np.arange(grid_size), indexing="ij")
    cx = (grid_size - 1) / 2.0
    dx = (x - cx).astype("float32")           # [H, W]
    dy = (y - cx).astype("float32")           # [H, W]
    distance_center = np.sqrt(dx ** 2 + dy ** 2).astype("float32")  # [H, W]
    assert dx.shape == (grid_size, grid_size)
    assert dy.shape == (grid_size, grid_size)
    assert distance_center.shape == (grid_size, grid_size)
    return distance_center, dx, dy


# ---------------------------------------------------------------------------
# Streaming HDF5 writer
# ---------------------------------------------------------------------------

class StreamingH5Writer:
    """Buffered, incremental HDF5 writer. Never holds the full dataset in RAM."""

    def __init__(self, out_path: str, grid_size: int, buffer_size: int = 256):
        self.f = h5py.File(out_path, "w")
        self.buffer_size = buffer_size
        self._buffer: list = []
        H = W = grid_size

        self._mk("precip/input", (0, INPUT_SEQ_LEN, H, W), (None, INPUT_SEQ_LEN, H, W),
                 "float32", chunks=(1, INPUT_SEQ_LEN, H, W))
        self._mk("precip/target", (0, 1, H, W), (None, 1, H, W), "float32",
                 chunks=(1, 1, H, W))
        self._mk("terrain", (0, 4, H, W), (None, 4, H, W), "float32", chunks=(1, 4, H, W))
        self._mk("track", (0, INPUT_SEQ_LEN, N_TRACK), (None, INPUT_SEQ_LEN, N_TRACK), "float32")

        for name in ("typhoon_id", "year", "start_time", "anchor_time", "target_time"):
            self._mk(f"meta/{name}", (0,), (None,), "int64")
        for name in ("anchor_lat", "anchor_lon"):
            self._mk(f"meta/{name}", (0,), (None,), "float32")
        self._mk("meta/grid_transform", (0, 6), (None, 6), "float64")
        self._mk("meta/input_imputed_mask", (0, INPUT_SEQ_LEN), (None, INPUT_SEQ_LEN), "uint8")
        self._mk("meta/gpm_match_offset", (0, INPUT_SEQ_LEN), (None, INPUT_SEQ_LEN), "float32")

    def _mk(self, name, shape, maxshape, dtype, chunks=True):
        kwargs = {"compression": "gzip"} if chunks else {}
        if chunks is True:
            chunks = (1,) + shape[1:]
        if chunks:
            kwargs["chunks"] = chunks
        self.f.create_dataset(name, shape=shape, maxshape=maxshape, dtype=dtype, **kwargs)

    def add(self, sample: dict) -> None:
        self._buffer.append(sample)
        if len(self._buffer) >= self.buffer_size:
            self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return
        n = len(self._buffer)
        keys = self._buffer[0].keys()
        for key in keys:
            arr = np.stack([s[key] for s in self._buffer], axis=0)
            ds = self.f[key]
            ds.resize(ds.shape[0] + n, axis=0)
            ds[-n:] = arr
        self._buffer = []

    def close(self, created_by: str = "scripts/build_paper_dataset.py") -> int:
        self._flush()
        n = self.f["precip/input"].shape[0]
        self.f.attrs["schema_version"] = SCHEMA_VERSION
        self.f.attrs["channel_names"] = CHANNEL_NAMES
        self.f.attrs["track_feature_names"] = TRACK_FEATURE_NAMES
        self.f.attrs["terrain_channel_names"] = TERRAIN_CHANNEL_NAMES
        self.f.attrs["seq_len"] = SEQ_LEN
        self.f.attrs["input_seq_len"] = INPUT_SEQ_LEN
        self.f.attrs["grid_size"] = GRID_SIZE
        self.f.attrs["precipitation_units"] = PRECIP_UNITS
        self.f.attrs["created_by"] = created_by
        self.f.close()
        return n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build schema-v2 paper dataset.")
    parser.add_argument("--cma-dir", default="CMABSTdata")
    parser.add_argument("--tif-dir", default="TIFdata")
    parser.add_argument("--dem", default="Global_DEM.tif")
    parser.add_argument("--out", default="ConvLSTM_Dataset_128.h5")
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--max-missing", type=int, default=2,
                        help="Max imputable INPUT frames per sample")
    parser.add_argument("--match-tolerance-sec", type=int, default=900,
                        help="GPM nearest-match tolerance (default 15 min)")
    parser.add_argument("--buffer-size", type=int, default=256)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    all_tracks = [parse_best_track(str(p)) for p in sorted(Path(args.cma_dir).glob("*.txt"))]
    cma = pd.concat(all_tracks, ignore_index=True) if all_tracks else pd.DataFrame()

    if cma.empty:
        raise RuntimeError(f"No CMA best-track files found in {args.cma_dir}")

    tif_index = build_tif_index(args.tif_dir)
    distance_center, dx, dy = static_grid_channels()

    # Reference GPM grid (CRS + transform) — read from the first available TIF.
    ref_crs, ref_transform = None, None
    for tid_frames in tif_index.values():
        for p in tid_frames.values():
            with rasterio.open(p) as src:
                ref_crs, ref_transform = src.crs, src.transform
            break
        if ref_transform is not None:
            break
    if ref_transform is None:
        raise RuntimeError("No GPM TIF files found to establish the reference grid.")

    n_samples = 0
    n_dropped_target = 0
    writer = None if args.dry_run else StreamingH5Writer(args.out, GRID_SIZE, args.buffer_size)

    for typhoon_id, group in cma.groupby("Typhoon_ID"):
        interp = interpolate_track(group)
        if interp.empty or len(interp) < args.seq_len:
            continue

        for i in range(0, len(interp) - args.seq_len + 1):
            window_rows = interp.iloc[i:i + args.seq_len]
            anchor = window_rows.iloc[INPUT_SEQ_LEN - 1]   # last INPUT frame
            anchor_lat, anchor_lon = float(anchor["Lat"]), float(anchor["Lon"])
            anchor_time = anchor["Time"]

            # ONE fixed grid centered on the anchor.
            window, win_transform, grid_crs = compute_anchor_grid(
                ref_transform, ref_crs, anchor_lat, anchor_lon
            )

            # DEM resampled onto this anchor grid (once per sample).
            dem = resample_dem_to_grid(args.dem, grid_crs, win_transform)
            if dem is None:
                n_dropped_target += 1
                continue
            dem, dh_dx, dh_dy, land_mask = terrain_from_dem(dem)
            terrain = np.stack([dem, dh_dx, dh_dy, land_mask], axis=0)  # [4,H,W]

            # Assemble precip for all 12 frames on the SAME anchor grid.
            precip = []          # [12,H,W] or None entries
            match_offsets = []   # per-frame offset (input only, target offset tracked separately)
            paths_used = set()

            for t, (_, row) in enumerate(window_rows.iterrows()):
                tif_path, offset = lookup_gpm(
                    tif_index, typhoon_id, row["Time"].to_pydatetime(),
                    args.match_tolerance_sec,
                )
                is_target = (t == INPUT_SEQ_LEN)  # index 11 is the target
                if tif_path is None:
                    precip.append(None)
                else:
                    crop = read_precip_window(tif_path, window)
                    precip.append(crop)
                    if tif_path in paths_used:
                        warnings.warn(
                            f"GPM frame {tif_path} reused within one sample for "
                            f"typhoon {typhoon_id}."
                        )
                    paths_used.add(tif_path)
                match_offsets.append(offset if offset is not None else -1.0)

            # TARGET must never be imputed — drop if missing.
            if precip[INPUT_SEQ_LEN] is None:
                n_dropped_target += 1
                continue

            # Impute only INPUT frames.
            imputed_mask = np.zeros(INPUT_SEQ_LEN, dtype="uint8")
            missing_input = [t for t in range(INPUT_SEQ_LEN) if precip[t] is None]
            if len(missing_input) > args.max_missing:
                continue
            for t in missing_input:
                left, right = t - 1, t + 1
                while left >= 0 and precip[left] is None:
                    left -= 1
                while right <= INPUT_SEQ_LEN - 1 and precip[right] is None:
                    right += 1
                if left >= 0 and right <= INPUT_SEQ_LEN - 1:
                    wl = (right - t) / (right - left)
                    wr = (t - left) / (right - left)
                    precip[t] = precip[left] * wl + precip[right] * wr
                elif left >= 0:
                    precip[t] = precip[left].copy()
                elif right <= INPUT_SEQ_LEN - 1:
                    precip[t] = precip[right].copy()
                else:
                    continue
                imputed_mask[t] = 1
                match_offsets[t] = -1.0

            precip_input = np.stack(precip[:INPUT_SEQ_LEN], axis=0)      # [11,H,W]
            precip_target = precip[INPUT_SEQ_LEN][None, ...]              # [1,H,W]

            # Track features for the 11 INPUT frames.
            inp = window_rows.iloc[:INPUT_SEQ_LEN]
            track = np.stack([
                inp["Lat"].to_numpy(), inp["Lon"].to_numpy(),
                inp["Wind_Speed"].to_numpy(), inp["Pressure"].to_numpy(),
                inp["u_move"].to_numpy(), inp["v_move"].to_numpy(),
            ], axis=1).astype("float32")                                  # [11,6]

            sample = {
                "precip/input": precip_input.astype("float32"),
                "precip/target": precip_target.astype("float32"),
                "terrain": terrain,
                "track": track,
                "meta/typhoon_id": np.int64(typhoon_id),
                "meta/year": np.int64(window_rows.iloc[-1]["Time"].year),
                "meta/start_time": np.int64(window_rows.iloc[0]["Time"].timestamp()),
                "meta/anchor_time": np.int64(anchor_time.timestamp()),
                "meta/target_time": np.int64(window_rows.iloc[-1]["Time"].timestamp()),
                "meta/anchor_lat": np.float32(anchor_lat),
                "meta/anchor_lon": np.float32(anchor_lon),
                "meta/grid_transform": np.asarray([
                    win_transform.a, win_transform.b, win_transform.c,
                    win_transform.d, win_transform.e, win_transform.f,
                ], dtype="float64"),
                "meta/input_imputed_mask": imputed_mask,
                "meta/gpm_match_offset": np.asarray(match_offsets[:INPUT_SEQ_LEN], dtype="float32"),
            }
            if args.dry_run:
                n_samples += 1
                continue
            writer.add(sample)
            n_samples += 1

    total = n_samples
    if not args.dry_run and writer is not None:
        total = writer.close()

    print(f"[build] samples written: {total}")
    print(f"[build] samples dropped (missing target / no DEM): {n_dropped_target}")


if __name__ == "__main__":
    main()
