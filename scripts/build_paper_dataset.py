"""Build the schema-v2 paper dataset with CAUSAL track reconstruction.

This builder enforces a forecast-realistic, leakage-free data pipeline:

1. Fixed anchor grid (no future-track oracle for the target crop).
2. Causal / as-of-anchor track features (P0):
   Track/intensity features at anchor time t depend ONLY on CMA best-track
   fixes with timestamp <= t. No full-event spline. Positions after the latest
   available fix use a constant-velocity estimate; wind/pressure use
   persistence. Future CMA fixes can never alter an earlier sample.
3. Physical terrain-gradient units (P0): dh_dx / dh_dy in m/km, computed from
   the anchor-grid geotransform + anchor latitude (no 10-km-pixel assumption).
4. GPM actual-time / forecast-lead audit (P0): lookup returns signed offsets;
   the actual anchor and target GPM times are stored so the validator can report
   the true forecast lead.

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
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import (  # noqa: E402
    CHANNEL_NAMES,
    TRACK_FEATURE_NAMES,
    TERRAIN_CHANNEL_NAMES,
    TERRAIN_GRADIENT_UNITS,
    SCHEMA_VERSION,
    GRID_SIZE,
)

# Schema is FIXED for v2 (option A: seq-len is not configurable).
SEQ_LEN = 12
INPUT_SEQ_LEN = 11
N_TRACK = len(TRACK_FEATURE_NAMES)   # 6
PRECIP_UNITS = "mm/h"
DT_H = 0.5                            # hours per frame

_KM_PER_DEG_LAT = 111.19
_TIF_TIME_RE = re.compile(r"(\d{8}-S\d{6})")


# ---------------------------------------------------------------------------
# CMA parsing
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
            lat = float(parts[2]) / 10.0
            lon = float(parts[3]) / 10.0
            pres = float(parts[4])
            wind = float(parts[5])
        except (ValueError, IndexError):
            continue
        # CMA uses 0-360 deg longitude; normalize to EPSG:4326 (-180..180).
        if lon > 180.0:
            lon -= 360.0
        rows.append({
            "Typhoon_ID": str(current_id).zfill(4),
            "Time": pd.to_datetime(parts[0], format="%Y%m%d%H"),
            "Lat": lat, "Lon": lon, "Pressure": pres, "Wind_Speed": wind,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Causal / as-of-anchor track reconstruction (P0)
# ---------------------------------------------------------------------------

def _linear_extrapolate(fix_ts, fix_vals, t):
    """Linear interp within bracket; constant-velocity extrapolation beyond ends."""
    if t <= fix_ts[0]:
        if len(fix_ts) >= 2:
            slope = (fix_vals[1] - fix_vals[0]) / (fix_ts[1] - fix_ts[0])
            return fix_vals[0] + slope * (t - fix_ts[0])
        return fix_vals[0]
    if t >= fix_ts[-1]:
        if len(fix_ts) >= 2:
            slope = (fix_vals[-1] - fix_vals[-2]) / (fix_ts[-1] - fix_ts[-2])
            return fix_vals[-1] + slope * (t - fix_ts[-1])
        return fix_vals[-1]
    return float(np.interp(t, fix_ts, fix_vals))


def _persist(fix_ts, fix_vals, t):
    """Interp within bracket; persistence beyond the last fix (wind/pressure)."""
    if t <= fix_ts[0]:
        return fix_vals[0]
    if t >= fix_ts[-1]:
        return fix_vals[-1]
    return float(np.interp(t, fix_ts, fix_vals))


def causal_track_at(raw: pd.DataFrame, anchor_time: pd.Timestamp):
    """Reconstruct the 11 input frames' track features causally at anchor_time.

    Uses ONLY CMA fixes with Time <= anchor_time. Returns:
        (track [11,6], latest_fix_time, cma_fix_age_sec)
    where track columns follow TRACK_FEATURE_NAMES
    [lat, lon, center_wind_speed, center_pressure, u_move, v_move].
    Returns None if fewer than 2 available fixes.
    """
    avail = raw[raw["Time"] <= anchor_time].sort_values("Time")
    if len(avail) < 2:
        return None

    fix_ts = avail["Time"].astype("datetime64[s]").astype("int64").to_numpy()
    fix_lat = avail["Lat"].to_numpy()
    fix_lon = avail["Lon"].to_numpy()
    fix_wind = avail["Wind_Speed"].to_numpy()
    fix_pres = avail["Pressure"].to_numpy()

    # n_input+1 times: anchor - INPUT_SEQ_LEN*DT_H ... anchor.
    anchor_sec = anchor_time.timestamp()
    offsets = -np.arange(INPUT_SEQ_LEN, -1, -1) * DT_H * 3600.0
    times = anchor_sec + offsets

    lat = np.array([_linear_extrapolate(fix_ts, fix_lat, t) for t in times])
    # Unwrap longitude (handle the +/-180 dateline discontinuity) before linear
    # extrapolation, then wrap the result back to [-180, 180].
    fix_lon_unwrapped = np.unwrap(fix_lon, period=360.0)
    lon_unwrapped = np.array([_linear_extrapolate(fix_ts, fix_lon_unwrapped, t) for t in times])
    lon = ((lon_unwrapped + 180.0) % 360.0) - 180.0
    wind = np.array([_persist(fix_ts, fix_wind, t) for t in times])
    pres = np.array([_persist(fix_ts, fix_pres, t) for t in times])

    # Translation velocities from causal positions (forward differences).
    # Use the UNWRAPPED longitude for dlon so dateline crossings do not
    # produce spurious huge zonal velocities.
    dlon = np.diff(lon_unwrapped)   # per DT_H hours
    dlat = np.diff(lat)
    cos_lat = np.cos(np.radians(lat[1:]))
    u_move = dlon * _KM_PER_DEG_LAT * cos_lat / DT_H
    v_move = dlat * _KM_PER_DEG_LAT / DT_H

    track = np.stack([
        lat[1:], lon[1:], wind[1:], pres[1:], u_move, v_move,
    ], axis=1).astype("float32")   # [11, 6]

    latest_fix_time = avail["Time"].iloc[-1]
    cma_fix_age = (anchor_time - latest_fix_time).total_seconds()
    return track, latest_fix_time, float(cma_fix_age)


# ---------------------------------------------------------------------------
# GPM index + lookup (signed offset, actual timestamp)
# ---------------------------------------------------------------------------

def build_tif_index(tif_dir: str) -> dict:
    index: dict = {}
    for tif_path in Path(tif_dir).rglob("*.tif"):
        m = _TIF_TIME_RE.search(tif_path.name)
        if not m:
            continue
        ts = datetime.strptime(m.group(1), "%Y%m%d-S%H%M%S")
        tid = tif_path.parent.name[2:6].zfill(4)
        index.setdefault(tid, {})[ts] = str(tif_path)
    return index


def lookup_gpm(index, typhoon_id, time, tolerance_sec):
    """Return (path, actual_timestamp, signed_offset_seconds).

    signed_offset = actual_time - requested_time (negative if GPM leads).
    Returns (None, None, None) if no frame within tolerance.
    """
    frames = index.get(typhoon_id, {})
    if not frames:
        return None, None, None
    if time in frames:
        return frames[time], time, 0.0

    best_ts, best_adt = None, None
    for ts in frames:
        adt = abs((ts - time).total_seconds())
        if adt <= tolerance_sec and (best_adt is None or adt < best_adt):
            best_adt, best_ts = adt, ts
    if best_ts is None:
        return None, None, None
    return frames[best_ts], best_ts, (best_ts - time).total_seconds()


# ---------------------------------------------------------------------------
# Anchor grid + DEM (physical terrain-gradient units)
# ---------------------------------------------------------------------------

def anchor_grid_transform(anchor_lat, anchor_lon, grid_size=GRID_SIZE, pixel=0.1):
    """Canonical NORTH-UP anchor grid transform (0.1-deg pixels, 128x128)."""
    half = (grid_size / 2.0) * pixel
    x_min = anchor_lon - half
    y_max = anchor_lat + half
    return from_origin(x_min, y_max, pixel, pixel)


def reproject_to_grid(src, dst_transform, dst_crs, grid_size=GRID_SIZE):
    """Reproject an OPEN rasterio source onto the canonical anchor grid.

    Uses the source's OWN transform/CRS, so per-file tile offsets are handled
    correctly. Returns a [grid_size, grid_size] north-up float32 array.
    """
    dest = np.zeros((grid_size, grid_size), dtype="float32")
    reproject(
        source=rasterio.band(src, 1),
        destination=dest,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
    )
    return dest


def precip_grid_from_path(raster_path, dst_transform, dst_crs, grid_size=GRID_SIZE):
    """Open a GPM TIF and read it onto the anchor grid, then clean values."""
    try:
        with rasterio.open(raster_path) as src:
            crop = reproject_to_grid(src, dst_transform, dst_crs, grid_size)
    except Exception:
        return None
    crop = np.where(np.isnan(crop), 0.0, crop)
    crop = np.where(crop < 0, 0.0, crop)
    return crop.astype("float32")


def terrain_from_dem(dem, anchor_transform, anchor_lat):
    """Terrain channels with PHYSICAL gradient units (m/km)."""
    dem = np.clip(dem, 0.0, None).astype("float32")
    land_mask = (dem > 0.0).astype("float32")

    pixel_deg_x = abs(float(anchor_transform.a))
    pixel_deg_y = abs(float(anchor_transform.e))
    km_per_deg_lon = _KM_PER_DEG_LAT * np.cos(np.radians(anchor_lat))
    dx_km = pixel_deg_x * km_per_deg_lon   # zonal km per pixel
    dy_km = pixel_deg_y * _KM_PER_DEG_LAT  # meridional km per pixel

    dh_dy, dh_dx = np.gradient(dem, dy_km, dx_km)  # m/km
    return dem, dh_dx.astype("float32"), dh_dy.astype("float32"), land_mask


# ---------------------------------------------------------------------------
# Streaming HDF5 writer (schema v2)
# ---------------------------------------------------------------------------

class StreamingH5Writer:
    def __init__(self, out_path, grid_size, buffer_size=256):
        self.f = h5py.File(out_path, "w")
        self.buffer_size = buffer_size
        self._buffer: list = []
        H = W = grid_size

        self._mk("precip/input", (0, INPUT_SEQ_LEN, H, W), (None, INPUT_SEQ_LEN, H, W),
                 "float32", (1, INPUT_SEQ_LEN, H, W))
        self._mk("precip/target", (0, 1, H, W), (None, 1, H, W), "float32", (1, 1, H, W))
        self._mk("terrain", (0, 4, H, W), (None, 4, H, W), "float32", (1, 4, H, W))
        self._mk("track", (0, INPUT_SEQ_LEN, N_TRACK), (None, INPUT_SEQ_LEN, N_TRACK), "float32")

        for name in ("typhoon_id", "year", "start_time", "anchor_time", "target_time",
                     "latest_cma_fix_time", "actual_anchor_gpm_time", "actual_target_gpm_time"):
            self._mk(f"meta/{name}", (0,), (None,), "int64")
        for name in ("anchor_lat", "anchor_lon", "cma_fix_age_sec", "target_gpm_match_offset"):
            self._mk(f"meta/{name}", (0,), (None,), "float32")
        self._mk("meta/grid_transform", (0, 6), (None, 6), "float64")
        self._mk("meta/input_imputed_mask", (0, INPUT_SEQ_LEN), (None, INPUT_SEQ_LEN), "uint8")
        self._mk("meta/input_gpm_match_offset", (0, INPUT_SEQ_LEN), (None, INPUT_SEQ_LEN), "float32")

    def _mk(self, name, shape, maxshape, dtype, chunks=None):
        kwargs = {}
        if chunks is not None:
            kwargs["chunks"] = chunks
            kwargs["compression"] = "gzip"
        self.f.create_dataset(name, shape=shape, maxshape=maxshape, dtype=dtype, **kwargs)

    def add(self, sample: dict) -> None:
        self._buffer.append(sample)
        if len(self._buffer) >= self.buffer_size:
            self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return
        n = len(self._buffer)
        for key in self._buffer[0].keys():
            arr = np.stack([s[key] for s in self._buffer], axis=0)
            ds = self.f[key]
            ds.resize(ds.shape[0] + n, axis=0)
            ds[-n:] = arr
        self._buffer = []

    def close(self) -> int:
        self._flush()
        n = self.f["precip/input"].shape[0]
        self.f.attrs["schema_version"] = SCHEMA_VERSION
        self.f.attrs["channel_names"] = CHANNEL_NAMES
        self.f.attrs["track_feature_names"] = TRACK_FEATURE_NAMES
        self.f.attrs["terrain_channel_names"] = TERRAIN_CHANNEL_NAMES
        self.f.attrs["terrain_gradient_units"] = TERRAIN_GRADIENT_UNITS
        self.f.attrs["seq_len"] = SEQ_LEN
        self.f.attrs["input_seq_len"] = INPUT_SEQ_LEN
        self.f.attrs["grid_size"] = GRID_SIZE
        self.f.attrs["precipitation_units"] = PRECIP_UNITS
        self.f.attrs["created_by"] = "scripts/build_paper_dataset.py"
        self.f.close()
        self.f = None  # release the OS file handle promptly (Windows)
        return n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build causal schema-v2 paper dataset.")
    parser.add_argument("--cma-dir", default="CMABSTdata")
    parser.add_argument("--tif-dir", default="TIFdata")
    parser.add_argument("--dem", default="Global_DEM.tif")
    parser.add_argument("--out", default="ConvLSTM_Dataset_128.h5")
    parser.add_argument("--max-missing", type=int, default=2)
    parser.add_argument("--match-tolerance-sec", type=int, default=900)
    parser.add_argument("--buffer-size", type=int, default=256)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    all_tracks = [parse_best_track(str(p)) for p in sorted(Path(args.cma_dir).glob("*.txt"))]
    cma = pd.concat(all_tracks, ignore_index=True) if all_tracks else pd.DataFrame()
    if cma.empty:
        raise RuntimeError(f"No CMA files in {args.cma_dir}")

    tif_index = build_tif_index(args.tif_dir)

    # GPM IMERG is EPSG:4326; the canonical anchor grid uses this fixed CRS.
    grid_crs = CRS.from_epsg(4326)

    # Open the (large) DEM once and reuse it across all samples.
    dem_src = None
    if Path(args.dem).exists():
        dem_src = rasterio.open(args.dem)

    n_samples = 0
    n_dropped = 0
    writer = None if args.dry_run else StreamingH5Writer(args.out, GRID_SIZE, args.buffer_size)

    for typhoon_id, raw in cma.groupby("Typhoon_ID"):
        raw = raw.sort_values("Time").reset_index(drop=True)
        if len(raw) < 2:
            continue
        # Skip typhoons with no GPM precipitation coverage (avoids wasted DEM
        # resampling for ~266 uncovered events, incl. unnamed '0000').
        if typhoon_id not in tif_index:
            continue

        grid = pd.date_range(
            raw["Time"].min().floor("30min"),
            raw["Time"].max().ceil("30min"),
            freq="30min",
        )

        for i in range(INPUT_SEQ_LEN, len(grid)):
            frame_times = grid[i - INPUT_SEQ_LEN:i + 1]   # 11 input + 1 target
            anchor_time = frame_times[INPUT_SEQ_LEN - 1]  # LAST input frame (grid[i-1])

            # Causal track (P0): only fixes <= anchor_time.
            result = causal_track_at(raw, anchor_time)
            if result is None:
                continue
            track, latest_fix_time, cma_fix_age = result

            anchor = pd.Timestamp(anchor_time)
            anchor_lat = float(track[-1, 0])
            anchor_lon = float(track[-1, 1])

            anchor_transform = anchor_grid_transform(anchor_lat, anchor_lon)
            if dem_src is None:
                n_dropped += 1
                continue
            dem = reproject_to_grid(dem_src, anchor_transform, grid_crs)
            dem, dh_dx, dh_dy, land_mask = terrain_from_dem(dem, anchor_transform, anchor_lat)
            terrain = np.stack([dem, dh_dx, dh_dy, land_mask], axis=0)

            # Precip for all 12 frames on the same anchor grid.
            precip = []
            input_offsets = np.full(INPUT_SEQ_LEN, np.nan, dtype="float32")
            target_offset = np.nan
            actual_anchor_gpm = None
            actual_target_gpm = None

            for t, ft in enumerate(frame_times):
                path, actual_ts, offset = lookup_gpm(
                    tif_index, typhoon_id, ft.to_pydatetime(), args.match_tolerance_sec
                )
                is_target = (t == INPUT_SEQ_LEN)
                if path is None:
                    precip.append(None)
                    continue
                crop = precip_grid_from_path(path, anchor_transform, grid_crs)
                precip.append(crop)
                if is_target:
                    target_offset = float(offset) if offset is not None else np.nan
                    actual_target_gpm = int(actual_ts.timestamp())
                else:
                    input_offsets[t] = float(offset) if offset is not None else np.nan
                    if t == INPUT_SEQ_LEN - 1:
                        actual_anchor_gpm = int(actual_ts.timestamp())

            # Target never imputed -> drop if missing.
            if precip[INPUT_SEQ_LEN] is None:
                n_dropped += 1
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
                input_offsets[t] = np.nan

            precip_input = np.stack(precip[:INPUT_SEQ_LEN], axis=0)
            precip_target = precip[INPUT_SEQ_LEN][None, ...]

            sample = {
                "precip/input": precip_input.astype("float32"),
                "precip/target": precip_target.astype("float32"),
                "terrain": terrain,
                "track": track,
                "meta/typhoon_id": np.int64(typhoon_id),
                "meta/year": np.int64(frame_times[-1].year),
                "meta/start_time": np.int64(frame_times[0].timestamp()),
                "meta/anchor_time": np.int64(anchor_time.timestamp()),
                "meta/target_time": np.int64(frame_times[-1].timestamp()),
                "meta/latest_cma_fix_time": np.int64(latest_fix_time.timestamp()),
                "meta/cma_fix_age_sec": np.float32(cma_fix_age),
                "meta/anchor_lat": np.float32(anchor_lat),
                "meta/anchor_lon": np.float32(anchor_lon),
                "meta/grid_transform": np.asarray([
                    anchor_transform.a, anchor_transform.b, anchor_transform.c,
                    anchor_transform.d, anchor_transform.e, anchor_transform.f,
                ], dtype="float64"),
                "meta/input_imputed_mask": imputed_mask,
                "meta/input_gpm_match_offset": input_offsets,
                "meta/target_gpm_match_offset": np.float32(target_offset),
                "meta/actual_anchor_gpm_time": np.int64(
                    actual_anchor_gpm if actual_anchor_gpm is not None
                    else int(anchor_time.timestamp())
                ),
                "meta/actual_target_gpm_time": np.int64(
                    actual_target_gpm if actual_target_gpm is not None
                    else int(frame_times[-1].timestamp())
                ),
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
    print(f"[build] samples dropped: {n_dropped}")


if __name__ == "__main__":
    main()
