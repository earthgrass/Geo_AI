"""Build the 12-channel paper dataset from CMA + GPM + DEM.

This script refactors the VALID data-engineering logic from the legacy
competition scripts WITHOUT importing them:

  * step1.1 data process.py          -> CMA best-track parsing + cubic-spline
                                        interpolation to 0.5h.
  * step2.1_spatial_dataloader.py    -> GPM TIF cropping + sliding-window assembly.
  * step2_3_generate_metrix.py       -> DEM crop (RealGeographyEngine.get_real_dem_and_mask).

Legacy synthetic 2D wind/pressure fields are intentionally NOT reproduced.
Instead:
  * center_wind_speed / center_pressure are CMA-observed scalars broadcast spatially.
  * u_move / v_move are STORM TRANSLATION velocities (never atmospheric wind).

Channel schema (canonical, imported from src.config):
    0  precipitation      GPM IMERG real precipitation (mm/h)
    1  center_wind_speed  CMA observed max wind speed, broadcast (m/s)
    2  center_pressure    CMA observed central pressure, broadcast (hPa)
    3  distance_center    distance from storm center (km)
    4  dx                 zonal relative coordinate (km, east-positive)
    5  dy                 meridional relative coordinate (km, north-positive)
    6  u_move             storm translation zonal velocity (km/h)
    7  v_move             storm translation meridional velocity (km/h)
    8  dem                terrain elevation (m)
    9  dh_dx              terrain gradient d(elev)/dx (m/km)
    10 dh_dy              terrain gradient d(elev)/dy (m/km)
    11 land_mask          land/ocean mask (0=ocean, 1=land)

HDF5 layout written:
    /data  [N, 12, 12, 128, 128]     (12 temporal frames = 11 input + 1 target)
    /meta/typhoon_id  [N] int
    /meta/year        [N] int
    /meta/start_time  [N] int64       (unix seconds)
    /meta/target_time [N] int64       (unix seconds)
    root attrs: schema_version, channel_names, seq_len, input_seq_len,
                grid_size, grid_resolution_km, created_by, precipitation_units

Run from repo root:
    python scripts/build_paper_dataset.py \
        --cma-dir CMABSTdata --tif-dir TIFdata --dem Global_DEM.tif \
        --out ConvLSTM_Dataset_128.h5
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import rasterio
from rasterio.windows import Window
from scipy.interpolate import CubicSpline

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import CHANNEL_NAMES, GRID_SIZE, GRID_RESOLUTION_KM  # noqa: E402

SEQ_LEN = 12          # total frames per sample (11 input + 1 target)
INPUT_SEQ_LEN = 11
N_CHANNELS = len(CHANNEL_NAMES)
SCHEMA_VERSION = "1.0"
PRECIP_UNITS = "mm/h"

_KM_PER_DEG_LAT = 111.19
_TIF_TIME_RE = re.compile(r"(\d{8}-S\d{6})")


# ---------------------------------------------------------------------------
# CMA best-track parsing + interpolation (refactor of step1.1)
# ---------------------------------------------------------------------------

def parse_best_track(txt_path: str) -> pd.DataFrame:
    """Parse a single CMA best-track TXT file into a DataFrame.

    Returns columns: Typhoon_ID, Time, Lat, Lon, Pressure, Wind_Speed.
    """
    rows = []
    current_id = None
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "66666":
            # Header record. The 4-digit typhoon ID is parts[4] in the CMA
            # best-track format used here (matching legacy step1.1).
            current_id = parts[4] if len(parts) > 4 else None
            continue
        if current_id is None:
            continue
        # Data line: time, intensity, lat(0.1deg), lon(0.1deg), pressure, wind.
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
            "Lat": lat,
            "Lon": lon,
            "Pressure": pres,
            "Wind_Speed": wind,
        })

    return pd.DataFrame(rows)


def interpolate_track(group: pd.DataFrame) -> pd.DataFrame:
    """Cubic-spline interpolate one typhoon's track to 0.5h resolution.

    Also computes the instantaneous translation velocity (u_move, v_move) in
    km/h. Returns a DataFrame with Time, Lat, Lon, Pressure, Wind_Speed,
    u_move, v_move.
    """
    group = group.sort_values("Time").reset_index(drop=True)
    if len(group) < 3:
        return pd.DataFrame()

    start = group["Time"].iloc[0]
    hours = (group["Time"] - start).dt.total_seconds().to_numpy() / 3600.0

    new_hours = np.arange(0.0, hours[-1] + 0.5, 0.5)
    new_times = start + pd.to_timedelta(new_hours, unit="h")

    cs = {
        col: CubicSpline(hours, group[col].to_numpy())
        for col in ("Lat", "Lon", "Pressure", "Wind_Speed")
    }

    out = pd.DataFrame({
        "Time": new_times,
        "Lat": cs["Lat"](new_hours),
        "Lon": cs["Lon"](new_hours),
        "Pressure": cs["Pressure"](new_hours),
        "Wind_Speed": np.clip(cs["Wind_Speed"](new_hours), a_min=0.0, a_max=None),
    })

    # Instantaneous translation velocity (km/h), forward difference, last value
    # repeats the previous step.
    dlat = np.diff(out["Lat"].to_numpy(), prepend=out["Lat"].iloc[0])
    dlon = np.diff(out["Lon"].to_numpy(), prepend=out["Lon"].iloc[0])
    dt_h = 0.5
    # Meridional velocity = dlat * km/deg; zonal = dlon * km/deg * cos(lat).
    out["v_move"] = dlat * _KM_PER_DEG_LAT / dt_h
    out["u_move"] = dlon * _KM_PER_DEG_LAT * np.cos(np.radians(out["Lat"])) / dt_h
    return out


# ---------------------------------------------------------------------------
# Raster cropping (refactor of step2.1 / step2_3)
# ---------------------------------------------------------------------------

def _window_for(grid_size: int, transform, lat: float, lon: float) -> Window:
    col, row = ~transform * (lon, lat)
    col, row = int(np.round(col)), int(np.round(row))
    return Window(col - grid_size // 2, row - grid_size // 2, grid_size, grid_size)


def read_precip_crop(tif_path: str, lat: float, lon: float,
                     grid_size: int = GRID_SIZE) -> np.ndarray | None:
    """Crop a 128x128 precipitation patch centered on (lat, lon)."""
    try:
        with rasterio.open(tif_path) as src:
            window = _window_for(grid_size, src.transform, lat, lon)
            crop = src.read(1, window=window, boundless=True, fill_value=0.0)
    except Exception:
        return None
    crop = np.where(np.isnan(crop), 0.0, crop)
    crop = np.where(crop < 0, 0.0, crop)
    return crop.astype("float32")


def read_dem_crop(dem_path: str, lat: float, lon: float,
                  grid_size: int = GRID_SIZE,
                  resolution_km: float = GRID_RESOLUTION_KM) -> tuple | None:
    """Crop DEM and derive land mask + terrain gradients.

    Returns (dem, dh_dx, dh_dy, land_mask), all [grid_size, grid_size], or None
    if the DEM file is unavailable.
    """
    if not Path(dem_path).exists():
        return None
    try:
        with rasterio.open(dem_path) as src:
            window = _window_for(grid_size, src.transform, lat, lon)
            dem = src.read(1, window=window, boundless=True, fill_value=0.0)
    except Exception:
        return None

    dem = np.clip(dem, 0.0, None)
    land_mask = (dem > 0.0).astype("float32")

    # Terrain gradient in m/km (elevation change per grid spacing).
    dh_dy, dh_dx = np.gradient(dem, resolution_km, resolution_km)

    return (
        dem.astype("float32"),
        dh_dx.astype("float32"),
        dh_dy.astype("float32"),
        land_mask.astype("float32"),
    )


# ---------------------------------------------------------------------------
# Static per-grid channels (distance_center, dx, dy)
# ---------------------------------------------------------------------------

def static_grid_channels(grid_size: int = GRID_SIZE,
                         resolution_km: float = GRID_RESOLUTION_KM):
    y, x = np.ogrid[-grid_size // 2:grid_size // 2, -grid_size // 2:grid_size // 2]
    distance_center = np.sqrt(x ** 2 + y ** 2) * resolution_km
    dx = (x * resolution_km).astype("float32")
    dy = (y * resolution_km).astype("float32")
    return distance_center.astype("float32"), dx, dy


# ---------------------------------------------------------------------------
# Frame assembly
# ---------------------------------------------------------------------------

def build_frame(row, tif_path, dem_crop, static, u_move, v_move) -> np.ndarray | None:
    """Assemble a single [12, H, W] frame.

    Returns None when the precipitation crop is missing (the frame will be
    dropped or imputed by the caller).
    """
    precip = read_precip_crop(tif_path, row["Lat"], row["Lon"])
    if precip is None:
        return None

    distance_center, dx, dy = static
    grid_size = distance_center.shape[0]

    wind = np.full((grid_size, grid_size), float(row["Wind_Speed"]), dtype="float32")
    press = np.full((grid_size, grid_size), float(row["Pressure"]), dtype="float32")
    u_f = np.full((grid_size, grid_size), float(u_move), dtype="float32")
    v_f = np.full((grid_size, grid_size), float(v_move), dtype="float32")

    if dem_crop is None:
        dem = np.zeros((grid_size, grid_size), dtype="float32")
        dh_dx = np.zeros_like(dem)
        dh_dy = np.zeros_like(dem)
        land_mask = np.zeros_like(dem)
    else:
        dem, dh_dx, dh_dy, land_mask = dem_crop

    frame = np.stack([
        precip, wind, press, distance_center, dx, dy, u_f, v_f,
        dem, dh_dx, dh_dy, land_mask,
    ], axis=0)  # [12, H, W]
    return frame.astype("float32")


def impute_precip_sequence(frames: list[np.ndarray | None], max_missing: int = 2):
    """Time-weighted linear imputation for missing precipitation frames."""
    missing = [i for i, f in enumerate(frames) if f is None]
    if len(missing) > max_missing:
        return None
    for idx in missing:
        left, right = idx - 1, idx + 1
        while left >= 0 and frames[left] is None:
            left -= 1
        while right < len(frames) and frames[right] is None:
            right += 1
        if left >= 0 and right < len(frames):
            wl = (right - idx) / (right - left)
            wr = (idx - left) / (right - left)
            frames[idx] = frames[left] * wl + frames[right] * wr
        elif left >= 0:
            frames[idx] = frames[left].copy()
        elif right < len(frames):
            frames[idx] = frames[right].copy()
        else:
            return None
    return frames


# ---------------------------------------------------------------------------
# TIF index
# ---------------------------------------------------------------------------

def build_tif_index(tif_dir: str) -> dict:
    """Build {folder_id: {datetime: tif_path}} by scanning the GPM archive."""
    index: dict = {}
    root = Path(tif_dir)
    for tif_path in root.rglob("*.tif"):
        m = _TIF_TIME_RE.search(tif_path.name)
        if not m:
            continue
        ts = datetime.strptime(m.group(1), "%Y%m%d-S%H%M%S")
        folder_id = tif_path.parent.name
        index.setdefault(folder_id, {})[ts] = str(tif_path)
    return index


def folder_to_typhoon_id(folder_id: str) -> str:
    """Map a GPM folder id (e.g. '2014100') to a 4-digit typhoon id ('1410')."""
    return folder_id[2:6]


def nearest_tif_path(tif_index: dict, typhoon_id: str, time: datetime) -> str | None:
    """Find the TIF closest in time for a typhoon event (within 30 min)."""
    folder_candidates = [
        fid for fid in tif_index
        if folder_to_typhoon_id(fid) == typhoon_id.zfill(4)
    ]
    best = None
    best_dt = None
    for fid in folder_candidates:
        for ts, path in tif_index[fid].items():
            dt = abs((ts - time).total_seconds())
            if best_dt is None or dt < best_dt:
                best_dt = dt
                best = path
    if best_dt is not None and best_dt <= 30 * 60:
        return best
    return None


# ---------------------------------------------------------------------------
# HDF5 writer
# ---------------------------------------------------------------------------

def write_hdf5(samples: list, out_path: str) -> None:
    """Write samples [each (frames[12,12,H,W], typhoon_id, year, start_ts, target_ts)]."""
    n = len(samples)
    frames = np.stack([s[0] for s in samples], axis=0).astype("float32")  # [N,12,12,H,W]
    typhoon_ids = np.array([s[1] for s in samples], dtype="int64")
    years = np.array([s[2] for s in samples], dtype="int64")
    start_time = np.array([s[3] for s in samples], dtype="int64")
    target_time = np.array([s[4] for s in samples], dtype="int64")

    with h5py.File(out_path, "w") as f:
        f.create_dataset(
            "data", data=frames, dtype="float32",
            compression="gzip", chunks=(1, 12, 12, frames.shape[3], frames.shape[4]),
        )
        meta = f.create_group("meta")
        meta.create_dataset("typhoon_id", data=typhoon_ids, dtype="int64")
        meta.create_dataset("year", data=years, dtype="int64")
        meta.create_dataset("start_time", data=start_time, dtype="int64")
        meta.create_dataset("target_time", data=target_time, dtype="int64")

        f.attrs["schema_version"] = SCHEMA_VERSION
        f.attrs["channel_names"] = CHANNEL_NAMES
        f.attrs["seq_len"] = SEQ_LEN
        f.attrs["input_seq_len"] = INPUT_SEQ_LEN
        f.attrs["grid_size"] = GRID_SIZE
        f.attrs["grid_resolution_km"] = GRID_RESOLUTION_KM
        f.attrs["created_by"] = "scripts/build_paper_dataset.py"
        f.attrs["precipitation_units"] = PRECIP_UNITS

    print(f"[build] wrote {n} samples to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build the 12-channel paper dataset.")
    parser.add_argument("--cma-dir", default="CMABSTdata")
    parser.add_argument("--tif-dir", default="TIFdata")
    parser.add_argument("--dem", default="Global_DEM.tif")
    parser.add_argument("--out", default="ConvLSTM_Dataset_128.h5")
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--max-missing", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true",
                        help="Count samples without writing.")
    args = parser.parse_args()

    # 1. Parse all CMA tracks.
    all_tracks = []
    for txt in sorted(Path(args.cma_dir).glob("*.txt")):
        all_tracks.append(parse_best_track(str(txt)))
    if not all_tracks:
        raise RuntimeError(f"No CMA best-track files found in {args.cma_dir}")
    cma = pd.concat(all_tracks, ignore_index=True)

    # 2. Build TIF index.
    tif_index = build_tif_index(args.tif_dir)

    # 3. Static grid channels.
    static = static_grid_channels()

    samples = []
    for typhoon_id, group in cma.groupby("Typhoon_ID"):
        interp = interpolate_track(group)
        if interp.empty or len(interp) < args.seq_len:
            continue

        frames = []
        for _, row in interp.iterrows():
            tif_path = nearest_tif_path(
                tif_index, typhoon_id, row["Time"].to_pydatetime()
            )
            dem_crop = read_dem_crop(args.dem, row["Lat"], row["Lon"])
            frame = build_frame(
                row, tif_path, dem_crop, static,
                row.get("u_move", 0.0), row.get("v_move", 0.0),
            )
            frames.append(frame)

        # Sliding window (stride=1), each window within this typhoon only.
        for i in range(0, len(frames) - args.seq_len + 1):
            window = frames[i:i + args.seq_len]
            imputed = impute_precip_sequence(list(window), args.max_missing)
            if imputed is None:
                continue
            seq = np.stack(imputed, axis=0)  # [12, 12, H, W]
            target_row = interp.iloc[i + args.seq_len - 1]
            start_row = interp.iloc[i]
            samples.append((
                seq,
                int(typhoon_id),
                int(target_row["Time"].year),
                int(start_row["Time"].timestamp()),
                int(target_row["Time"].timestamp()),
            ))

    print(f"[build] total samples: {len(samples)}")
    if args.dry_run:
        return
    if samples:
        write_hdf5(samples, args.out)
    else:
        print("[build] no samples generated; nothing written.")


if __name__ == "__main__":
    main()
