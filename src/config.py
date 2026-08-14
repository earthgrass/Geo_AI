"""Global configuration and path resolution for the PI-ResConvLSTM project."""

import os
from pathlib import Path

# --- Repository root detection ---
REPO_ROOT = Path(__file__).resolve().parent.parent

# --- Directory paths ---
DATA_DIR = REPO_ROOT
DOCS_DIR = REPO_ROOT / "docs"
SRC_DIR = REPO_ROOT / "src"
OUTPUT_DIR = REPO_ROOT / "outputs"
CONFIG_DIR = REPO_ROOT / "configs"
ARCHIVE_DIR = REPO_ROOT / "archive"

# --- Sub-directories ---
CHECKPOINT_DIR = OUTPUT_DIR / "models"
FIGURE_DIR = OUTPUT_DIR / "figures"
LOG_DIR = OUTPUT_DIR / "logs"

# Ensure output directories exist
for d in [CHECKPOINT_DIR, FIGURE_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --- Physical constants ---
EARTH_RADIUS_KM = 6371.0
OMEGA_EARTH = 7.2921e-5  # rad/s, Earth's angular velocity

# --- Data constants ---
GRID_SIZE = 128
GRID_RESOLUTION_KM = 10.0
SEQ_LEN = 12  # Total frames per sample (input + target)
SEQ_LEN_INPUT = 11  # Number of input frames

# --- Reproducibility ---
DEFAULT_SEED = 42

# --- Data file paths (relative to REPO_ROOT) ---
DEFAULT_H5_PATH = str(DATA_DIR / "ConvLSTM_Dataset_128.h5")
DEFAULT_DEM_PATH = str(DATA_DIR / "Global_DEM.tif")
DEFAULT_CMA_DIR = str(DATA_DIR / "CMABSTdata")
DEFAULT_TIF_DIR = str(DATA_DIR / "TIFdata")
DEFAULT_FEATURES_CSV = str(DATA_DIR / "Typhoon_Full_Dataset_Q1.csv")

# --- Precipitation CMA categories (mm/h) ---
CMA_CATEGORIES = {
    'light':       (0.1, 2.0),
    'moderate':    (2.0, 5.0),
    'heavy':       (5.0, 10.0),
    'torrential':  (10.0, 20.0),
    'extreme':     (20.0, float('inf')),
}

# --- Default evaluation thresholds (mm/h) ---
DEFAULT_METRIC_THRESHOLDS = [1.0, 5.0, 10.0, 20.0, 50.0]

# --- Channel layout (canonical 12-channel paper schema) ---
# This is the SINGLE canonical source of channel names. Do not duplicate a
# divergent list elsewhere.
CHANNEL_NAMES = [
    'precipitation',     # 0: GPM IMERG real precipitation (mm/h)
    'center_wind_speed', # 1: CMA observed max wind speed, broadcast spatially (m/s)
    'center_pressure',   # 2: CMA observed central pressure, broadcast spatially (hPa)
    'r_norm',            # 3: Normalized distance from grid center (dimensionless)
    'dx_norm',           # 4: Normalized zonal coordinate, in [-1, 1]
    'dy_norm',           # 5: Normalized meridional coordinate, in [-1, 1]
    'u_move',            # 6: Storm translation zonal velocity (km/h)
    'v_move',            # 7: Storm translation meridional velocity (km/h)
    'dem',               # 8: DEM terrain elevation (m)
    'dh_dx',             # 9: Terrain gradient d(elev)/dx (m/km)
    'dh_dy',             # 10: Terrain gradient d(elev)/dy (m/km)
    'land_mask',         # 11: Land/ocean mask (0=ocean, 1=land)
]
PRECIP_CHANNEL_IDX = 0

# Physical units of the terrain gradient channels (dh_dx, dh_dy).
TERRAIN_GRADIENT_UNITS = "m/km"

# --- HDF5 schema v2 constants ---
SCHEMA_VERSION = "2.0"

# Track feature names stored in /track [N, 11, F]. This is the canonical order.
TRACK_FEATURE_NAMES = [
    'lat',                  # 0: center latitude (deg N)
    'lon',                  # 1: center longitude (deg E)
    'center_wind_speed',    # 2: CMA observed max wind speed (m/s)
    'center_pressure',      # 3: CMA observed central pressure (hPa)
    'u_move',               # 4: storm translation zonal velocity (km/h)
    'v_move',               # 5: storm translation meridional velocity (km/h)
]

# Terrain channels stored in /terrain [N, 4, H, W]. Canonical order.
TERRAIN_CHANNEL_NAMES = [
    'dem',       # terrain elevation (m)
    'dh_dx',     # terrain gradient d(elev)/dx
    'dh_dy',     # terrain gradient d(elev)/dy
    'land_mask', # land/ocean mask (0/1)
]

# Static grid channels are NOT stored — they are reconstructed on-the-fly by the
# dataset from grid geometry (they depend only on grid size, not on the sample).
STATIC_GRID_CHANNELS = ['r_norm', 'dx_norm', 'dy_norm']

# Note on physics semantics:
#   u_move / v_move (channels 6, 7) are STORM TRANSLATION velocities, NOT
#   environmental atmospheric wind components. They must never be used as the
#   wind terms of an orographic uplift constraint.
#
#   center_wind_speed / center_pressure (channels 1, 2) come from CMA best-track
#   observations and are broadcast scalar fields. The legacy parametric synthetic
#   wind/pressure fields are intentionally NOT part of this schema.
