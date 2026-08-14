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

# --- Channel layout (for expanded 16-channel input) ---
CHANNEL_NAMES = [
    'precipitation',     # 0: GPM IMERG precipitation (mm/h)
    'wind_field',        # 1: Parametric wind speed (m/s)
    'pressure_field',    # 2: Parametric pressure (hPa)
    'dist_center',       # 3: Distance from typhoon center (km)
    'dx',                # 4: Zonal offset from center (normalized)
    'dy',                # 5: Meridional offset (normalized)
    'u_move',            # 6: Typhoon zonal movement (km/h)
    'v_move',            # 7: Typhoon meridional movement (km/h)
    'dem',               # 8: DEM elevation (m)
    'dh_dx',             # 9: Terrain gradient d(elev)/dx
    'dh_dy',             # 10: Terrain gradient d(elev)/dy
    'land_mask',         # 11: Land-sea mask (0=ocean, 1=land)
    'r_norm',            # 12: Normalized distance from center
    'cos_theta',         # 13: cos(azimuth angle from center)
    'sin_theta',         # 14: sin(azimuth angle from center)
    'landfall_flag',     # 15: Landfall status (0=at sea, 1=over land)
]
PRECIP_CHANNEL_IDX = 0
