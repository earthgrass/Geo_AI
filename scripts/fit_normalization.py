"""Fit per-feature normalization statistics using TRAIN EVENTS ONLY.

Writes ``configs/normalization_v1.json``. Statistics are fitted on the training
split and reused (frozen) for validation and test.

Normalization policy:
    - precipitation: min-max (vmin=0, vmax = train max)
    - track features (center_wind_speed, center_pressure, u_move, v_move): z-score
    - terrain (dem, dh_dx, dh_dy): z-score
    - land_mask: NOT normalized (0/1)
    - log1p precipitation: DISABLED by default (future experiment option)

Train events are selected by --train-years (range proxy) or --train-typhoons
(explicit event list). The final event split is NOT frozen by this script.

Run from repo root:
    python scripts/fit_normalization.py --h5 ConvLSTM_Dataset_128.h5 \
        --train-years 2014 2022 --out configs/normalization_v1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import h5py

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import TRACK_FEATURE_NAMES  # noqa: E402

_ZSCORE_TRACK = ["center_wind_speed", "center_pressure", "u_move", "v_move"]
_ZSCORE_TERRAIN = ["dem", "dh_dx", "dh_dy"]  # indices 0,1,2 of /terrain


def _streaming_stats(ds, indices, channel=None, chunk: int = 256):
    """Streaming (Welford) mean/std/max over selected HDF5 samples.

    Args:
        ds: h5py dataset.
        indices: sample indices to include.
        channel: optional integer channel to select (e.g. terrain channel k).
    """
    n = 0
    mean = 0.0
    M2 = 0.0
    vmax = -float("inf")
    for i in range(0, len(indices), chunk):
        idx = sorted(int(x) for x in indices[i:i + chunk])
        if channel is not None:
            x = ds[idx, channel].astype("float64")
        else:
            x = ds[idx].astype("float64")

        cnt = x.size
        x_mean = float(x.mean())
        x_M2 = float(((x - x_mean) ** 2).sum())
        x_max = float(x.max())

        delta = x_mean - mean
        new_n = n + cnt
        mean = mean + delta * cnt / new_n
        M2 = M2 + x_M2 + delta ** 2 * n * cnt / new_n
        n = new_n
        vmax = max(vmax, x_max)

    std = float(np.sqrt(M2 / n)) if n > 1 else 0.0
    return mean, std, vmax


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit normalization statistics.")
    parser.add_argument("--h5", default="ConvLSTM_Dataset_128.h5")
    parser.add_argument("--train-years", nargs=2, type=int, default=None,
                        help="Train year range [min max] (proxy; prefer event list)")
    parser.add_argument("--train-typhoons", default=None,
                        help="Comma-separated train typhoon IDs or a file path")
    parser.add_argument("--out", default="configs/normalization_v1.json")
    args = parser.parse_args()

    with h5py.File(args.h5, "r") as f:
        years = f["meta/year"][:]
        tids = f["meta/typhoon_id"][:]

        mask = np.zeros(len(years), dtype=bool)
        if args.train_typhoons is not None:
            p = Path(args.train_typhoons)
            if p.exists():
                ids = {int(x) for x in p.read_text().split() if x.strip()}
            else:
                ids = {int(x) for x in args.train_typhoons.split(",") if x.strip()}
            mask = np.isin(tids, list(ids))
        elif args.train_years is not None:
            lo, hi = args.train_years
            mask = (years >= lo) & (years <= hi)
        else:
            raise SystemExit("Provide --train-years or --train-typhoons to define train events.")

        train_idx = np.where(mask)[0]
        if len(train_idx) == 0:
            raise RuntimeError("No train samples matched the selection.")

        # Precipitation vmax (min-max), streamed.
        _, _, p_vmax = _streaming_stats(f["precip/input"], train_idx)

        # Track features (small — load all train rows).
        track = f["track"][sorted(train_idx.tolist())].astype("float64")  # [M,11,6]
        track_stats = {}
        for name in _ZSCORE_TRACK:
            j = TRACK_FEATURE_NAMES.index(name)
            vals = track[:, :, j]
            track_stats[name] = {"mean": float(vals.mean()), "std": float(vals.std())}

        # Terrain (large — stream per channel).
        terrain_stats = {}
        for k, name in enumerate(_ZSCORE_TERRAIN):
            mean, std, _ = _streaming_stats(f["terrain"], train_idx, channel=k)
            terrain_stats[name] = {"mean": mean, "std": std}

    out = {
        "version": "1.0",
        "fitted_on": "train events only",
        "train_sample_count": int(len(train_idx)),
        "precipitation": {"normalize": "minmax", "vmin": 0.0, "vmax": round(p_vmax, 3)},
        "track_features": track_stats,
        "terrain": terrain_stats,
        "land_mask": {"normalize": False},
        "log1p_precipitation": {"enabled": False, "note": "future experiment option"},
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[fit] wrote {out_path}")


if __name__ == "__main__":
    main()
