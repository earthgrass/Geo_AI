"""Fit terrain-regime evaluation thresholds using TRAIN events ONLY.

Writes ``configs/evaluation_thresholds_v1.json``. Thresholds are frozen from the
training split and must NOT be tuned on validation/test.

Thresholds:
    - LAND: land_mask == 1 (0/1 binary)
    - HIGH_DEM: 75th percentile of DEM over TRAIN-LAND pixels
    - HIGH_GRAD: 75th percentile of |grad h| over TRAIN-LAND pixels

Run from repo root:
    python scripts/fit_eval_thresholds.py --h5 ConvLSTM_Dataset_128.h5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import h5py
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5", default="ConvLSTM_Dataset_128.h5")
    p.add_argument("--splits", default="configs/splits_v1.yaml")
    p.add_argument("--out", default="configs/evaluation_thresholds_v1.json")
    args = p.parse_args()

    split = yaml.safe_load(open(args.splits, encoding="utf-8"))
    train_ids = set(split["train"])

    with h5py.File(args.h5, "r") as f:
        tids = f["meta/typhoon_id"][:]
        mask = np.isin(tids, list(train_ids))
        train_idx = np.where(mask)[0]

        # terrain channels: [dem, dh_dx, dh_dy, land_mask]
        dem = f["terrain"][sorted(train_idx.tolist()), 0].astype("float32")
        dh_dx = f["terrain"][sorted(train_idx.tolist()), 1].astype("float32")
        dh_dy = f["terrain"][sorted(train_idx.tolist()), 2].astype("float32")
        land = f["terrain"][sorted(train_idx.tolist()), 3].astype("float32")

    land_mask = land > 0.5
    grad_mag = np.sqrt(dh_dx ** 2 + dh_dy ** 2)

    land_dem = dem[land_mask]
    land_grad = grad_mag[land_mask]

    high_dem = float(np.percentile(land_dem, 75.0)) if land_dem.size else 0.0
    high_grad = float(np.percentile(land_grad, 75.0)) if land_grad.size else 0.0

    out = {
        "version": "1.0",
        "fitted_on": "train events only (train-land pixels)",
        "land_mask": {"land": 1.0, "ocean": 0.0},
        "high_dem_threshold_m": round(high_dem, 2),
        "high_grad_threshold_m_per_km": round(high_grad, 2),
    }
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[fit] wrote {args.out}")
    print(f"  HIGH_DEM = {high_dem:.2f} m")
    print(f"  HIGH_GRAD = {high_grad:.2f} m/km")


if __name__ == "__main__":
    main()
