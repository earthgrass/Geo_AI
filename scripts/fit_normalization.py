"""Fit per-feature normalization statistics using TRAIN EVENTS ONLY.

Writes ``configs/normalization_v1.json``. Statistics are fitted on the training
split and reused (frozen) for validation and test.

Precipitation statistics cover BOTH training input AND training target
precipitation, and record mean / std / max / p95 / p99 / p99.9 (percentiles are
estimated from a subsample). No robust scaling or log1p decision is made here —
these numbers support the later literature-informed normalization choice.

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
_ZSCORE_TERRAIN = ["dem", "dh_dx", "dh_dy"]

_PERCENTILE_SAMPLE_CAP = 2_000_000


def _streaming_mean_std_max(ds, indices, channel=None, chunk=256):
    n = 0
    mean = 0.0
    M2 = 0.0
    vmax = -float("inf")
    for i in range(0, len(indices), chunk):
        idx = sorted(int(x) for x in indices[i:i + chunk])
        x = ds[idx, channel].astype("float64") if channel is not None else ds[idx].astype("float64")
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


def _precip_stats(ds_input, ds_target, indices, chunk=128):
    """Streaming mean/std/max + subsampled percentiles over input AND target."""
    n = 0
    mean = 0.0
    M2 = 0.0
    vmax = -float("inf")
    samples = []
    for i in range(0, len(indices), chunk):
        idx = sorted(int(x) for x in indices[i:i + chunk])
        x = np.concatenate([
            ds_input[idx].astype("float64").ravel(),
            ds_target[idx].astype("float64").ravel(),
        ])
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

        if len(samples) < _PERCENTILE_SAMPLE_CAP:
            remaining = _PERCENTILE_SAMPLE_CAP - len(samples)
            take = x if x.size <= remaining else x[:remaining]
            samples.append(take)

    all_samples = np.concatenate(samples)
    p95 = float(np.percentile(all_samples, 95.0))
    p99 = float(np.percentile(all_samples, 99.0))
    p99_9 = float(np.percentile(all_samples, 99.9))
    std = float(np.sqrt(M2 / n)) if n > 1 else 0.0
    return mean, std, vmax, p95, p99, p99_9


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit normalization statistics.")
    parser.add_argument("--h5", default="ConvLSTM_Dataset_128.h5")
    parser.add_argument("--train-years", nargs=2, type=int, default=None)
    parser.add_argument("--train-typhoons", default=None)
    parser.add_argument("--out", default="configs/normalization_v1.json")
    args = parser.parse_args()

    with h5py.File(args.h5, "r") as f:
        years = f["meta/year"][:]
        tids = f["meta/typhoon_id"][:]

        mask = np.zeros(len(years), dtype=bool)
        if args.train_typhoons is not None:
            p = Path(args.train_typhoons)
            ids = {int(x) for x in (p.read_text().split() if p.exists()
                                    else args.train_typhoons.split(",")) if x.strip()}
            mask = np.isin(tids, list(ids))
        elif args.train_years is not None:
            lo, hi = args.train_years
            mask = (years >= lo) & (years <= hi)
        else:
            raise SystemExit("Provide --train-years or --train-typhoons.")

        train_idx = np.where(mask)[0]
        if len(train_idx) == 0:
            raise RuntimeError("No train samples matched the selection.")

        p_mean, p_std, p_max, p95, p99, p99_9 = _precip_stats(
            f["precip/input"], f["precip/target"], train_idx
        )

        track = f["track"][sorted(train_idx.tolist())].astype("float64")
        track_stats = {}
        for name in _ZSCORE_TRACK:
            j = TRACK_FEATURE_NAMES.index(name)
            vals = track[:, :, j]
            track_stats[name] = {"mean": float(vals.mean()), "std": float(vals.std())}

        terrain_stats = {}
        for k, name in enumerate(_ZSCORE_TERRAIN):
            mean, std, _ = _streaming_mean_std_max(f["terrain"], train_idx, channel=k)
            terrain_stats[name] = {"mean": mean, "std": std}

    out = {
        "version": "1.0",
        "fitted_on": "train events only (input + target precipitation)",
        "train_sample_count": int(len(train_idx)),
        "precipitation": {
            "mean": round(p_mean, 4),
            "std": round(p_std, 4),
            "max": round(p_max, 3),
            "p95": round(p95, 3),
            "p99": round(p99, 3),
            "p99_9": round(p99_9, 3),
        },
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
