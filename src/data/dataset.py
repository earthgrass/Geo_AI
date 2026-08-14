"""Typhoon precipitation dataset for the schema-v2 HDF5.

Reads the v2 layout (precip / terrain / track / meta) and reconstructs the
canonical 12-channel model input ON-THE-FLY:

    channel 0  precipitation   <- /precip/input
    channel 1  center_wind_speed <- /track[:, :, wind_idx]  (broadcast scalar)
    channel 2  center_pressure   <- /track[:, :, pres_idx]  (broadcast scalar)
    channel 3  distance_center   <- static grid geometry
    channel 4  dx                <- static grid geometry
    channel 5  dy                <- static grid geometry
    channel 6  u_move            <- /track[:, :, u_idx]     (broadcast scalar)
    channel 7  v_move            <- /track[:, :, v_idx]     (broadcast scalar)
    channel 8  dem               <- /terrain[0]
    channel 9  dh_dx             <- /terrain[1]
    channel 10 dh_dy             <- /terrain[2]
    channel 11 land_mask         <- /terrain[3]

Leakage safety: metadata requirements are FIELD-SPECIFIC. Requesting a year
split requires `/meta/year`; requesting an event split requires
`/meta/typhoon_id`. If the required field is missing, a RuntimeError is raised
(never a silent fall-back to all samples).
"""

import warnings

import h5py
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from typing import Optional, Dict, List, Tuple

from .splits import assert_no_duplicate_events
from ..config import (
    TRACK_FEATURE_NAMES,
    TERRAIN_CHANNEL_NAMES,
    STATIC_GRID_CHANNELS,
)


def static_grid_channels(grid_size: int):
    """Return (distance_center, dx, dy), each [grid_size, grid_size]."""
    y, x = np.meshgrid(np.arange(grid_size), np.arange(grid_size), indexing="ij")
    c = (grid_size - 1) / 2.0
    dx = (x - c).astype("float32")
    dy = (y - c).astype("float32")
    distance_center = np.sqrt(dx ** 2 + dy ** 2).astype("float32")
    return distance_center, dx, dy


class TyphoonDataset(Dataset):
    """PyTorch Dataset for the schema-v2 typhoon HDF5.

    Args:
        h5_path: Path to the schema-v2 HDF5 file.
        seq_len: Number of input frames (default 11).
        split_years: (min_year, max_year) inclusive; requires /meta/year.
        typhoon_ids: List of typhoon IDs to include; requires /meta/typhoon_id.
        transform: Optional callable applied to (X, Y, meta).
        meta_csv_path: Optional sidecar CSV (fallback for legacy v1 HDF5).
        strict_metadata: If True (default), raise on missing/ inconsistent
            metadata instead of warning.
    """

    def __init__(
        self,
        h5_path: str,
        seq_len: int = 11,
        split_years: Optional[Tuple[int, int]] = None,
        typhoon_ids: Optional[List[int]] = None,
        transform=None,
        meta_csv_path: Optional[str] = None,
        strict_metadata: bool = True,
    ):
        self.h5_path = h5_path
        self.seq_len = seq_len
        self.transform = transform
        self.strict_metadata = strict_metadata

        with h5py.File(self.h5_path, 'r') as f:
            self.is_v2 = "precip" in f
            total_samples = self._total_samples(f)

            # Field-specific metadata resolution.
            years = self._read_field(f, "year", meta_csv_path)
            tids = self._read_field(f, "typhoon_id", meta_csv_path)

        self._years = years
        self._typhoon_ids = tids

        # Validate metadata consistency.
        if years is not None and len(years) != total_samples:
            raise RuntimeError(
                f"metadata year length {len(years)} != sample count {total_samples}."
            )
        if tids is not None and len(tids) != total_samples:
            raise RuntimeError(
                f"metadata typhoon_id length {len(tids)} != sample count {total_samples}."
            )

        # Field-specific filtering (never silently fall back).
        indices = list(range(total_samples))
        if split_years is not None:
            if years is None:
                raise RuntimeError(
                    "split_years requested but /meta/year (and no sidecar 'year') "
                    "was found. Refusing to fall back to all samples."
                )
            min_y, max_y = split_years
            indices = [i for i in indices if min_y <= int(years[i]) <= max_y]
        if typhoon_ids is not None:
            if tids is None:
                raise RuntimeError(
                    "typhoon_ids requested but /meta/typhoon_id (and no sidecar "
                    "'typhoon_id') was found. Refusing to fall back to all samples."
                )
            tidset = set(int(t) for t in typhoon_ids)
            indices = [i for i in indices if int(tids[i]) in tidset]

        self.indices = indices
        print(f"[Dataset] {len(self.indices)} samples loaded "
              f"(split_years={split_years}, typhoon_ids={typhoon_ids})")

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _total_samples(f) -> int:
        if "precip" in f and "input" in f["precip"]:
            return int(f["precip/input"].shape[0])
        if "data" in f:
            return int(f["data"].shape[0])
        raise RuntimeError("HDF5 has neither /precip/input nor /data.")

    def _read_field(self, f, field: str, meta_csv_path: Optional[str]):
        """Read a scalar per-sample field from /meta, falling back to a sidecar."""
        if "meta" in f and field in f["meta"]:
            return np.asarray(f["meta"][field][:])

        df = self._load_sidecar_metadata(meta_csv_path)
        if df is not None and field in df:
            return df[field].to_numpy()
        return None

    def _load_sidecar_metadata(self, meta_csv_path: Optional[str]) -> Optional[pd.DataFrame]:
        candidates = []
        if meta_csv_path is not None:
            candidates.append(Path(meta_csv_path))
        h5 = Path(self.h5_path)
        candidates.extend([
            h5.with_name(f"{h5.stem}_metadata.csv"),
            h5.with_name(f"{h5.stem}_sample_metadata.csv"),
        ])
        for path in candidates:
            if not path.exists():
                continue
            df = pd.read_csv(path)
            if 'sample_idx' in df.columns:
                idx = df['sample_idx'].to_numpy()
                # sample_idx must be exactly the full range.
                expected = set(range(len(df)))
                if set(idx.tolist()) != expected:
                    raise RuntimeError(
                        f"sidecar '{path}' sample_idx is not exactly "
                        f"set(range({len(df)}))."
                    )
                df = df.sort_values('sample_idx').reset_index(drop=True)
            return df
        return None

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        real_idx = self.indices[idx]

        with h5py.File(self.h5_path, 'r') as f:
            if self.is_v2:
                precip_input = f["precip/input"][real_idx]      # [11, H, W]
                precip_target = f["precip/target"][real_idx]    # [1, H, W]
                terrain = f["terrain"][real_idx]                # [4, H, W]
                track = f["track"][real_idx]                    # [11, 6]
            else:
                # Legacy v1 fallback: [12, C, H, W] with C channels already stored.
                sample = f["data"][real_idx]
                X_np = sample[:self.seq_len]
                Y_np = sample[self.seq_len, 0:1]
                X = torch.tensor(X_np, dtype=torch.float32)
                Y = torch.tensor(Y_np, dtype=torch.float32)
                meta = {"P_prev": X[-1, 0:1]}
                if self.transform is not None:
                    X, Y, meta = self.transform(X, Y, meta)
                return X, Y, meta

        X = self._reconstruct_input(precip_input, track, terrain)
        Y = torch.tensor(precip_target, dtype=torch.float32)
        P_prev = torch.tensor(precip_input[-1], dtype=torch.float32).unsqueeze(0)

        meta = {"P_prev": P_prev}
        if self._years is not None:
            meta["year"] = int(self._years[real_idx])
        if self._typhoon_ids is not None:
            meta["typhoon_id"] = int(self._typhoon_ids[real_idx])

        if self.transform is not None:
            X, Y, meta = self.transform(X, Y, meta)

        return X, Y, meta

    def _reconstruct_input(self, precip_input, track, terrain) -> torch.Tensor:
        """Reconstruct the canonical 12-channel input [11, 12, H, W]."""
        K, H, W = precip_input.shape
        distance_center, dx, dy = static_grid_channels(H)

        # Track feature indices (canonical order, see src.config.TRACK_FEATURE_NAMES).
        wind_idx = TRACK_FEATURE_NAMES.index("center_wind_speed")
        pres_idx = TRACK_FEATURE_NAMES.index("center_pressure")
        u_idx = TRACK_FEATURE_NAMES.index("u_move")
        v_idx = TRACK_FEATURE_NAMES.index("v_move")

        X = torch.zeros((K, 12, H, W), dtype=torch.float32)
        X[:, 0] = torch.tensor(precip_input, dtype=torch.float32)
        X[:, 1] = torch.tensor(track[:, wind_idx], dtype=torch.float32)[:, None, None]
        X[:, 2] = torch.tensor(track[:, pres_idx], dtype=torch.float32)[:, None, None]
        X[:, 3] = torch.tensor(distance_center, dtype=torch.float32)[None]
        X[:, 4] = torch.tensor(dx, dtype=torch.float32)[None]
        X[:, 5] = torch.tensor(dy, dtype=torch.float32)[None]
        X[:, 6] = torch.tensor(track[:, u_idx], dtype=torch.float32)[:, None, None]
        X[:, 7] = torch.tensor(track[:, v_idx], dtype=torch.float32)[:, None, None]
        X[:, 8] = torch.tensor(terrain[0], dtype=torch.float32)[None]
        X[:, 9] = torch.tensor(terrain[1], dtype=torch.float32)[None]
        X[:, 10] = torch.tensor(terrain[2], dtype=torch.float32)[None]
        X[:, 11] = torch.tensor(terrain[3], dtype=torch.float32)[None]
        return X

    # ------------------------------------------------------------------
    # Split helpers
    # ------------------------------------------------------------------

    @staticmethod
    def split_by_year(h5_path, train_years=(2014, 2022), val_years=(2023, 2023),
                      test_years=(2024, 2024), **kwargs):
        train = TyphoonDataset(h5_path, split_years=train_years, **kwargs)
        val = TyphoonDataset(h5_path, split_years=val_years, **kwargs)
        test = TyphoonDataset(h5_path, split_years=test_years, **kwargs)
        return train, val, test

    def get_typhoon_ids(self) -> List[int]:
        ids = set()
        if self._typhoon_ids is not None:
            for idx in self.indices:
                ids.add(int(self._typhoon_ids[idx]))
            return sorted(ids)
        with h5py.File(self.h5_path, 'r') as f:
            if 'meta' in f and 'typhoon_id' in f['meta']:
                for idx in self.indices:
                    ids.add(int(f['meta/typhoon_id'][idx]))
        return sorted(ids)
