"""Typhoon precipitation dataset with HDF5 backend.

Leakage-safe by construction:
  * Supports temporal (year) and per-typhoon (event) splitting.
  * When a split is requested but per-sample metadata (typhoon_id / year) is
    unavailable, the dataset raises by default instead of silently falling back
    to "all samples" (which would reintroduce train/test leakage).
  * Validates that the metadata sidecar is consistent with the HDF5 sample count
    and that sample indices are unique and complete.
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


class TyphoonDataset(Dataset):
    """PyTorch Dataset for typhoon spatiotemporal precipitation data.

    Expected HDF5 structure:
        /data    [N, seq_len + 1, C, H, W]  float32
        /meta/typhoon_id  [N]  int        (recommended, for event splitting)
        /meta/year        [N]  int        (recommended, for temporal splitting)

    Args:
        h5_path: Path to HDF5 dataset file.
        seq_len: Number of input frames (default 11). Total frames per sample
                 = seq_len + 1 (last frame is the target).
        precip_channel_idx: Index of precipitation channel (default 0).
        split_years: (min_year, max_year) inclusive. Only samples within this
                     year range are included.
        typhoon_ids: List of typhoon IDs to include. If None, all included.
        transform: Optional callable applied to (X, Y, meta).
        load_into_memory: If True, load the whole dataset into RAM.
        meta_csv_path: Optional explicit path to a sidecar metadata CSV.
        strict_metadata: If True (default), raise RuntimeError when a split is
            requested but per-sample metadata is unavailable, instead of
            silently using all samples.
    """

    def __init__(
        self,
        h5_path: str,
        seq_len: int = 11,
        precip_channel_idx: int = 0,
        split_years: Optional[Tuple[int, int]] = None,
        typhoon_ids: Optional[List[int]] = None,
        transform=None,
        load_into_memory: bool = False,
        meta_csv_path: Optional[str] = None,
        strict_metadata: bool = True,
    ):
        self.h5_path = h5_path
        self.seq_len = seq_len
        self.precip_channel_idx = precip_channel_idx
        self.transform = transform
        self.strict_metadata = strict_metadata

        # Resolve per-sample metadata (year, typhoon_id). Returns (None, None)
        # when neither the HDF5 /meta group nor a sidecar CSV is available.
        self._years, self._typhoon_ids = self._resolve_sample_metadata(meta_csv_path)

        with h5py.File(self.h5_path, 'r') as f:
            total_samples = f['data'].shape[0]
            self._data = None
            if load_into_memory:
                self._data = f['data'][:]

        # Validate metadata consistency when present.
        if self._years is not None or self._typhoon_ids is not None:
            self._validate_metadata(total_samples)

        # Filter indices by year / typhoon. Never silently fall back to "all".
        indices = list(range(total_samples))
        needs_filter = split_years is not None or typhoon_ids is not None
        if needs_filter and self._years is None and self._typhoon_ids is None:
            msg = (
                f"split requested (split_years={split_years}, typhoon_ids={typhoon_ids}) "
                f"but no per-sample metadata (typhoon_id / year) was found in "
                f"'{self.h5_path}' (/meta group) or a sidecar CSV. Falling back "
                f"to all samples would silently leak train/test typhoon events."
            )
            if self.strict_metadata:
                raise RuntimeError(msg)
            # Non-strict path is explicitly UNSAFE and must never be silent.
            warnings.warn(
                msg + " Proceeding with ALL samples (UNSAFE for paper experiments).",
                RuntimeWarning,
            )

        if needs_filter:
            if split_years is not None and self._years is not None:
                min_y, max_y = split_years
                indices = [i for i in indices if min_y <= int(self._years[i]) <= max_y]
            if typhoon_ids is not None and self._typhoon_ids is not None:
                tids = set(int(t) for t in typhoon_ids)
                indices = [i for i in indices if int(self._typhoon_ids[i]) in tids]

        self.indices = indices

        print(f"[Dataset] {len(self.indices)} samples loaded "
              f"(split_years={split_years}, typhoon_ids={typhoon_ids})")

    # ------------------------------------------------------------------
    # Metadata resolution + validation
    # ------------------------------------------------------------------

    def _resolve_sample_metadata(
        self,
        meta_csv_path: Optional[str],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return (years, typhoon_ids) arrays, or (None, None) if unavailable.

        Preference order: HDF5 /meta group > explicit sidecar CSV > auto-detected
        sidecar CSV (``<h5stem>_metadata.csv`` / ``<h5stem>_sample_metadata.csv``).
        """
        # 1. HDF5 /meta group.
        try:
            with h5py.File(self.h5_path, 'r') as f:
                if 'meta' in f:
                    years = np.asarray(f['meta/year'][:]) if 'year' in f['meta'] else None
                    tids = np.asarray(f['meta/typhoon_id'][:]) if 'typhoon_id' in f['meta'] else None
                    return years, tids
        except Exception:
            pass  # fall through to sidecar

        # 2. Sidecar CSV.
        df = self._load_sidecar_metadata(meta_csv_path)
        if df is not None:
            years = df['year'].to_numpy() if 'year' in df else None
            tids = df['typhoon_id'].to_numpy() if 'typhoon_id' in df else None
            return years, tids

        return None, None

    def _validate_metadata(self, total_samples: int) -> None:
        """Validate metadata consistency; raise RuntimeError on mismatch."""
        if self._years is not None and len(self._years) != total_samples:
            raise RuntimeError(
                f"metadata year array length {len(self._years)} != HDF5 sample "
                f"count {total_samples}. Rebuild the sidecar metadata."
            )
        if self._typhoon_ids is not None and len(self._typhoon_ids) != total_samples:
            raise RuntimeError(
                f"metadata typhoon_id array length {len(self._typhoon_ids)} != "
                f"HDF5 sample count {total_samples}. Rebuild the sidecar metadata."
            )

        if self._years is not None:
            if not np.issubdtype(self._years.dtype, np.integer) and not np.issubdtype(
                self._years.dtype, np.floating
            ):
                raise RuntimeError(f"metadata year dtype {self._years.dtype} is not numeric.")
            if np.isnan(self._years.astype(float)).any():
                raise RuntimeError("metadata contains NaN year values.")

        if self._typhoon_ids is not None:
            if self._typhoon_ids.dtype == object or np.isnan(
                self._typhoon_ids.astype(float)
            ).any():
                raise RuntimeError("metadata typhoon_id contains missing/NaN values.")

    def _load_sidecar_metadata(
        self,
        meta_csv_path: Optional[str],
    ) -> Optional[pd.DataFrame]:
        candidates = []
        if meta_csv_path is not None:
            candidates.append(Path(meta_csv_path))
        h5 = Path(self.h5_path)
        candidates.extend([
            h5.with_name(f"{h5.stem}_metadata.csv"),
            h5.with_name(f"{h5.stem}_sample_metadata.csv"),
        ])
        for path in candidates:
            if path.exists():
                df = pd.read_csv(path)
                if 'sample_idx' in df.columns:
                    # Validate sample_idx uniqueness + coverage before sorting.
                    idx = df['sample_idx'].to_numpy()
                    if len(set(idx.tolist())) != len(idx):
                        raise RuntimeError(
                            f"sidecar metadata '{path}' has duplicate sample_idx values."
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
            if self._data is not None:
                sample = self._data[real_idx]
            else:
                sample = f['data'][real_idx]

            X_np = sample[:self.seq_len]              # [seq_len, C, H, W]
            Y_np = sample[self.seq_len,
                          self.precip_channel_idx:
                          self.precip_channel_idx + 1]  # [1, H, W]
            P_prev_np = sample[self.seq_len - 1,
                               self.precip_channel_idx:
                               self.precip_channel_idx + 1]  # [1, H, W]

            meta = {'P_prev': torch.tensor(P_prev_np, dtype=torch.float32)}
            if 'meta' in f:
                for key in f['meta'].keys():
                    meta[key] = int(f['meta'][key][real_idx])
            elif self._years is not None or self._typhoon_ids is not None:
                if self._years is not None:
                    meta['year'] = int(self._years[real_idx])
                if self._typhoon_ids is not None:
                    meta['typhoon_id'] = int(self._typhoon_ids[real_idx])

        X = torch.tensor(X_np, dtype=torch.float32)
        Y = torch.tensor(Y_np, dtype=torch.float32)

        if self.transform is not None:
            X, Y, meta = self.transform(X, Y, meta)

        return X, Y, meta

    # ------------------------------------------------------------------
    # Split helpers
    # ------------------------------------------------------------------

    @staticmethod
    def split_by_year(
        h5_path: str,
        train_years: Tuple[int, int] = (2014, 2022),
        val_years: Tuple[int, int] = (2023, 2023),
        test_years: Tuple[int, int] = (2024, 2024),
        **kwargs,
    ) -> Tuple["TyphoonDataset", "TyphoonDataset", "TyphoonDataset"]:
        """Create train/val/test splits by year (no leakage)."""
        train = TyphoonDataset(h5_path, split_years=train_years, **kwargs)
        val = TyphoonDataset(h5_path, split_years=val_years, **kwargs)
        test = TyphoonDataset(h5_path, split_years=test_years, **kwargs)
        return train, val, test

    def get_typhoon_ids(self) -> List[int]:
        """Return unique typhoon IDs in this dataset split."""
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
