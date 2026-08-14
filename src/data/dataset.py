"""Typhoon precipitation dataset with HDF5 backend.

Key improvements over competition version (step2.1):
    1. Supports temporal split (by typhoon year) — no data leakage
    2. Supports leave-one-typhoon-out split
    3. Returns metadata (P_prev for residual loss, typhoon ID, year)
    4. Configurable precipitation channel index
    5. Backward-compatible with existing HDF5 datasets
"""

import h5py
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from typing import Optional, Dict, List, Tuple


class TyphoonDataset(Dataset):
    """PyTorch Dataset for typhoon spatiotemporal precipitation data.

    Expected HDF5 structure:
        /data    [N, seq_len, C, H, W]  float32
        /meta/typhoon_id  [N]  int        (optional, for splitting)
        /meta/year       [N]  int        (optional, for splitting)

    Args:
        h5_path: Path to HDF5 dataset file.
        seq_len: Number of input frames (default 11). Total frames per
                 sample = seq_len + 1 (last frame is target).
        precip_channel_idx: Index of precipitation channel (default 0).
        split_years: (min_year, max_year) inclusive range. Only samples
                     within this year range are included.
        typhoon_ids: List of typhoon IDs to include. If None, all included.
        transform: Optional callable to apply to (X, Y, meta).
        load_into_memory: If True, load entire dataset into RAM.
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
    ):
        self.h5_path = h5_path
        self.seq_len = seq_len
        self.precip_channel_idx = precip_channel_idx
        self.transform = transform
        self._meta_df = self._load_sidecar_metadata(meta_csv_path)

        # Read metadata and filter valid indices
        with h5py.File(self.h5_path, 'r') as f:
            total_samples = f['data'].shape[0]
            all_indices = list(range(total_samples))

            # Filter by year (for temporal split)
            if split_years is not None:
                if 'meta' in f and 'year' in f['meta']:
                    years = f['meta/year'][:]
                    min_y, max_y = split_years
                    all_indices = [
                        i for i in all_indices
                        if min_y <= years[i] <= max_y
                    ]
                elif self._meta_df is not None and 'year' in self._meta_df:
                    min_y, max_y = split_years
                    years = self._meta_df['year'].to_numpy()
                    all_indices = [
                        i for i in all_indices
                        if min_y <= years[i] <= max_y
                    ]
                else:
                    print("[WARNING] split_years requested but no "
                          "'meta/year' in HDF5 or sidecar CSV. Using all samples.")

            # Filter by typhoon ID (for leave-one-out)
            if typhoon_ids is not None:
                if 'meta' in f and 'typhoon_id' in f['meta']:
                    tids = set(typhoon_ids)
                    ids = f['meta/typhoon_id'][:]
                    all_indices = [
                        i for i in all_indices
                        if ids[i] in tids
                    ]
                elif self._meta_df is not None and 'typhoon_id' in self._meta_df:
                    tids = set(typhoon_ids)
                    ids = self._meta_df['typhoon_id'].to_numpy()
                    all_indices = [
                        i for i in all_indices
                        if ids[i] in tids
                    ]
                else:
                    print("[WARNING] typhoon_ids requested but no "
                          "'meta/typhoon_id' in HDF5 or sidecar CSV. Using all samples.")

            self.indices = all_indices

            # Optionally load all data into memory for faster access
            if load_into_memory and len(self.indices) > 0:
                self._data = f['data'][:]
            else:
                self._data = None

        print(f"[Dataset] {len(self.indices)} samples loaded "
              f"(years: {split_years}, typhoons: {typhoon_ids})"
              if split_years or typhoon_ids
              else f"[Dataset] {len(self.indices)} samples loaded")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Returns (X, Y, meta).

        X: [seq_len, C, H, W] — input sequence
        Y: [1, H, W] — target precipitation (frame seq_len+1)
        meta: dict with:
            - 'P_prev': [1, H, W] last input frame precipitation
            - 'typhoon_id': int (if available in HDF5)
            - 'year': int (if available in HDF5)
        """
        real_idx = self.indices[idx]

        with h5py.File(self.h5_path, 'r') as f:
            if self._data is not None:
                sample = self._data[real_idx]  # [total_frames, C, H, W]
            else:
                sample = f['data'][real_idx]

            # Split into input sequence and target
            X_np = sample[:self.seq_len]              # [seq_len, C, H, W]
            Y_np = sample[self.seq_len,
                          self.precip_channel_idx:
                          self.precip_channel_idx + 1]  # [1, H, W]

            # Previous frame precipitation (for residual loss and smoothness)
            P_prev_np = sample[self.seq_len - 1,
                               self.precip_channel_idx:
                               self.precip_channel_idx + 1]  # [1, H, W]

            X = torch.tensor(X_np, dtype=torch.float32)
            Y = torch.tensor(Y_np, dtype=torch.float32)
            P_prev = torch.tensor(P_prev_np, dtype=torch.float32)

            # Metadata
            meta = {'P_prev': P_prev}
            if 'meta' in f:
                for key in f['meta'].keys():
                    meta[key] = int(f['meta'][key][real_idx])
            elif self._meta_df is not None:
                row = self._meta_df.iloc[real_idx]
                for key, value in row.items():
                    if key == 'sample_idx':
                        continue
                    meta[key] = value.item() if hasattr(value, 'item') else value
            else:
                meta['typhoon_id'] = -1
                meta['year'] = -1

        if self.transform is not None:
            X, Y, meta = self.transform(X, Y, meta)

        return X, Y, meta

    @staticmethod
    def split_by_year(
        h5_path: str,
        train_years: Tuple[int, int] = (2014, 2022),
        val_years: Tuple[int, int] = (2023, 2023),
        test_years: Tuple[int, int] = (2024, 2024),
        **kwargs,
    ) -> Tuple["TyphoonDataset", "TyphoonDataset", "TyphoonDataset"]:
        """Create train/val/test splits by year.

        Recommended split (from Summary document):
            Train: 2014-2022
            Val:   2023
            Test:  2024

        This prevents data leakage since frames from the same typhoon
        cannot appear in both train and test sets.
        """
        train = TyphoonDataset(h5_path, split_years=train_years, **kwargs)
        val = TyphoonDataset(h5_path, split_years=val_years, **kwargs)
        test = TyphoonDataset(h5_path, split_years=test_years, **kwargs)
        return train, val, test

    def get_typhoon_ids(self) -> List[int]:
        """Return unique typhoon IDs in this dataset split."""
        ids = set()
        with h5py.File(self.h5_path, 'r') as f:
            if 'meta' in f and 'typhoon_id' in f['meta']:
                for idx in self.indices:
                    ids.add(int(f['meta/typhoon_id'][idx]))
            elif self._meta_df is not None and 'typhoon_id' in self._meta_df:
                for idx in self.indices:
                    ids.add(int(self._meta_df.iloc[idx]['typhoon_id']))
        return sorted(ids)

    def _load_sidecar_metadata(
        self,
        meta_csv_path: Optional[str],
    ) -> Optional[pd.DataFrame]:
        """Load sample-level metadata CSV when HDF5 lacks /meta."""
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
                    df = df.sort_values('sample_idx').reset_index(drop=True)
                return df
        return None
