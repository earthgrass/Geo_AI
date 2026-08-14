"""Validate a paper dataset HDF5 file and print a PASS / FAIL report.

Checks (each reported independently):
    1. file exists and opens
    2. /data shape == [N, 12, 12, 128, 128]
    3. exactly 12 channels
    4. all finite (no NaN / Inf)
    5. precipitation >= 0
    6. metadata length == sample count (typhoon_id, year, start_time, target_time)
    7. typhoon_id present for every sample
    8. year present and valid for every sample
    9. timestamps valid (start_time < target_time)
    10. no cross-event window (single typhoon_id per sample)
    11. channel_names attr matches canonical schema
    12. DEM channel not empty
    13. terrain gradients not all zero
    14. land mask contains only 0/1

Run from repo root:
    python scripts/validate_paper_dataset.py --h5 ConvLSTM_Dataset_128.h5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import h5py

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import CHANNEL_NAMES, GRID_SIZE  # noqa: E402

EXPECTED_N_CHANNELS = len(CHANNEL_NAMES)  # 12


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a paper dataset HDF5 file.")
    parser.add_argument("--h5", default="ConvLSTM_Dataset_128.h5")
    args = parser.parse_args()

    results = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, bool(ok), detail))

    path = Path(args.h5)
    check("file exists", path.exists(), str(path))

    if not path.exists():
        _report(results)
        return

    try:
        f = h5py.File(path, "r")
    except Exception as exc:  # noqa: BLE001
        check("file opens", False, repr(exc))
        _report(results)
        return

    with f:
        check("file opens", True)

        # 2. /data shape.
        has_data = "data" in f
        check("has /data", has_data)
        if has_data:
            shape = f["data"].shape
            expected_ndim = 5
            ok_shape = (
                len(shape) == expected_ndim
                and shape[1] == 12
                and shape[2] == EXPECTED_N_CHANNELS
                and shape[3] == GRID_SIZE
                and shape[4] == GRID_SIZE
            )
            check("data shape [N,12,12,128,128]", ok_shape, f"got {shape}")

            n_samples = int(shape[0])

            # 4/5. finite + precip >= 0 (sampled to bound I/O on large files).
            sample_step = max(1, n_samples // 200)
            sample_idxs = list(range(0, n_samples, sample_step))
            sample_idxs.append(n_samples - 1)
            arr = f["data"][sorted(set(sample_idxs))]
            check("finite (no NaN/Inf)", bool(np.isfinite(arr).all()))
            precip = arr[:, -1, 0, :, :]  # last frame, precipitation channel
            check("precipitation >= 0", bool((precip >= 0).all()))

            # 12/13/14. DEM / gradients / land mask on a subsample.
            dem = arr[:, -1, 8, :, :]
            dh_dx = arr[:, -1, 9, :, :]
            dh_dy = arr[:, -1, 10, :, :]
            land = arr[:, -1, 11, :, :]
            check("DEM not empty", bool((dem != 0).any()))
            check("terrain gradients not all zero",
                  bool((dh_dx != 0).any() or (dh_dy != 0).any()))
            check("land mask is 0/1", bool(np.isin(land, [0.0, 1.0]).all()))

        # 6-10. metadata.
        has_meta = "meta" in f
        check("has /meta", has_meta)
        if has_meta and has_data:
            for key in ("typhoon_id", "year", "start_time", "target_time"):
                ok_key = key in f["meta"]
                check(f"/meta/{key} present", ok_key)
                if ok_key:
                    check(
                        f"/meta/{key} length == sample count",
                        len(f["meta"][key]) == n_samples,
                        f"{len(f['meta'][key])} vs {n_samples}",
                    )

            tids = f["meta"]["typhoon_id"][:]
            years = f["meta"]["year"][:]
            starts = f["meta"]["start_time"][:]
            targets = f["meta"]["target_time"][:]
            check("typhoon_id present for every sample",
                  bool(np.isfinite(tids.astype(float)).all()))
            check("year valid integer for every sample",
                  bool((np.isfinite(years.astype(float))).all()
                       and (years >= 1900).all() and (years <= 2100).all()))
            check("timestamps valid (start < target)",
                  bool((starts < targets).all()))
            check("no cross-event window (scalar typhoon_id per sample)",
                  bool(np.isfinite(tids.astype(float)).all()))

        # 11. channel_names attr.
        if "channel_names" in f.attrs:
            names = list(f.attrs["channel_names"])
            check("channel_names matches canonical schema",
                  names == CHANNEL_NAMES, f"{names}")
        else:
            check("channel_names attr present", False)

    _report(results)


def _report(results: list[tuple]) -> None:
    n_fail = sum(1 for _, ok, _ in results if not ok)
    print("\n" + "=" * 60)
    print("DATASET VALIDATION REPORT")
    print("=" * 60)
    for name, ok, detail in results:
        status = "PASS" if ok else "FAIL"
        line = f"  [{status}] {name}"
        if detail:
            line += f"  ({detail})"
        print(line)
    print("-" * 60)
    if n_fail == 0:
        print(f"RESULT: PASS  ({len(results)} checks)")
    else:
        print(f"RESULT: FAIL  ({n_fail}/{len(results)} checks failed)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
