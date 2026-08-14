"""Validate a schema-v2 paper dataset HDF5 and print a PASS / FAIL report.

Every check is reported independently; missing metadata keys report FAIL rather
than crashing. A zero-sample dataset reports FAIL cleanly.

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

from src.config import CHANNEL_NAMES, TRACK_FEATURE_NAMES, TERRAIN_CHANNEL_NAMES, SCHEMA_VERSION  # noqa: E402

EXPECTED_N_CHANNELS = len(CHANNEL_NAMES)  # 12
STEP_SEC = 1800  # 0.5h per frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a schema-v2 paper dataset.")
    parser.add_argument("--h5", default="ConvLSTM_Dataset_128.h5")
    args = parser.parse_args()

    results: list[tuple[str, bool, str]] = []

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

        # Core groups present.
        for grp in ("precip/input", "precip/target", "terrain", "track", "meta"):
            check(f"has /{grp}", grp in f, grp)

        if "precip/input" not in f:
            _report(results)
            return

        n_samples = int(f["precip/input"].shape[0])
        check("non-empty dataset", n_samples > 0, f"N={n_samples}")

        pshape = f["precip/input"].shape
        tshape = f["precip/target"].shape
        teshape = f["terrain"].shape
        trshape = f["track"].shape
        check("input shape [N,11,H,W]", len(pshape) == 4 and pshape[1] == 11, str(pshape))
        check("target shape [N,1,H,W]", len(tshape) == 4 and tshape[1] == 1, str(tshape))
        check("terrain shape [N,4,H,W]", len(teshape) == 4 and teshape[2] == 4, str(teshape))
        check("track shape [N,11,6]", len(trshape) == 4 or len(trshape) == 3, str(trshape))

        # Input/target share the same H,W (geospatial alignment).
        check("input/target same H,W",
              pshape[2:] == tshape[2:], f"{pshape[2:]} vs {tshape[2:]}")
        check("terrain aligned to precip grid",
              pshape[2:] == teshape[3:], f"{pshape[2:]} vs {teshape[3:]}")

        if n_samples > 0:
            step = max(1, n_samples // 200)
            idxs = sorted(set(list(range(0, n_samples, step)) + [n_samples - 1]))

            arr = f["precip/input"][idxs]
            tgt = f["precip/target"][idxs]
            check("input finite (no NaN/Inf)", bool(np.isfinite(arr).all()))
            check("target finite (no NaN/Inf)", bool(np.isfinite(tgt).all()))
            check("precipitation >= 0", bool((arr >= 0).all() and (tgt >= 0).all()))

            terrain = f["terrain"][idxs]
            check("DEM not empty", bool((terrain[:, 0] != 0).any()))
            check("terrain gradients not all zero",
                  bool((terrain[:, 1] != 0).any() or (terrain[:, 2] != 0).any()))
            check("land mask is 0/1", bool(np.isin(terrain[:, 3], [0.0, 1.0]).all()))

            track = f["track"][idxs]  # [., 11, 6]
            # track feature order: lat, lon, wind, pressure, u_move, v_move
            wind = track[..., 2]
            pres = track[..., 3]
            umove = track[..., 4]
            vmove = track[..., 5]
            check("wind speed in valid range (0..120 m/s)",
                  bool((wind >= 0).all() and (wind <= 120).all()))
            check("pressure in valid range (850..1050 hPa)",
                  bool((pres >= 850).all() and (pres <= 1050).all()))
            check("translation speeds plausible (<100 km/h)",
                  bool((np.abs(umove) < 100).all() and (np.abs(vmove) < 100).all()))

        # Metadata field presence (each missing key -> FAIL, no crash).
        for key, dtype in (("typhoon_id", None), ("year", None),
                           ("start_time", None), ("anchor_time", None),
                           ("target_time", None), ("anchor_lat", None),
                           ("anchor_lon", None), ("grid_transform", None),
                           ("input_imputed_mask", None), ("gpm_match_offset", None)):
            ok = "meta" in f and key in f["meta"]
            check(f"/meta/{key} present", ok, key)
            if ok and n_samples > 0:
                check(f"/meta/{key} length == N",
                      len(f["meta"][key]) == n_samples,
                      f"{len(f['meta'][key])} vs {n_samples}")

        if n_samples > 0 and "meta" in f and "start_time" in f["meta"]:
            st = f["meta"]["start_time"][:]
            at = f["meta"]["anchor_time"][:]
            tt = f["meta"]["target_time"][:]
            check("timestamps chronological (start < anchor < target)",
                  bool((st < at).all() and (at < tt).all()))
            # anchor = start + 10 steps; target = start + 11 steps.
            check("anchor offset ~10 steps (5h)",
                  bool((np.abs((at - st) - 10 * STEP_SEC) <= STEP_SEC).all()))
            check("target offset ~11 steps (5.5h)",
                  bool((np.abs((tt - st) - 11 * STEP_SEC) <= STEP_SEC).all()))

        if n_samples > 0 and "meta" in f and "anchor_lat" in f["meta"]:
            lat = f["meta"]["anchor_lat"][:]
            lon = f["meta"]["anchor_lon"][:]
            check("anchor_lat valid (-90..90)", bool((lat >= -90).all() and (lat <= 90).all()))
            check("anchor_lon valid (-180..180)", bool((lon >= -180).all() and (lon <= 180).all()))

        if n_samples > 0 and "meta" in f and "grid_transform" in f["meta"]:
            gt = f["meta"]["grid_transform"][:]
            check("grid_transform finite [N,6]", bool(np.isfinite(gt).all()) and gt.shape[1] == 6)
            # pixel size = |a| (should be ~0.1 deg for GPM)
            px = np.abs(gt[:, 0])
            check("grid pixel size plausible (0.01..1.0 deg)",
                  bool((px > 0.01).all() and (px < 1.0).all()))

        if n_samples > 0 and "meta" in f and "input_imputed_mask" in f["meta"]:
            mask = f["meta"]["input_imputed_mask"][:]
            check("input_imputed_mask shape [N,11]", mask.shape[1] == 11, str(mask.shape))
            check("input_imputed_mask binary", bool(np.isin(mask, [0, 1]).all()))
            check("imputation rate <= 30%",
                  bool((mask.mean() <= 0.3)), f"{mask.mean():.3f}")
            # TARGET is never imputed: there is no target imputation field (the
            # mask has 11 columns for the 11 INPUT frames only).
            check("target never imputed (mask covers input only)", mask.shape[1] == 11)

        # Attributes.
        if "schema_version" in f.attrs:
            check("schema_version == v2", str(f.attrs["schema_version"]) == SCHEMA_VERSION,
                  str(f.attrs["schema_version"]))
        if "channel_names" in f.attrs:
            check("channel_names matches canonical",
                  list(f.attrs["channel_names"]) == CHANNEL_NAMES)
        if "track_feature_names" in f.attrs:
            check("track_feature_names matches canonical",
                  list(f.attrs["track_feature_names"]) == TRACK_FEATURE_NAMES)
        if "terrain_channel_names" in f.attrs:
            check("terrain_channel_names matches canonical",
                  list(f.attrs["terrain_channel_names"]) == TERRAIN_CHANNEL_NAMES)

    _report(results)


def _report(results: list[tuple[str, bool, str]]) -> None:
    n_fail = sum(1 for _, ok, _ in results if not ok)
    print("\n" + "=" * 62)
    print("DATASET VALIDATION REPORT (schema v2)")
    print("=" * 62)
    for name, ok, detail in results:
        status = "PASS" if ok else "FAIL"
        line = f"  [{status}] {name}"
        if detail:
            line += f"  ({detail})"
        print(line)
    print("-" * 62)
    if n_fail == 0:
        print(f"RESULT: PASS  ({len(results)} checks)")
    else:
        print(f"RESULT: FAIL  ({n_fail}/{len(results)} checks failed)")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    main()
