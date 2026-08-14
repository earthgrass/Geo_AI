"""GPM grid preflight: verify GPM TIF files share a compatible geotransform.

The dataset builder assumes all GPM precipitation rasters share the same CRS,
transform, resolution, width and height. This script scans the GPM metadata and
fails if incompatible grids are detected (rather than silently cropping on a
mixed grid).

Run from repo root:
    python scripts/audit_gpm_grid.py --tif-dir TIFdata
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import rasterio


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit GPM raster grid compatibility.")
    parser.add_argument("--tif-dir", default="TIFdata")
    parser.add_argument("--max-files", type=int, default=200,
                        help="Cap metadata scans (all files share one product grid).")
    args = parser.parse_args()

    tifs = list(Path(args.tif_dir).rglob("*.tif"))[:args.max_files]
    if not tifs:
        print("[audit] no GPM TIF files found.")
        return

    crs_counter = Counter()
    transform_counter = Counter()
    size_counter = Counter()
    nodata_counter = Counter()
    bad = []

    for p in tifs:
        try:
            with rasterio.open(p) as src:
                crs_counter[src.crs.to_string()] += 1
                transform_counter[tuple(src.transform)] += 1
                size_counter[(src.width, src.height)] += 1
                nodata_counter[src.nodata] += 1
        except Exception as exc:  # noqa: BLE001
            bad.append((str(p), repr(exc)))

    print("=" * 62)
    print("GPM GRID PREFLIGHT REPORT")
    print("=" * 62)
    print(f"scanned files: {len(tifs)}")
    print(f"distinct CRS: {len(crs_counter)}")
    for crs, n in crs_counter.most_common(5):
        print(f"  - {crs}: {n}")
    print(f"distinct transforms: {len(transform_counter)}")
    for tf, n in transform_counter.most_common(5):
        print(f"  - {tf}: {n}")
    print(f"distinct (width,height): {len(size_counter)}")
    for s, n in size_counter.most_common(5):
        print(f"  - {s}: {n}")
    print(f"distinct nodata: {len(nodata_counter)}")
    for nd, n in nodata_counter.most_common(5):
        print(f"  - {nd}: {n}")
    if bad:
        print(f"unreadable files: {len(bad)}")
        for p, e in bad[:5]:
            print(f"  - {p}: {e}")

    ok = (
        len(crs_counter) <= 1
        and len(transform_counter) <= 1
        and len(size_counter) <= 1
        and len(bad) == 0
    )
    print("-" * 62)
    print(f"RESULT: {'PASS' if ok else 'FAIL'}")
    print("=" * 62 + "\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
