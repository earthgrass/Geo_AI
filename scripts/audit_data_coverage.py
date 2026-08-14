"""Audit CMA + GPM data coverage WITHOUT building the full dataset.

Reports, per typhoon event, the GPM match status (exact / nearest / missing) at
0.5h grid points, plus how many usable 11->1 windows exist.

Two tolerance regimes are required before freezing the event split:
    A. exact matching:   --match-tolerance-sec 0
    B. diagnostic ±15m:  --match-tolerance-sec 900

Run from repo root:
    python scripts/audit_data_coverage.py --tolerance 0    --label EXACT
    python scripts/audit_data_coverage.py --tolerance 900  --label T15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for p in (SCRIPT_DIR, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from build_paper_dataset import (  # noqa: E402
    parse_best_track,
    build_tif_index,
    lookup_gpm,
    SEQ_LEN,
    INPUT_SEQ_LEN,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit CMA + GPM coverage.")
    parser.add_argument("--cma-dir", default="CMABSTdata")
    parser.add_argument("--tif-dir", default="TIFdata")
    parser.add_argument("--tolerance", type=int, default=0,
                        help="GPM match tolerance in seconds (0 = exact)")
    parser.add_argument("--max-missing", type=int, default=2)
    parser.add_argument("--label", default="EXACT")
    args = parser.parse_args()

    all_tracks = [parse_best_track(str(p)) for p in sorted(Path(args.cma_dir).glob("*.txt"))]
    cma = pd.concat(all_tracks, ignore_index=True) if all_tracks else pd.DataFrame()
    tif_index = build_tif_index(args.tif_dir)

    rows = []
    for typhoon_id, raw in cma.groupby("Typhoon_ID"):
        raw = raw.sort_values("Time").reset_index(drop=True)
        if raw.empty:
            continue
        grid = pd.date_range(
            raw["Time"].min().floor("30min"), raw["Time"].max().ceil("30min"), freq="30min"
        )
        n_grid = len(grid)

        exact = nearest = missing = 0
        present = [False] * n_grid
        for t, ft in enumerate(grid):
            _, actual_ts, offset = lookup_gpm(
                tif_index, typhoon_id, ft.to_pydatetime(), args.tolerance
            )
            if actual_ts is None:
                missing += 1
            elif offset == 0.0:
                exact += 1
                present[t] = True
            else:
                nearest += 1
                present[t] = True

        usable = target_missing = 0
        for i in range(0, n_grid - SEQ_LEN + 1):
            if not present[i + INPUT_SEQ_LEN]:
                target_missing += 1
                continue
            missing_input = sum(1 for t in range(i, i + INPUT_SEQ_LEN) if not present[t])
            if missing_input > args.max_missing:
                continue
            usable += 1

        rows.append({
            "typhoon_id": int(typhoon_id),
            "year": int(grid[0].year),
            "cma_orig_frames": len(raw),
            "grid_frames": n_grid,
            "gpm_frames": len(tif_index.get(typhoon_id, {})),
            "exact_matches": exact,
            "nearest_matches": nearest,
            "missing_frames": missing,
            "usable_windows": usable,
            "target_missing_windows": target_missing,
            "duration_h": round((grid[-1] - grid[0]).total_seconds() / 3600.0, 1),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("[audit] no typhoon events found.")
        return

    out_csv = f"DATA_COVERAGE_{args.label}.csv"
    out_md = f"DATA_COVERAGE_{args.label}_REPORT.md"
    df.to_csv(out_csv, index=False)
    print(f"[audit] wrote {out_csv} ({len(df)} typhoons)")

    total_exact = df["exact_matches"].sum()
    total_nearest = df["nearest_matches"].sum()
    total_missing = df["missing_frames"].sum()
    total_grid = df["grid_frames"].sum()
    total_usable = df["usable_windows"].sum()

    lines = [
        f"# DATA COVERAGE REPORT ({args.label})",
        "",
        f"- Match tolerance: {args.tolerance} s",
        f"- Typhoon events: {len(df)}",
        f"- Total grid frames: {total_grid}",
        f"- Exact matches: {total_exact} ({total_exact / max(total_grid, 1):.1%})",
        f"- Nearest matches: {total_nearest}",
        f"- Missing frames: {total_missing} ({total_missing / max(total_grid, 1):.1%})",
        f"- Usable 11->1 windows: {total_usable}",
        "",
        "| typhoon_id | year | cma_orig | grid | gpm | exact | nearest | missing | usable | target_missing | duration_h |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {int(r['typhoon_id'])} | {int(r['year'])} | {int(r['cma_orig_frames'])} "
            f"| {int(r['grid_frames'])} | {int(r['gpm_frames'])} | {int(r['exact_matches'])} "
            f"| {int(r['nearest_matches'])} | {int(r['missing_frames'])} | {int(r['usable_windows'])} "
            f"| {int(r['target_missing_windows'])} | {r['duration_h']:.1f} |"
        )
    lines.append("")
    lines.append("Inspect the EXACT vs T15 reports together before freezing the event split.")
    Path(out_md).write_text("\n".join(lines), encoding="utf-8")
    print(f"[audit] wrote {out_md}")


if __name__ == "__main__":
    main()
