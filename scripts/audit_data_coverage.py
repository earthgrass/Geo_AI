"""Audit CMA + GPM data coverage WITHOUT building the full dataset.

Scans the CMA best-track archive and the GPM TIF archive, then reports, per
typhoon event, how many frames match, how many usable 11->1 windows exist, and
the expected input-imputation / target-missing rates.

Outputs:
    DATA_COVERAGE.csv           per-typhoon rows
    DATA_COVERAGE_REPORT.md     human-readable summary

This report is the input for designing the final train/val/test EVENT split
(do NOT freeze 2014-2022 / 2023 / 2024 by assumption).

Run from repo root:
    python scripts/audit_data_coverage.py --cma-dir CMABSTdata --tif-dir TIFdata
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
    interpolate_track,
    build_tif_index,
    lookup_gpm,
    SEQ_LEN,
    INPUT_SEQ_LEN,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit CMA + GPM coverage.")
    parser.add_argument("--cma-dir", default="CMABSTdata")
    parser.add_argument("--tif-dir", default="TIFdata")
    parser.add_argument("--match-tolerance-sec", type=int, default=900)
    parser.add_argument("--max-missing", type=int, default=2)
    parser.add_argument("--out-csv", default="DATA_COVERAGE.csv")
    parser.add_argument("--out-md", default="DATA_COVERAGE_REPORT.md")
    args = parser.parse_args()

    all_tracks = [parse_best_track(str(p)) for p in sorted(Path(args.cma_dir).glob("*.txt"))]
    cma = pd.concat(all_tracks, ignore_index=True) if all_tracks else pd.DataFrame()
    tif_index = build_tif_index(args.tif_dir)

    rows = []
    for typhoon_id, group in cma.groupby("Typhoon_ID"):
        n_cma_orig = len(group)
        interp = interpolate_track(group)
        if interp.empty:
            continue
        n_interp = len(interp)

        # Per-frame GPM match status.
        exact = nearest = missing = 0
        present = [False] * n_interp
        for t, (_, row) in enumerate(interp.iterrows()):
            _, offset = lookup_gpm(
                tif_index, typhoon_id, row["Time"].to_pydatetime(),
                args.match_tolerance_sec,
            )
            if offset is None:
                missing += 1
            elif offset == 0.0:
                exact += 1
                present[t] = True
            else:
                nearest += 1
                present[t] = True

        # Usable windows (simulate sliding window, no TIF read).
        usable = 0
        target_missing = 0
        total_imputed = 0
        for i in range(0, n_interp - SEQ_LEN + 1):
            target_ok = present[i + INPUT_SEQ_LEN]
            if not target_ok:
                target_missing += 1
                continue
            missing_input = sum(1 for t in range(i, i + INPUT_SEQ_LEN) if not present[t])
            if missing_input > args.max_missing:
                continue
            usable += 1
            total_imputed += missing_input

        imputation_rate = total_imputed / (usable * INPUT_SEQ_LEN) if usable else 0.0
        duration_h = (interp["Time"].iloc[-1] - interp["Time"].iloc[0]).total_seconds() / 3600.0

        rows.append({
            "typhoon_id": int(typhoon_id),
            "year": int(interp["Time"].iloc[0].year),
            "cma_orig_frames": n_cma_orig,
            "cma_interp_frames": n_interp,
            "gpm_frames": len(tif_index.get(typhoon_id, {})),
            "exact_matches": exact,
            "nearest_matches": nearest,
            "missing_frames": missing,
            "usable_windows": usable,
            "target_missing_windows": target_missing,
            "input_imputation_rate": round(imputation_rate, 4),
            "duration_h": round(duration_h, 1),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("[audit] no typhoon events found.")
        return

    df.to_csv(args.out_csv, index=False)
    print(f"[audit] wrote {args.out_csv} ({len(df)} typhoons)")

    # Markdown report.
    total_exact = df["exact_matches"].sum()
    total_nearest = df["nearest_matches"].sum()
    total_missing = df["missing_frames"].sum()
    total_interp = df["cma_interp_frames"].sum()
    total_usable = df["usable_windows"].sum()

    lines = [
        "# DATA COVERAGE REPORT",
        "",
        f"- Typhoon events: {len(df)}",
        f"- Total interpolated frames: {total_interp}",
        f"- Exact GPM matches: {total_exact} ({total_exact / max(total_interp, 1):.1%})",
        f"- Nearest GPM matches: {total_nearest}",
        f"- Missing GPM frames: {total_missing} ({total_missing / max(total_interp, 1):.1%})",
        f"- Usable 11->1 windows: {total_usable}",
        f"- Overall input-imputation rate: {df['input_imputation_rate'].mean():.2%}",
        "",
        "## Per-typhoon summary",
        "",
        "| typhoon_id | year | cma_orig | cma_interp | gpm | exact | nearest | missing | usable | target_missing | impute_rate | duration_h |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {int(r['typhoon_id'])} | {int(r['year'])} | {int(r['cma_orig_frames'])} "
            f"| {int(r['cma_interp_frames'])} | {int(r['gpm_frames'])} | {int(r['exact_matches'])} "
            f"| {int(r['nearest_matches'])} | {int(r['missing_frames'])} | {int(r['usable_windows'])} "
            f"| {int(r['target_missing_windows'])} | {r['input_imputation_rate']:.3f} "
            f"| {r['duration_h']:.1f} |"
        )
    lines.append("")
    lines.append("## Event-split guidance")
    lines.append("")
    lines.append("Do NOT freeze the train/val/test split by year assumption. Use this ")
    lines.append("per-typhoon table to choose an event-level split that balances years, ")
    lines.append("event counts, and usable windows across train/val/test.")

    Path(args.out_md).write_text("\n".join(lines), encoding="utf-8")
    print(f"[audit] wrote {args.out_md}")


if __name__ == "__main__":
    main()
