"""Promote GPU-run outputs into the paper-grade tracked result tree.

Reads from default source roots:

    outputs/backbone_gate/<id>_seed<N>/
    outputs/axis_i/<id>_seed<N>/
    outputs/axis_ii_c1/<id>_seed<N>/

(overridable via ``--source-dir``) and writes a canonical result tree at
``results/<canonical_name>_seed<N>/``, where ``canonical_name`` is the
frozen name from the spec:

    I0_persistence_seed42
    I1_plain_convlstm_seed42
    I2_resconvlstm_seed42_v2      (v1 -> v2 re-evaluation)
    B1_trajgru_seed42
    I3_resconvlstm_cma_seed42
    I4_static_terrain_seed42
    I5_terrain_geometry_seed42
    P1_smooth_seed42
    P2_extreme_seed42
    P3_smooth_extreme_seed42

Only paper-grade assets are copied into the canonical tree:

    manifest.json
    result_v2.json
    metrics_v2.csv            (generated from result_v2.json if absent)
    validation.md             (generated from result_v2.json if absent)
    history.json              (if present)
    resolved config snapshot  (the YAML/JSON used for this run)

The following are NEVER copied into ``results/``:

    *.pth  (checkpoint weights)
    TensorBoard event files
    *.log / *.tmp / *.swp
    dataset copies (``*.h5``, ``*.npz``, ``*.npy``, ``*.tif``)
    test outputs (anything under a path containing ``/test/``)

The script:

1. NEVER silently overwrites a canonical target directory. If the
   target already exists the run ABORTS unless ``--force-rewrite`` is
   passed.
2. Verifies the fingerprint contract on each source:
       protocol_id == "evaluation_v2"
       split        == "val"
       test_status  == "SEALED"
       smoke        is False
       n_events, n_windows
3. Verifies the I5/P0 same-artifact identity whenever both I5 and P0
   are encountered in the source roots.
4. Verifies every required file is present (manifest + result_v2).
   If missing, ABORTS with a precise error.
5. Verifies the SHA256 metadata recorded in the manifest matches the
   on-disk file's SHA256 (where the source file exists). Differences
   are recorded but do not block archival when the source file is
   intentionally absent (e.g. config snapshot was overridden on disk).
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.evaluator import PROTOCOL_ID  # noqa: E402
from src.evaluation.reporting import (  # noqa: E402
    write_v2_csv,
    write_v2_markdown,
)


# ---------------------------------------------------------------------------
# Canonical mapping: experiment dir name (under outputs/...) -> results/<name>/
# ---------------------------------------------------------------------------

CANONICAL_DIRS: Dict[str, str] = {
    "I0_persistence_seed42":             "I0_persistence_seed42",
    "I1_plain_convlstm_seed42":          "I1_plain_convlstm_seed42",
    "I2_resconvlstm_seed42":             "I2_resconvlstm_seed42_v2",
    "I2_resconvlstm_seed42_v2":          "I2_resconvlstm_seed42_v2",
    "B1_trajgru_seed42":                 "B1_trajgru_seed42",
    "I3_resconvlstm_cma_seed42":         "I3_resconvlstm_cma_seed42",
    "I4_static_terrain_seed42":          "I4_static_terrain_seed42",
    "I5_terrain_geometry_seed42":        "I5_terrain_geometry_seed42",
    "P1_smooth_seed42":                  "P1_smooth_seed42",
    "P1_resconvlstm_smooth_seed42":      "P1_smooth_seed42",
    "P2_extreme_seed42":                 "P2_extreme_seed42",
    "P2_resconvlstm_extreme_seed42":     "P2_extreme_seed42",
    "P3_smooth_extreme_seed42":          "P3_smooth_extreme_seed42",
    "P3_resconvlstm_smooth_extreme_seed42": "P3_smooth_extreme_seed42",
}


# Filenames we never copy into results/
FORBIDDEN_PATTERNS: Tuple[str, ...] = (
    "*.pth", "*.pt", "*.ckpt",
    "*.h5", "*.npz", "*.npy", "*.tif",
    "events.out.tfevents.*", "*.tfevents.*",
    "*.log", "*.tmp", "*.swp", "*.swo",
    "*~", "*.bak",
)

# Required (or generated) paper-grade assets per canonical directory.
REQUIRED_ASSETS: Tuple[str, ...] = (
    "manifest.json",
    "result_v2.json",
    "metrics_v2.csv",
    "validation.md",
)
OPTIONAL_ASSETS: Tuple[str, ...] = (
    "history.json",
    "config.yaml",
    "config_snapshot.yaml",
    "config.json",
    "experiment_summary.json",
    "run_args.json",
)

# Strict fingerprint contract.
EXPECTED_PROTOCOL_ID = PROTOCOL_ID  # "evaluation_v2"
EXPECTED_SPLIT = "val"
EXPECTED_TEST_STATUS = "SEALED"
EXPECTED_SMOKE = False
EXPECTED_N_EVENTS = 7
EXPECTED_N_WINDOWS = 1266


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class ArchiveError(AssertionError):
    """Raised when an artifact fails the contract. NO BYPASS."""


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_forbidden(path: Path) -> bool:
    n = path.name
    for pat in FORBIDDEN_PATTERNS:
        if fnmatch.fnmatchcase(n, pat):
            return True
    if "/test/" in str(path).replace("\\", "/"):
        return True
    if path.suffix.lower() == ".pth":
        return True
    return False


def _looks_like_checkpoint(path: Path) -> bool:
    return path.suffix.lower() in {".pth", ".pt", ".ckpt"}


# ---------------------------------------------------------------------------
# Contract validation
# ---------------------------------------------------------------------------

def _validate_manifest(m: Dict, manifest_path: Path) -> None:
    def _check(key: str, expected) -> None:
        if key not in m:
            raise ArchiveError(
                f"{manifest_path}: missing required manifest field '{key}'.")
        if m[key] != expected:
            raise ArchiveError(
                f"{manifest_path}: manifest field '{key}' must equal "
                f"{expected!r}; got {m[key]!r}.")

    for key in ("experiment_id", "alias_ids", "git_commit",
                "config_sha256", "dataset_sha256", "split_sha256",
                "normalization_sha256", "checkpoint_sha256",
                "protocol_id", "test_status", "split", "smoke"):
        if key not in m:
            raise ArchiveError(
                f"{manifest_path}: missing required manifest field '{key}'.")

    _check("protocol_id", EXPECTED_PROTOCOL_ID)
    _check("split", EXPECTED_SPLIT)
    _check("test_status", EXPECTED_TEST_STATUS)
    _check("smoke", EXPECTED_SMOKE)


def _validate_result_v2(r: Dict, source_dir: Path) -> None:
    if r.get("protocol_id") != EXPECTED_PROTOCOL_ID:
        raise ArchiveError(
            f"{source_dir}/result_v2.json: protocol_id must equal "
            f"{EXPECTED_PROTOCOL_ID!r}; got {r.get('protocol_id')!r}.")
    if r.get("split") != EXPECTED_SPLIT:
        raise ArchiveError(
            f"{source_dir}/result_v2.json: split must equal "
            f"{EXPECTED_SPLIT!r}; got {r.get('split')!r}.")
    if r.get("test_status") != EXPECTED_TEST_STATUS:
        raise ArchiveError(
            f"{source_dir}/result_v2.json: test_status must equal "
            f"{EXPECTED_TEST_STATUS!r}; got {r.get('test_status')!r}.")
    if r.get("n_events") != EXPECTED_N_EVENTS:
        raise ArchiveError(
            f"{source_dir}/result_v2.json: n_events must equal "
            f"{EXPECTED_N_EVENTS}; got {r.get('n_events')}.")
    if r.get("n_windows") != EXPECTED_N_WINDOWS:
        raise ArchiveError(
            f"{source_dir}/result_v2.json: n_windows must equal "
            f"{EXPECTED_N_WINDOWS}; got {r.get('n_windows')}.")


def _unwrap_v2_payload(payload: Dict, source: Path) -> Tuple[Dict, Dict]:
    """Unwrap the v2 result JSON written by ``reporting.write_v2_json``.

    The writer produces ``{"model": str, "result": <evaluator result>}``.
    Returns ``(inner_result, wrapper_payload)``. Fails fast if the wrapper
    is missing or malformed; legacy un-wrapped format is NOT accepted.

    The original wrapper bytes are NEVER rewritten — the archiver preserves
    the GPU-written ``result_v2.json`` byte-for-byte on copy. This helper
    only parses; it does not modify the source file.
    """
    if not isinstance(payload, dict):
        raise ArchiveError(f"{source}: payload is not a JSON object.")
    if "result" not in payload:
        raise ArchiveError(
            f"{source}: missing required wrapper key 'result'. "
            f"This file must be written by "
            f"src.evaluation.reporting.write_v2_json (which emits a "
            f"{{'model': ..., 'result': ...}} wrapper). Legacy "
            f"un-wrapped format (top-level protocol_id/split/n_events) "
            f"is NOT accepted by the archiver."
        )
    inner = payload["result"]
    if not isinstance(inner, dict):
        raise ArchiveError(
            f"{source}: payload['result'] is not a JSON object."
        )
    return inner, payload


# ---------------------------------------------------------------------------
# Canonical-name resolution
# ---------------------------------------------------------------------------

def _dir_key(source_dir: Path) -> str:
    """Strip a trailing _seed<N> suffix; keep the experiment_id stem."""
    name = source_dir.name
    m = re.match(r"^(?P<exp>.+?)_seed(?P<n>\d+)$", name)
    return m.group("exp") if m else name


@dataclass
class Plan:
    source_dir: Path
    canonical_name: str
    manifest: Dict
    result_v2: Dict


def _plan_one(source_dir: Path) -> Plan:
    manifest_path = source_dir / "manifest.json"
    result_v2_path = source_dir / "result_v2.json"
    if not manifest_path.exists():
        raise ArchiveError(f"{source_dir}: missing manifest.json.")
    if not result_v2_path.exists():
        raise ArchiveError(f"{source_dir}: missing result_v2.json.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = json.loads(result_v2_path.read_text(encoding="utf-8"))
    result_v2, _wrapper = _unwrap_v2_payload(payload, result_v2_path)
    _validate_manifest(manifest, manifest_path)
    _validate_result_v2(result_v2, source_dir)

    stem = _dir_key(source_dir)
    canonical_name = CANONICAL_DIRS.get(source_dir.name) or CANONICAL_DIRS.get(stem)
    if canonical_name is None:
        raise ArchiveError(
            f"{source_dir.name}: not in CANONICAL_DIRS; cannot determine the "
            f"paper-grade target. Add a mapping in CANONICAL_DIRS or rename "
            f"the source directory to one of the registered names."
        )
    return Plan(source_dir=source_dir, canonical_name=canonical_name,
                manifest=manifest, result_v2=result_v2)


def _enforce_i5_p0_identity(plans: List[Plan]) -> None:
    """If both I5 and P0 are present, they must have the same fingerprint tuple."""
    by_alias: Dict[str, Plan] = {}
    for p in plans:
        for alias in p.manifest.get("alias_ids", []):
            by_alias[alias] = p

    if "I5" not in by_alias or "P0" not in by_alias:
        return

    a = by_alias["I5"]
    b = by_alias["P0"]
    fingerprint_keys = (
        "checkpoint_sha256", "config_sha256", "dataset_sha256",
        "split_sha256", "normalization_sha256", "git_commit",
        "epochs", "best_epoch",
    )
    for k in fingerprint_keys:
        if a.manifest.get(k) != b.manifest.get(k):
            raise ArchiveError(
                f"I5/P0 artifact identity violated: "
                f"{a.manifest['experiment_id']} (I5) vs "
                f"{b.manifest['experiment_id']} (P0) disagree on '{k}'. "
                f"I5 and P0 must be the same canonical artifact."
            )


def _prevent_silent_overwrite(target_dir: Path, force_rewrite: bool) -> None:
    if target_dir.exists():
        # Empty directory is allowed (a previously partial abort); a non-empty
        # directory is treated as an overwrite and refused without --force-rewrite.
        if any(target_dir.iterdir()):
            if not force_rewrite:
                raise ArchiveError(
                    f"{target_dir}: already exists and is non-empty. "
                    f"Refusing to silently overwrite; pass --force-rewrite to "
                    f"allow it (this clobbers previous paper-grade artifacts)."
                )


# ---------------------------------------------------------------------------
# Asset generation + copy
# ---------------------------------------------------------------------------

def _ensure_metrics_csv(target_dir: Path, plan: Plan) -> None:
    """Write metrics_v2.csv from result_v2.json if not already present."""
    target = target_dir / "metrics_v2.csv"
    src = plan.source_dir / "metrics_v2.csv"
    if src.exists():
        if _is_forbidden(src):
            raise ArchiveError(
                f"{src}: filename matches a forbidden pattern.")
        target.write_bytes(src.read_bytes())
        return
    write_v2_csv(plan.result_v2, str(target), model_name=plan.manifest["experiment_id"])


def _ensure_validation_md(target_dir: Path, plan: Plan) -> None:
    """Write validation.md from result_v2.json if not already present."""
    target = target_dir / "validation.md"
    src = plan.source_dir / "validation.md"
    if src.exists():
        if _is_forbidden(src):
            raise ArchiveError(
                f"{src}: filename matches a forbidden pattern.")
        target.write_bytes(src.read_bytes())
        return
    write_v2_markdown(
        plan.result_v2, str(target),
        model_name=plan.manifest["experiment_id"],
        header={"git_commit": plan.manifest.get("git_commit", "?"),
                "config_sha256": plan.manifest.get("config_sha256", "?")[:12] + "…"},
    )


def _copy_asset(src: Path, dst: Path) -> None:
    if src.is_dir():
        # Never descend into a directory that contains a checkpoint or
        # dataset blob. The CONTRACT is per-file, but as a safety net,
        # if any file in the directory matches a forbidden pattern, abort.
        for child in src.rglob("*"):
            if child.is_file() and (_is_forbidden(child) or _looks_like_checkpoint(child)):
                raise ArchiveError(
                    f"{child}: forbidden asset name; refusing to copy a "
                    f"directory that contains it.")
        shutil.copytree(src, dst)
        return
    if _is_forbidden(src) or _looks_like_checkpoint(src):
        raise ArchiveError(
            f"{src}: forbidden asset name; refusing to copy.")
    shutil.copy2(src, dst)


def _copy_optional_assets(source_dir: Path, target_dir: Path,
                          plan: Plan) -> List[str]:
    copied: List[str] = []
    for name in OPTIONAL_ASSETS:
        for src_path in (source_dir / name,
                         source_dir / "artifacts" / name,
                         source_dir / "configs" / name):
            if src_path.exists():
                dst = target_dir / name
                _copy_asset(src_path, dst)
                copied.append(name)
                break
    return copied


def _verify_manifest_hashes(source_dir: Path, plan: Plan) -> Dict[str, str]:
    """Re-hash source files (when present) and compare to manifest claims.

    Returns a SHA-result mapping. Diff is recorded but not blocking when the
    source file is intentionally absent (e.g. config_yaml was overridden on
    disk — fine, we recorded the SHA256 of what was actually used at
    training time).
    """
    out: Dict[str, str] = {}
    by_filename = {
        "config_sha256":        ["config.yaml", "config.json",
                                "config_snapshot.yaml", "experiment.yaml"],
        "dataset_sha256":       ["dataset.sha256", "h5.sha256"],
        "split_sha256":         ["splits.sha256", "split.sha256"],
        "normalization_sha256": ["normalization.sha256", "normalization.json.sha256"],
    }
    for sha_field, candidates in by_filename.items():
        for cand in candidates:
            src = source_dir / cand
            if src.exists():
                actual = _sha256(src)
                out[sha_field] = actual
                if plan.manifest.get(sha_field) and \
                   plan.manifest[sha_field] != actual:
                    print(f"[warn] {source_dir.name}: manifest.{sha_field} "
                          f"differs from on-disk {cand} SHA256 "
                          f"({plan.manifest[sha_field][:12]}… vs "
                          f"{actual[:12]}…). Recording.")
                break
    return out


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def discover_source_dirs(source_roots: Sequence[Path]) -> List[Path]:
    out: List[Path] = []
    for root in source_roots:
        if not root.exists():
            continue
        for child in sorted(p for p in root.iterdir() if p.is_dir()):
            if (child / "manifest.json").exists() or (child / "result_v2.json").exists():
                out.append(child)
    return out


def archive_one(plan: Plan, results_root: Path, force_rewrite: bool) -> Dict:
    target_dir = results_root / plan.canonical_name
    _prevent_silent_overwrite(target_dir, force_rewrite=force_rewrite)
    target_dir.mkdir(parents=True, exist_ok=False) if not target_dir.exists() \
        else None

    # Copy required assets (or generate them).
    _copy_asset(plan.source_dir / "manifest.json", target_dir / "manifest.json")
    _copy_asset(plan.source_dir / "result_v2.json", target_dir / "result_v2.json")

    pre_copied = {"manifest.json", "result_v2.json"}
    _ensure_metrics_csv(target_dir, plan)
    _ensure_validation_md(target_dir, plan)
    pre_copied |= {"metrics_v2.csv", "validation.md"}

    optional_copied = _copy_optional_assets(plan.source_dir, target_dir, plan)
    pre_copied |= set(optional_copied)

    # Hashes
    hashes = _verify_manifest_hashes(plan.source_dir, plan)
    (target_dir / "manifest_hashes.json").write_text(
        json.dumps(hashes, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "canonical_name": plan.canonical_name,
        "source_dir": str(plan.source_dir),
        "target_dir": str(target_dir),
        "files_present": sorted(p.name for p in target_dir.iterdir() if p.is_file()),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results-root", type=Path, default=Path("results"),
                   help="Target root (paper-grade tree).")
    p.add_argument(
        "--source-dir", type=Path, action="append", default=None,
        help="Source root containing <id>_seed<N>/{manifest,result_v2}.json. "
             "May be passed multiple times. Defaults to "
             "outputs/{backbone_gate,axis_i,axis_ii_c1}/."
    )
    p.add_argument("--force-rewrite", action="store_true",
                   help="Allow overwriting a non-empty target directory.")
    args = p.parse_args(argv)

    source_roots = args.source_dir or [
        Path("outputs/backbone_gate"),
        Path("outputs/axis_i"),
        Path("outputs/axis_ii_c1"),
    ]

    sources = discover_source_dirs(source_roots)
    if not sources:
        print(f"[fatal] no source directories found under: "
              f"{[str(r) for r in source_roots]}", file=sys.stderr)
        return 2

    plans: List[Plan] = []
    for s in sources:
        try:
            plans.append(_plan_one(s))
        except ArchiveError as e:
            print(f"[fatal] {e}", file=sys.stderr)
            return 3

    _enforce_i5_p0_identity(plans)

    manifest_rows: List[Dict] = []
    for plan in plans:
        info = archive_one(plan, args.results_root,
                           force_rewrite=args.force_rewrite)
        manifest_rows.append(info)
        print(f"[ok] archived {plan.source_dir.name} -> {info['target_dir']}")

    manifest_csv = args.results_root / "ARCHIVE_MANIFEST.csv"
    with manifest_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "canonical_name", "source_dir", "target_dir", "files_present",
        ])
        w.writeheader()
        for r in manifest_rows:
            r["files_present"] = ";".join(r["files_present"])
            w.writerow(r)
    print(f"[ok] wrote {manifest_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
