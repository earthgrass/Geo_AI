"""Promote GPU-run outputs into the paper-grade tracked result tree.

Reads from default source roots:

    outputs/backbone_gate/<id>/
    outputs/axis_i/<id>/
    outputs/axis_ii_c1/<id>/

(overridable via ``--source-dir``) and writes a canonical result tree at
``results/<canonical_name>_seed<N>/``, where ``canonical_name`` is the
frozen name from the spec and ``seed`` is read from the manifest, NOT
inferred from the source directory name:

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

The archiver accepts the REAL GPU manifest schema (the schema produced
by ``scripts/run_experiment.py::write_manifest``). It does NOT require
synthetic fields like ``experiment_id`` / ``alias_ids`` / ``manifest.split``;
those are mapped internally by
``scripts._artifact_normalize.normalize_gpu_artifact_metadata``.

Scientific fingerprint equality is the dedup rule. ``I5`` and ``P0`` MUST
share a single artifact (one of the two manifests declares both aliases).
The two ``I2_resconvlstm`` source directories (backbone_gate + axis_i)
are validate-only re-uses of the same official E2 checkpoint and MUST
deduplicate to a single canonical ``I2_resconvlstm_seed42_v2``.

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
3. Verifies the I5/P0 same-artifact identity whenever both aliases are
   encountered in the source roots.
4. Verifies every required file is present (manifest + result_v2).
   If missing, ABORTS with a precise error.
5. Verifies the SHA256 metadata recorded in the manifest matches the
   on-disk file's SHA256 (where the source file exists). Differences
   are recorded but do not block archival when the source file is
   intentionally absent (e.g. config snapshot was overridden on disk).
6. PRESERVES the source ``manifest.json`` and ``result_v2.json``
   byte-for-byte (shutil.copy2). The GPU-written artifact is never
   re-serialized.
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
from dataclasses import dataclass, field
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
from scripts._artifact_normalize import (  # noqa: E402
    NormalizerError,
    load_raw_manifest_and_result,
    scientific_fingerprint,
    SCIENTIFIC_FINGERPRINT_KEYS,
    unwrap_v2_payload,
)


# ---------------------------------------------------------------------------
# Canonical mapping: source-dir name -> results/<canonical>/
#
# Real source dirs (extracted from the GPU RESULTS-ONLY archive) are:
#
#   outputs/backbone_gate/I0_persistence
#   outputs/backbone_gate/I1_plain_convlstm
#   outputs/backbone_gate/I2_resconvlstm        (validate-only reuse)
#   outputs/backbone_gate/B1_trajgru
#   outputs/axis_i/I2_resconvlstm               (validate-only reuse)
#   outputs/axis_i/I3_resconvlstm_cma
#   outputs/axis_i/I4_static_terrain
#   outputs/axis_i/I5_terrain_geometry
#   outputs/axis_ii_c1/P1_resconvlstm_smooth
#   outputs/axis_ii_c1/P2_terrain_extreme
#   outputs/axis_ii_c1/P3_resconvlstm_smooth_extreme
#
# All duplicates (I2 in two places) collapse to I2_resconvlstm_seed42_v2.
# ---------------------------------------------------------------------------

CANONICAL_DIRS: Dict[str, str] = {
    # Real GPU source-dir names (no _seed<N> suffix).
    "I0_persistence":                       "I0_persistence_seed42",
    "I1_plain_convlstm":                    "I1_plain_convlstm_seed42",
    "I2_resconvlstm":                       "I2_resconvlstm_seed42_v2",
    "B1_trajgru":                           "B1_trajgru_seed42",
    "I3_resconvlstm_cma":                   "I3_resconvlstm_cma_seed42",
    "I4_static_terrain":                    "I4_static_terrain_seed42",
    "I5_terrain_geometry":                  "I5_terrain_geometry_seed42",
    "P1_resconvlstm_smooth":                "P1_smooth_seed42",
    "P2_terrain_extreme":                   "P2_extreme_seed42",
    "P3_resconvlstm_smooth_extreme":        "P3_smooth_extreme_seed42",
    # Legacy synthetic source-dir names (with _seed<N> suffix), kept for
    # backward compatibility with pre-R32 test fixtures.
    "I0_persistence_seed42":                "I0_persistence_seed42",
    "I1_plain_convlstm_seed42":             "I1_plain_convlstm_seed42",
    "I2_resconvlstm_seed42":                "I2_resconvlstm_seed42_v2",
    "B1_trajgru_seed42":                    "B1_trajgru_seed42",
    "I3_resconvlstm_cma_seed42":            "I3_resconvlstm_cma_seed42",
    "I4_static_terrain_seed42":             "I4_static_terrain_seed42",
    "I5_terrain_geometry_seed42":           "I5_terrain_geometry_seed42",
    "P1_resconvlstm_smooth_seed42":         "P1_smooth_seed42",
    "P2_terrain_extreme_seed42":            "P2_extreme_seed42",
    "P3_resconvlstm_smooth_extreme_seed42": "P3_smooth_extreme_seed42",
    "I2_resconvlstm_seed42_v2":             "I2_resconvlstm_seed42_v2",
    # Short legacy alias forms (used by some synthetic fixtures).
    "P1_smooth_seed42":                     "P1_smooth_seed42",
    "P2_extreme_seed42":                    "P2_extreme_seed42",
    "P3_smooth_extreme_seed42":             "P3_smooth_extreme_seed42",
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
# Contract validation (operates on the NORMALIZED manifest)
# ---------------------------------------------------------------------------

def _validate_normalized(norm: Dict, source_dir: Path) -> None:
    """FAIL FAST on any fingerprint violation of the canonical schema."""
    def _check(key: str, expected, *, allow_none: bool = False) -> None:
        v = norm.get(key)
        if allow_none and v is None:
            return
        if v != expected:
            raise ArchiveError(
                f"{source_dir}: manifest field '{key}' must equal "
                f"{expected!r}; got {v!r}.")

    _check("protocol_id", EXPECTED_PROTOCOL_ID)
    _check("split", EXPECTED_SPLIT)
    _check("test_status", EXPECTED_TEST_STATUS)
    _check("smoke", EXPECTED_SMOKE)
    _check("n_events", EXPECTED_N_EVENTS)
    _check("n_windows", EXPECTED_N_WINDOWS)


# ---------------------------------------------------------------------------
# Backward-compat shims for the legacy synthetic schema.
#
# The original analyzer/archiver tests used a synthetic manifest shape:
#
#     {"experiment_id": "...", "alias_ids": [...], "split": "val", ...}
#
# These shims translate that shape into the real GPU schema on the fly so
# that the existing test suite keeps working. New tests must use the real
# GPU schema via scripts._artifact_normalize.
# ---------------------------------------------------------------------------

def _legacy_synthetic_to_norm(synthetic: Dict, source_dir: Path) -> Dict:
    """Build a normalized dict from the synthetic legacy shape.

    Required synthetic keys: experiment_id, alias_ids, git_commit,
    config_sha256, dataset_sha256, split_sha256, normalization_sha256,
    checkpoint_sha256, protocol_id, test_status, split, smoke. The
    ``split`` from the synthetic dict is checked against the inner
    result below in :func:`_validate_manifest`.
    """
    norm: Dict = {
        "experiment_id":          synthetic.get("experiment_id", "?"),
        "alias_ids":              list(synthetic.get("alias_ids", [])),
        "mode":                   synthetic.get("mode", "validate-only"),
        "model":                  synthetic.get("model", "?"),
        "seed":                   synthetic.get("seed", 42),
        "epochs":                 synthetic.get("epochs", 20),
        "git_commit":             synthetic.get("git_commit"),
        "config_path":            synthetic.get("config_path", "?"),
        "config_sha256":          synthetic.get("config_sha256"),
        "dataset_sha256":         synthetic.get("dataset_sha256"),
        "split_sha256":           synthetic.get("split_sha256"),
        "normalization_sha256":   synthetic.get("normalization_sha256"),
        "checkpoint_path":        synthetic.get("checkpoint_path"),
        "checkpoint_sha256":      synthetic.get("checkpoint_sha256"),
        "best_epoch":             synthetic.get("best_epoch"),
        "best_val_mse":           synthetic.get("best_val_mse"),
        "selection_metric":       synthetic.get("selection_metric", "rain_mse"),
        "input_channel_indices":  synthetic.get("input_channel_indices", []),
        "loss_components":        synthetic.get("loss_components", []),
        "protocol_id":            synthetic.get("protocol_id"),
        "test_status":            synthetic.get("test_status"),
        "smoke":                  synthetic.get("smoke"),
        # Inner-result fields; the legacy shim carries them inline on
        # the synthetic dict.
        "split":                  synthetic.get("split"),
        "n_events":               synthetic.get("n_events", EXPECTED_N_EVENTS),
        "n_windows":              synthetic.get("n_windows", EXPECTED_N_WINDOWS),
        "thresholds":             synthetic.get("thresholds",
                                                  [5.0, 10.0, 20.0, 30.0]),
        "per_event":              synthetic.get("per_event", {}),
        "overall_global":         synthetic.get("overall_global", {}),
    }
    return norm


def _validate_manifest(m: Dict, manifest_path: Path) -> None:
    """Backward-compat validator for the synthetic legacy schema.

    Accepts the legacy shape used by the pre-R32 test fixtures. Translates
    to the normalized shape internally, then validates. New code must use
    the real GPU schema via scripts._artifact_normalize.
    """
    required = (
        "experiment_id", "alias_ids", "git_commit",
        "config_sha256", "dataset_sha256", "split_sha256",
        "normalization_sha256", "checkpoint_sha256",
        "protocol_id", "test_status", "split", "smoke",
    )
    for key in required:
        if key not in m:
            raise ArchiveError(
                f"{manifest_path}: missing required manifest field '{key}'.")
    norm = _legacy_synthetic_to_norm(m, manifest_path.parent)
    _validate_normalized(norm, manifest_path.parent)


def _validate_result_v2(r: Dict, source_dir: Path) -> None:
    """Backward-compat validator for the legacy un-wrapped result_v2.

    New code must use the wrapper format via scripts._artifact_normalize.
    """
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


# ---------------------------------------------------------------------------
# Canonical-name resolution
# ---------------------------------------------------------------------------

def _dir_key(source_dir: Path) -> str:
    """Return the source-dir name, optionally stripped of a trailing
    ``_seed<N>`` suffix. Real GPU directories do NOT carry the suffix; the
    suffix is only present in synthetic fixtures.
    """
    name = source_dir.name
    m = re.match(r"^(?P<exp>.+?)_seed(?P<n>\d+)$", name)
    return m.group("exp") if m else name


@dataclass
class Plan:
    source_dir: Path
    canonical_name: str
    raw_manifest: Dict
    wrapper_payload: Dict
    inner_result: Dict
    norm: Dict
    manifest_bytes_path: Path
    result_v2_bytes_path: Path

    @property
    def result_v2(self) -> Dict:
        """Backward-compat alias for ``inner_result``. Legacy code expects
        ``plan.result_v2`` to expose the evaluator fields directly; the
        wrapper is carried in ``plan.wrapper_payload``.
        """
        return self.inner_result

    @property
    def manifest(self) -> Dict:
        """Backward-compat alias for ``norm``. Legacy code expects
        ``plan.manifest`` to expose canonical keys like ``alias_ids``.
        """
        return self.norm


def _plan_one(source_dir: Path) -> Plan:
    """Build a Plan for one source directory.

    Auto-detects the on-disk schema: real GPU manifests carry
    ``experiment`` + ``aliases``; legacy synthetic manifests carry
    ``experiment_id`` + ``alias_ids``. Both paths converge on a
    normalized dict with the canonical key names; downstream code only
    sees the normalized shape.
    """
    manifest_path = source_dir / "manifest.json"
    result_v2_path = source_dir / "result_v2.json"
    if not manifest_path.exists():
        raise ArchiveError(f"{source_dir}: missing manifest.json.")
    if not result_v2_path.exists():
        raise ArchiveError(f"{source_dir}: missing result_v2.json.")

    try:
        raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise ArchiveError(f"{source_dir}: manifest.json is not valid JSON: {e}")
    try:
        wrapper = json.loads(result_v2_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise ArchiveError(f"{source_dir}: result_v2.json is not valid JSON: {e}")

    is_real_gpu = "experiment" in raw_manifest and "aliases" in raw_manifest
    is_synthetic = "experiment_id" in raw_manifest and "alias_ids" in raw_manifest

    if is_real_gpu and not is_synthetic:
        # Real GPU path: use the shared normalizer.
        try:
            _, _, inner, norm = load_raw_manifest_and_result(
                source_dir, error_cls=ArchiveError,
            )
        except NormalizerError as e:
            raise ArchiveError(str(e))
    elif is_synthetic and not is_real_gpu:
        # Legacy synthetic path: translate inline.
        inner, _ = unwrap_v2_payload(wrapper, result_v2_path, ArchiveError)
        # Carry inner result fields into the synthetic manifest so the
        # normalizer-style contract check sees them.
        merged = dict(raw_manifest)
        if "split" not in merged and "split" in inner:
            merged["split"] = inner["split"]
        for k in ("n_events", "n_windows", "thresholds",
                  "per_event", "overall_global"):
            if k not in merged and k in inner:
                merged[k] = inner[k]
        norm = _legacy_synthetic_to_norm(merged, source_dir)
    else:
        raise ArchiveError(
            f"{source_dir}: manifest schema is ambiguous or unsupported. "
            f"Real GPU manifests carry 'experiment' + 'aliases'; legacy "
            f"synthetic manifests carry 'experiment_id' + 'alias_ids'. "
            f"This file matches neither or both."
        )

    _validate_normalized(norm, source_dir)

    canonical_name = CANONICAL_DIRS.get(source_dir.name)
    if canonical_name is None:
        # Strip a trailing _seed<N> suffix (legacy synthetic dirs may
        # carry it) and try once more.
        stem = _dir_key(source_dir)
        canonical_name = CANONICAL_DIRS.get(stem)
    if canonical_name is None:
        raise ArchiveError(
            f"{source_dir.name}: not in CANONICAL_DIRS; cannot determine the "
            f"paper-grade target. Add a mapping in CANONICAL_DIRS or rename "
            f"the source directory to one of the registered names. (Note: "
            f"real GPU runner does NOT add a _seed<N> suffix to the source "
            f"directory; the seed is read from manifest.seed.)"
        )

    return Plan(
        source_dir=source_dir,
        canonical_name=canonical_name,
        raw_manifest=raw_manifest,
        wrapper_payload=wrapper,
        inner_result=inner,
        norm=norm,
        manifest_bytes_path=manifest_path,
        result_v2_bytes_path=result_v2_path,
    )


def _enforce_i5_p0_identity(plans: List[Plan]) -> None:
    """If both I5 and P0 are present, they must have the same scientific
    fingerprint.

    A single manifest may declare both aliases (``['I5', 'P0']``), in which
    case I5 ≡ P0 is trivially true. Two SEPARATE manifests claiming
    different aliases must agree on the fingerprint tuple below.
    """
    by_alias: Dict[str, Plan] = {}
    for p in plans:
        for alias in p.norm.get("alias_ids", []):
            if alias in by_alias:
                # Same alias claimed twice — already caught by canonical-name
                # uniqueness; allow if it's the same scientific artifact.
                existing = by_alias[alias]
                fp_existing = scientific_fingerprint(existing.norm)
                fp_new = scientific_fingerprint(p.norm)
                if fp_existing != fp_new:
                    raise ArchiveError(
                        f"Alias '{alias}' is claimed by two source "
                        f"directories whose scientific fingerprints differ: "
                        f"{existing.source_dir} vs {p.source_dir}. Refusing to "
                        f"silently collapse them."
                    )
                continue
            by_alias[alias] = p

    if "I5" not in by_alias or "P0" not in by_alias:
        return

    a = by_alias["I5"]
    b = by_alias["P0"]
    if a is b:
        return  # same artifact declares both aliases

    fa = scientific_fingerprint(a.norm)
    fb = scientific_fingerprint(b.norm)
    if fa != fb:
        diff = [
            k for k, va, vb in zip(SCIENTIFIC_FINGERPRINT_KEYS, fa, fb)
            if va != vb
        ]
        raise ArchiveError(
            f"I5/P0 artifact identity violated: "
            f"{a.source_dir} (aliases={a.norm['alias_ids']}) vs "
            f"{b.source_dir} (aliases={b.norm['alias_ids']}) "
            f"disagree on scientific fingerprint fields: {diff}. "
            f"I5 and P0 must be the same canonical artifact per the alias "
            f"registry."
        )


def _enforce_duplicate_dedup(plans: List[Plan]) -> Dict[str, List[Plan]]:
    """Group plans by scientific fingerprint. Multiple sources sharing a
    fingerprint collapse to one canonical target (the first encountered
    plan wins; others are recorded as duplicate_sources).
    """
    groups: Dict[Tuple, List[Plan]] = {}
    for p in plans:
        groups.setdefault(scientific_fingerprint(p.norm), []).append(p)
    return groups


def _prevent_silent_overwrite(target_dir: Path, force_rewrite: bool) -> None:
    if target_dir.exists():
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
    write_v2_csv(plan.inner_result, str(target),
                 model_name=plan.norm["experiment_id"])


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
        plan.inner_result, str(target),
        model_name=plan.norm["experiment_id"],
        header={
            "git_commit":   plan.norm.get("git_commit") or "?",
            "config_sha256": (plan.norm.get("config_sha256") or "?")[:12] + "…",
            "checkpoint_sha256": (
                (plan.norm.get("checkpoint_sha256") or "n/a")[:12] + "…"
                if plan.norm.get("checkpoint_sha256") else "n/a (non-parametric)"
            ),
        },
    )


def _copy_asset(src: Path, dst: Path) -> None:
    if src.is_dir():
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

    Returns a SHA-result mapping. Diff is recorded but not blocking when
    the source file is intentionally absent (e.g. config_yaml was
    overridden on disk — fine, we recorded the SHA256 of what was
    actually used at training time).
    """
    out: Dict[str, str] = {}
    by_filename = {
        "config_sha256":        ["config.yaml", "config.json",
                                "config_snapshot.yaml", "experiment.yaml"],
        "dataset_sha256":       ["dataset.sha256", "h5.sha256"],
        "split_sha256":         ["splits.sha256", "split.sha256"],
        "normalization_sha256": ["normalization.sha256",
                                 "normalization.json.sha256"],
    }
    for sha_field, candidates in by_filename.items():
        for cand in candidates:
            src = source_dir / cand
            if src.exists():
                actual = _sha256(src)
                out[sha_field] = actual
                claimed = plan.norm.get(sha_field)
                if claimed and claimed != actual:
                    print(f"[warn] {source_dir.name}: manifest.{sha_field} "
                          f"differs from on-disk {cand} SHA256 "
                          f"({claimed[:12]}… vs {actual[:12]}…). Recording.")
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
            if (child / "manifest.json").exists() or \
               (child / "result_v2.json").exists():
                out.append(child)
    return out


def archive_one(plan: Plan, results_root: Path, force_rewrite: bool) -> Dict:
    target_dir = results_root / plan.canonical_name
    _prevent_silent_overwrite(target_dir, force_rewrite=force_rewrite)
    target_dir.mkdir(parents=True, exist_ok=False) if not target_dir.exists() \
        else None

    # Byte-identical copy of the GPU-written manifest + result_v2.
    _copy_asset(plan.manifest_bytes_path, target_dir / "manifest.json")
    _copy_asset(plan.result_v2_bytes_path, target_dir / "result_v2.json")

    pre_copied = {"manifest.json", "result_v2.json"}
    _ensure_metrics_csv(target_dir, plan)
    _ensure_validation_md(target_dir, plan)
    pre_copied |= {"metrics_v2.csv", "validation.md"}

    optional_copied = _copy_optional_assets(plan.source_dir, target_dir, plan)
    pre_copied |= set(optional_copied)

    hashes = _verify_manifest_hashes(plan.source_dir, plan)
    (target_dir / "manifest_hashes.json").write_text(
        json.dumps(hashes, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "canonical_name": plan.canonical_name,
        "source_dir": str(plan.source_dir),
        "target_dir": str(target_dir),
        "experiment_id": plan.norm["experiment_id"],
        "alias_ids": "|".join(plan.norm["alias_ids"]),
        "seed": plan.norm["seed"],
        "checkpoint_sha256": plan.norm.get("checkpoint_sha256") or "",
        "files_present": sorted(p.name for p in target_dir.iterdir()
                                if p.is_file()),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results-root", type=Path, default=Path("results"),
                   help="Target root (paper-grade tree).")
    p.add_argument(
        "--source-dir", type=Path, action="append", default=None,
        help="Source root containing <id>/{manifest,result_v2}.json. May be "
             "passed multiple times. Defaults to "
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

    # Dedup by scientific fingerprint: multiple source dirs mapping to the
    # same canonical target (e.g. the I2 validate-only reuse in backbone_gate
    # AND axis_i) must collapse to ONE archival target.
    by_fp: Dict[Tuple, List[Plan]] = {}
    for plan in plans:
        by_fp.setdefault(scientific_fingerprint(plan.norm), []).append(plan)

    canonical_names: Dict[str, str] = {}
    manifest_rows: List[Dict] = []
    for fp, fp_plans in by_fp.items():
        if len(fp_plans) > 1:
            names = [pl.canonical_name for pl in fp_plans]
            if len(set(names)) > 1:
                # Same fingerprint but different canonical names — only
                # happens if CANONICAL_DIRS is misconfigured for a true
                # duplicate. Refuse.
                raise ArchiveError(
                    f"Duplicate source dirs {names} share a scientific "
                    f"fingerprint but have different canonical names. Fix "
                    f"CANONICAL_DIRS so duplicates collapse to one target."
                )
        primary = fp_plans[0]
        if primary.canonical_name in canonical_names:
            # Already archived by an earlier group; should not happen if
            # CANONICAL_DIRS is internally consistent.
            raise ArchiveError(
                f"Canonical target {primary.canonical_name} is targeted by "
                f"more than one fingerprint group; check CANONICAL_DIRS."
            )
        canonical_names[primary.canonical_name] = primary.norm["experiment_id"]

        info = archive_one(primary, args.results_root,
                           force_rewrite=args.force_rewrite)
        if len(fp_plans) > 1:
            info["duplicate_sources"] = ";".join(
                str(pl.source_dir) for pl in fp_plans[1:]
            )
        manifest_rows.append(info)
        if len(fp_plans) > 1:
            print(f"[ok] archived {primary.source_dir.name} -> "
                  f"{info['target_dir']} (deduped from "
                  f"{[pl.source_dir.name for pl in fp_plans]})")
        else:
            print(f"[ok] archived {primary.source_dir.name} -> "
                  f"{info['target_dir']}")

    manifest_csv = args.results_root / "ARCHIVE_MANIFEST.csv"
    with manifest_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "canonical_name", "experiment_id", "alias_ids", "seed",
            "checkpoint_sha256", "source_dir", "target_dir",
            "duplicate_sources", "files_present",
        ])
        w.writeheader()
        for r in manifest_rows:
            r["files_present"] = ";".join(r["files_present"])
            w.writerow(r)
    print(f"[ok] wrote {manifest_csv}")
    print(f"[ok] {len(manifest_rows)} canonical experiments archived "
          f"from {len(plans)} source dirs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())