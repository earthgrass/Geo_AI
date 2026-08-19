"""Frozen-contract tests for scripts/archive_validation_results.py.

Each test pins one item from the archiver's contract:
  - never copies *.pth / *.h5 / *.log / tfevents
  - never copies test-set artifacts
  - validates test_status=SEALED, split=val, protocol=evaluation_v2,
    smoke=false, n_events=7, n_windows=1266
  - prevents silent overwrites
  - enforces I5 == P0 same-artifact identity
  - never silently promotes a partial source
  - resolves canonical names (I2_resconvlstm_seed42 -> I2_resconvlstm_seed42_v2)
  - generates metrics_v2.csv / validation.md when absent
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.archive_validation_results import (  # noqa: E402
    ArchiveError,
    CANONICAL_DIRS,
    EXPECTED_PROTOCOL_ID,
    EXPECTED_SPLIT,
    EXPECTED_TEST_STATUS,
    EXPECTED_SMOKE,
    EXPECTED_N_EVENTS,
    EXPECTED_N_WINDOWS,
    _copy_asset,
    _dir_key,
    _enforce_i5_p0_identity,
    _is_forbidden,
    _looks_like_checkpoint,
    _plan_one,
    _prevent_silent_overwrite,
    _validate_manifest,
    _validate_result_v2,
    Plan,
    archive_one,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _manifest_dict(**overrides) -> dict:
    m = {
        "experiment_id": "I2_resconvlstm_seed42_v2",
        "alias_ids": ["I2"],
        "seed": 42,
        "git_commit": "deadbeef" * 4,
        "config_sha256": "ab" * 32,
        "dataset_sha256": "cd" * 32,
        "split_sha256": "ef" * 32,
        "normalization_sha256": "12" * 32,
        "checkpoint_sha256": "34" * 32,
        "protocol_id": EXPECTED_PROTOCOL_ID,
        "test_status": EXPECTED_TEST_STATUS,
        "split": EXPECTED_SPLIT,
        "smoke": EXPECTED_SMOKE,
        "epochs": 20,
        "best_epoch": 17,
    }
    m.update(overrides)
    return m


def _result_v2_dict(per_event: dict[int, dict] | None = None,
                    model: str = "I2_resconvlstm_seed42_v2",
                    flat: bool = False) -> dict:
    """Return a v2 result dict in the wrapper format produced by
    ``src.evaluation.reporting.write_v2_json``:

        {"model": str, "result": {...evaluator fields...}}

    Set ``flat=True`` to emit the legacy un-wrapped format (used only by
    negative tests that assert the archiver rejects it).
    """
    pe: dict = {}
    for tid in range(1, EXPECTED_N_EVENTS + 1):
        pe[str(tid)] = {
            "MAE_event": 1.0, "RMSE_event": 1.0, "SSIM_event_mean": 0.5,
            "categorical": {
                f"{t:g}mmh" if float(t).is_integer() else f"{t}mmh":
                    {"CSI": 0.2, "POD": 0.4, "FAR": 0.6,
                     "HSS": 0.1, "BIAS": 1.0, "ACC": 0.6}
                for t in (5, 10, 20, 30)
            },
        }
    inner = {
        "protocol_id": EXPECTED_PROTOCOL_ID,
        "split": EXPECTED_SPLIT,
        "test_status": EXPECTED_TEST_STATUS,
        "smoke": EXPECTED_SMOKE,
        "n_events": EXPECTED_N_EVENTS,
        "n_windows": EXPECTED_N_WINDOWS,
        "thresholds": [5.0, 10.0, 20.0, 30.0],
        "per_event": pe,
    }
    if flat:
        return inner
    return {"model": model, "result": inner}


def _write_source(out_root: Path, exp_id: str,
                  manifest_overrides: dict | None = None,
                  extra_files: list[tuple[str, bytes | str]] | None = None,
                  missing: list[str] | None = None,
                  alias_ids: list | None = None) -> Path:
    """Create outputs/<axis_root>/<exp_id>/{manifest, result_v2, extras}."""
    src_dir = out_root / exp_id
    src_dir.mkdir(parents=True, exist_ok=True)
    missing = missing or []
    if alias_ids is None:
        # Default to the experiment prefix (first underscore-separated token).
        alias_ids = [exp_id.split("_")[0]]
    if "manifest.json" not in missing:
        (src_dir / "manifest.json").write_text(
            json.dumps(_manifest_dict(
                exp_id=exp_id,
                alias_ids=alias_ids,
                **(manifest_overrides or {}),
            )),
            encoding="utf-8",
        )
    if "result_v2.json" not in missing:
        (src_dir / "result_v2.json").write_text(
            json.dumps(_result_v2_dict()), encoding="utf-8")
    for fname, content in (extra_files or []):
        if isinstance(content, bytes):
            (src_dir / fname).write_bytes(content)
        else:
            (src_dir / fname).write_text(content, encoding="utf-8")
    return src_dir


# ---------------------------------------------------------------------------
# Forbidden file detection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "model_best.pth", "checkpoint.pt", "weights.ckpt",
    "dataset.h5", "tiles.npz", "labels.npy", "img.tif",
    "events.out.tfevents.123.host", "x.tfevents.foo",
    "train.log", "scratch.tmp", "x.swp",
])
def test_is_forbidden_matches(name):
    assert _is_forbidden(Path(name))


@pytest.mark.parametrize("name", [
    "manifest.json", "result_v2.json", "metrics_v2.csv",
    "validation.md", "history.json", "config.yaml",
])
def test_is_forbidden_does_not_match_safe(name):
    assert not _is_forbidden(Path(name))


def test_is_forbidden_blocks_test_path():
    p = Path("outputs/test/I2_resconvlstm_seed42_v2/model.pt")
    # The full path matches via `/test/`.
    assert _is_forbidden(p)


def test_looks_like_checkpoint():
    assert _looks_like_checkpoint(Path("x.pth"))
    assert _looks_like_checkpoint(Path("x.pt"))
    assert _looks_like_checkpoint(Path("x.ckpt"))
    assert not _looks_like_checkpoint(Path("x.json"))


# ---------------------------------------------------------------------------
# Copy semantics
# ---------------------------------------------------------------------------

def test_copy_asset_rejects_pth(tmp_path):
    src = tmp_path / "model_best.pth"
    src.write_bytes(b"\x80\x00\x01")
    dst = tmp_path / "dst.pth"
    with pytest.raises(ArchiveError):
        _copy_asset(src, dst)


def test_copy_asset_rejects_h5(tmp_path):
    src = tmp_path / "data.h5"
    src.write_bytes(b"\x89HDF\r\n\x1a\n")
    dst = tmp_path / "dst.h5"
    with pytest.raises(ArchiveError):
        _copy_asset(src, dst)


def test_copy_asset_rejects_log(tmp_path):
    src = tmp_path / "train.log"
    src.write_text("log content")
    dst = tmp_path / "dst.log"
    with pytest.raises(ArchiveError):
        _copy_asset(src, dst)


def test_copy_asset_rejects_tfevents(tmp_path):
    src = tmp_path / "events.out.tfevents.42.host"
    src.write_bytes(b"\x00\x01")
    dst = tmp_path / "x"
    with pytest.raises(ArchiveError):
        _copy_asset(src, dst)


def test_copy_asset_rejects_directory_with_pth(tmp_path):
    src_dir = tmp_path / "saved"
    src_dir.mkdir()
    (src_dir / "config.yaml").write_text("a: 1")
    (src_dir / "model_best.pth").write_bytes(b"\x80\x00")
    dst = tmp_path / "dst"
    with pytest.raises(ArchiveError):
        _copy_asset(src_dir, dst)


def test_copy_asset_copies_safe_file(tmp_path):
    src = tmp_path / "history.json"
    src.write_text("[]")
    dst = tmp_path / "dst.json"
    _copy_asset(src, dst)
    assert dst.read_text() == "[]"


# ---------------------------------------------------------------------------
# Manifest / result validation
# ---------------------------------------------------------------------------

def test_validate_manifest_missing_required_field(tmp_path):
    m = _manifest_dict()
    del m["checkpoint_sha256"]
    with pytest.raises(ArchiveError, match="missing required manifest field"):
        _validate_manifest(m, tmp_path / "manifest.json")


def test_validate_manifest_wrong_protocol(tmp_path):
    m = _manifest_dict(protocol_id="evaluation_v1")
    with pytest.raises(ArchiveError, match="protocol_id"):
        _validate_manifest(m, tmp_path / "manifest.json")


def test_validate_manifest_wrong_split(tmp_path):
    m = _manifest_dict(split="test")
    with pytest.raises(ArchiveError, match="split"):
        _validate_manifest(m, tmp_path / "manifest.json")


def test_validate_manifest_wrong_test_status(tmp_path):
    m = _manifest_dict(test_status="UNSEALED")
    with pytest.raises(ArchiveError, match="test_status"):
        _validate_manifest(m, tmp_path / "manifest.json")


def test_validate_manifest_smoke_true(tmp_path):
    m = _manifest_dict(smoke=True)
    with pytest.raises(ArchiveError, match="smoke"):
        _validate_manifest(m, tmp_path / "manifest.json")


def test_validate_result_v2_wrong_n_events(tmp_path):
    r = _result_v2_dict(flat=True)
    r["n_events"] = 5
    with pytest.raises(ArchiveError, match="n_events"):
        _validate_result_v2(r, tmp_path)


def test_validate_result_v2_wrong_split(tmp_path):
    r = _result_v2_dict(flat=True)
    r["split"] = "test"
    with pytest.raises(ArchiveError, match="split"):
        _validate_result_v2(r, tmp_path)


def test_validate_result_v2_wrong_protocol(tmp_path):
    r = _result_v2_dict(flat=True)
    r["protocol_id"] = "evaluation_v1"
    with pytest.raises(ArchiveError, match="protocol_id"):
        _validate_result_v2(r, tmp_path)


def test_validate_result_v2_unsealed(tmp_path):
    r = _result_v2_dict(flat=True)
    r["test_status"] = "UNSEALED"
    with pytest.raises(ArchiveError, match="test_status"):
        _validate_result_v2(r, tmp_path)


# ---------------------------------------------------------------------------
# Canonical-name resolution
# ---------------------------------------------------------------------------

def test_dir_key_strips_seed_suffix():
    assert _dir_key(Path("I2_resconvlstm_seed42")) == "I2_resconvlstm"
    assert _dir_key(Path("P3_resconvlstm_smooth_extreme_seed42")) == \
        "P3_resconvlstm_smooth_extreme"


def test_canonical_dirs_includes_required_aliases():
    must = {
        "I0_persistence_seed42",
        "I1_plain_convlstm_seed42",
        "I2_resconvlstm_seed42",
        "B1_trajgru_seed42",
        "I3_resconvlstm_cma_seed42",
        "I4_static_terrain_seed42",
        "I5_terrain_geometry_seed42",
        "P1_smooth_seed42",
        "P2_extreme_seed42",
        "P3_smooth_extreme_seed42",
    }
    assert must.issubset(set(CANONICAL_DIRS.keys()))


def test_canonical_dirs_i2_resolves_to_v2():
    assert (CANONICAL_DIRS["I2_resconvlstm_seed42"]
            == "I2_resconvlstm_seed42_v2")
    assert (CANONICAL_DIRS["I2_resconvlstm_seed42_v2"]
            == "I2_resconvlstm_seed42_v2")


def test_plan_one_renames_i2_to_v2(tmp_path):
    outputs = tmp_path / "outputs" / "backbone_gate"
    src = _write_source(outputs, "I2_resconvlstm_seed42")
    plan = _plan_one(src)
    assert plan.canonical_name == "I2_resconvlstm_seed42_v2"


def test_plan_one_missing_manifest(tmp_path):
    outputs = tmp_path / "outputs"
    src = outputs / "I2_resconvlstm_seed42"
    src.mkdir(parents=True)
    (src / "result_v2.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ArchiveError, match="missing manifest.json"):
        _plan_one(src)


def test_plan_one_missing_result_v2(tmp_path):
    outputs = tmp_path / "outputs"
    src = outputs / "I2_resconvlstm_seed42"
    src.mkdir(parents=True)
    (src / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ArchiveError, match="missing result_v2.json"):
        _plan_one(src)


def test_plan_one_unknown_canonical_name(tmp_path):
    outputs = tmp_path / "outputs"
    src = outputs / "I9_unknown_seed42"
    src.mkdir(parents=True)
    (src / "manifest.json").write_text(
        json.dumps(_manifest_dict(
            exp_id="I9_unknown_seed42", alias_ids=["I9"])), encoding="utf-8")
    (src / "result_v2.json").write_text(
        json.dumps(_result_v2_dict()), encoding="utf-8")
    with pytest.raises(ArchiveError, match="not in CANONICAL_DIRS"):
        _plan_one(src)


# ---------------------------------------------------------------------------
# Overwrite protection
# ---------------------------------------------------------------------------

def test_prevent_silent_overwrite_on_empty_dir(tmp_path):
    d = tmp_path / "empty_target"
    d.mkdir()
    # empty dir is allowed without --force-rewrite
    _prevent_silent_overwrite(d, force_rewrite=False)


def test_prevent_silent_overwrite_on_non_empty_dir(tmp_path):
    d = tmp_path / "non_empty_target"
    d.mkdir()
    (d / "previous.txt").write_text("previous")
    with pytest.raises(ArchiveError, match="already exists"):
        _prevent_silent_overwrite(d, force_rewrite=False)


def test_prevent_silent_overwrite_with_force(tmp_path):
    d = tmp_path / "target"
    d.mkdir()
    (d / "previous.txt").write_text("previous")
    # Force flag bypasses.
    _prevent_silent_overwrite(d, force_rewrite=True)


# ---------------------------------------------------------------------------
# I5 == P0 same-artifact identity
# ---------------------------------------------------------------------------

def test_enforce_i5_p0_identity_match(tmp_path):
    outputs = tmp_path / "outputs"
    a = _write_source(outputs, "I5_terrain_geometry_seed42",
                      manifest_overrides={"best_epoch": 17})
    # Patch the manifest to claim both aliases.
    m = json.loads((a / "manifest.json").read_text(encoding="utf-8"))
    m["alias_ids"] = ["I5", "P0"]
    (a / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    plan_a = _plan_one(a)
    _enforce_i5_p0_identity([plan_a])  # no error


def test_enforce_i5_p0_identity_mismatch(tmp_path):
    outputs = tmp_path / "outputs"
    # I5 directory declares only its own alias; a separate P0 directory
    # declares only its own alias. The fingerprint mismatch is caught by
    # _enforce_i5_p0_identity without triggering the duplicate-alias path.
    a = _write_source(outputs, "I5_terrain_geometry_seed42",
                      manifest_overrides={"best_epoch": 17})
    b = _write_source(outputs, "P1_resconvlstm_smooth_seed42",
                      manifest_overrides={"best_epoch": 12})
    # Patch the second source's alias_ids so it claims "P0" but with a
    # different fingerprint.
    m2 = json.loads((b / "manifest.json").read_text(encoding="utf-8"))
    m2["alias_ids"] = ["P0"]
    (b / "manifest.json").write_text(json.dumps(m2), encoding="utf-8")
    plan_a = _plan_one(a)
    plan_b = _plan_one(b)
    with pytest.raises(ArchiveError, match="artifact identity"):
        _enforce_i5_p0_identity([plan_a, plan_b])


# ---------------------------------------------------------------------------
# Wrapper-format integration (real artifact schema)
# ---------------------------------------------------------------------------

def test_plan_one_requires_wrapper_result_key(tmp_path):
    """Source result_v2.json without the wrapper ``'result'`` key MUST be
    rejected — the archiver cannot adapt to it without rewriting the source
    bytes. The error message must reference ``write_v2_json``."""
    outputs = tmp_path / "outputs"
    src = outputs / "I2_resconvlstm_seed42"
    src.mkdir(parents=True)
    (src / "manifest.json").write_text(
        json.dumps(_manifest_dict(
            exp_id="I2_resconvlstm_seed42", alias_ids=["I2"])),
        encoding="utf-8",
    )
    # Write legacy un-wrapped format (top-level evaluator fields):
    inner = _result_v2_dict(flat=True)
    (src / "result_v2.json").write_text(json.dumps(inner), encoding="utf-8")
    with pytest.raises(ArchiveError,
                       match="missing required wrapper key 'result'"):
        _plan_one(src)


def test_plan_one_accepts_canonical_wrapper(tmp_path):
    """Source result_v2.json with the canonical wrapper is accepted."""
    outputs = tmp_path / "outputs" / "backbone_gate"
    src = _write_source(outputs, "I2_resconvlstm_seed42")
    plan = _plan_one(src)
    # The plan stores the unwrapped inner result (so downstream callers
    # don't have to re-unwrap).
    assert plan.result_v2["protocol_id"] == EXPECTED_PROTOCOL_ID
    assert plan.result_v2["n_events"] == EXPECTED_N_EVENTS


def test_archive_one_preserves_source_result_v2_bytes(tmp_path):
    """Byte-for-byte preservation: the archiver must copy the source
    ``result_v2.json`` as-is (no rewrite, no re-serialization)."""
    outputs = tmp_path / "outputs" / "backbone_gate"
    src = _write_source(outputs, "I2_resconvlstm_seed42")
    src_bytes = (src / "result_v2.json").read_bytes()
    plan = _plan_one(src)
    target_root = tmp_path / "results"
    archive_one(plan, target_root, force_rewrite=False)
    target = target_root / "I2_resconvlstm_seed42_v2"
    # Bytes identical to source (same indentation, same newline style).
    assert (target / "result_v2.json").read_bytes() == src_bytes


# ---------------------------------------------------------------------------
# archive_one: generation when absent
# ---------------------------------------------------------------------------

def test_archive_one_generates_metrics_csv_and_validation_md(tmp_path):
    outputs = tmp_path / "outputs" / "backbone_gate"
    src = _write_source(outputs, "I2_resconvlstm_seed42")
    plan = _plan_one(src)
    target_root = tmp_path / "results"
    info = archive_one(plan, target_root, force_rewrite=False)
    target = target_root / "I2_resconvlstm_seed42_v2"
    assert target.exists()
    files = {p.name for p in target.iterdir()}
    assert "manifest.json" in files
    assert "result_v2.json" in files
    assert "metrics_v2.csv" in files
    assert "validation.md" in files
    assert "manifest_hashes.json" in files
    # *.pth MUST NOT be present.
    assert not any(p.suffix == ".pth" for p in target.iterdir())
    assert info["canonical_name"] == "I2_resconvlstm_seed42_v2"


def test_archive_one_copies_metrics_csv_if_present(tmp_path):
    outputs = tmp_path / "outputs" / "backbone_gate"
    src = _write_source(outputs, "I3_resconvlstm_cma_seed42",
                        extra_files=[("metrics_v2.csv", "metric,value\nMAE,1.0\n")])
    plan = _plan_one(src)
    target_root = tmp_path / "results"
    archive_one(plan, target_root, force_rewrite=False)
    target = target_root / "I3_resconvlstm_cma_seed42"
    assert (target / "metrics_v2.csv").read_text() == \
        "metric,value\nMAE,1.0\n"


def test_archive_one_refuses_overwrite_without_force(tmp_path):
    outputs = tmp_path / "outputs" / "backbone_gate"
    src = _write_source(outputs, "I4_static_terrain_seed42")
    plan = _plan_one(src)
    target_root = tmp_path / "results"
    archive_one(plan, target_root, force_rewrite=False)
    # Second archive with the same canonical target must fail without --force.
    src2 = _write_source(outputs, "I4_static_terrain_seed42")
    plan2 = _plan_one(src2)
    with pytest.raises(ArchiveError, match="already exists"):
        archive_one(plan2, target_root, force_rewrite=False)


def test_archive_one_force_rewrite_allowed(tmp_path):
    outputs = tmp_path / "outputs" / "backbone_gate"
    src = _write_source(outputs, "I4_static_terrain_seed42")
    plan = _plan_one(src)
    target_root = tmp_path / "results"
    archive_one(plan, target_root, force_rewrite=False)
    # Second archive with force_rewrite succeeds.
    src2 = _write_source(outputs, "I4_static_terrain_seed42")
    plan2 = _plan_one(src2)
    info = archive_one(plan2, target_root, force_rewrite=True)
    assert info["canonical_name"] == "I4_static_terrain_seed42"


# ---------------------------------------------------------------------------
# Top-level discovery
# ---------------------------------------------------------------------------

def test_discover_source_dirs_skips_non_run_dirs(tmp_path):
    from scripts.archive_validation_results import discover_source_dirs
    outputs = tmp_path / "outputs"
    (outputs / "backbone_gate").mkdir(parents=True)
    src = _write_source(outputs / "backbone_gate", "I0_persistence_seed42")
    # junk dir without manifest.json or result_v2.json
    junk = outputs / "backbone_gate" / "scratch"
    junk.mkdir()
    (junk / "temp.txt").write_text("not a run")
    found = discover_source_dirs([outputs / "backbone_gate"])
    assert src in found
    assert junk not in found


def test_discover_source_dirs_empty_when_no_outputs(tmp_path):
    from scripts.archive_validation_results import discover_source_dirs
    assert discover_source_dirs([tmp_path / "nonexistent"]) == []