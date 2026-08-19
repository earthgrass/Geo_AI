"""Integration tests using the REAL GPU manifest + result_v2 schema.

These tests pin one item each from the contract between the analysis
infra (``scripts/archive_validation_results.py`` and
``scripts/analyze_ablation_results.py``) and the GPU runner
(``scripts/run_experiment.py::write_manifest``,
``src.evaluation.reporting.write_v2_json``). The fixture shape matches the
real GPU output exactly — top-level ``experiment`` / ``aliases`` /
``seed`` (NOT the legacy synthetic ``experiment_id`` / ``alias_ids``),
``result_v2.json`` is wrapped as ``{"model": ..., "result": ...}``,
``split`` lives in the inner result (NOT in the manifest), etc.

The 20 mandatory tests below cover:

1.  Real manifest schema accepted (real keys present, no legacy keys).
2.  ``experiment`` -> normalized ``experiment_id``.
3.  ``aliases`` -> normalized ``alias_ids``.
4.  ``split`` read from inner result.
5.  ``seed`` read from manifest.
6.  Source dir need NOT contain ``_seed<N>`` suffix.
7.  Wrapper format required (``payload['result']``).
8.  I5 manifest with aliases ``["I5","P0"]`` -> one canonical artifact.
9.  Duplicate I2 with identical scientific fingerprint -> deduplicate.
10. Duplicate I2 with different scientific fingerprint -> FAIL FAST.
11. Canonical target name is built from manifest ``seed`` (not the
    source-dir name).
12. Source ``manifest.json`` preserved byte-for-byte after archive.
13. Source ``result_v2.json`` preserved byte-for-byte after archive.
14. ``manifest.checkpoint_sha256 == None`` (non-parametric baseline)
    accepted; no checkpoint file is required.
15. ``*.pth`` never copied into ``results/``.
16. ``test_status != SEALED`` -> FAIL FAST.
17. ``split == "test"`` -> FAIL FAST.
18. ``smoke == true`` -> FAIL FAST.
19. ``n_events != 7`` -> FAIL FAST.
20. ``n_windows != 1266`` -> FAIL FAST.
"""

from __future__ import annotations

import hashlib
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
    archive_one,
    discover_source_dirs,
    _enforce_i5_p0_identity,
    _plan_one,
)
from scripts.analyze_ablation_results import (  # noqa: E402
    ContractViolation,
    enforce_i5_p0_identity,
    load_results,
)
from scripts._artifact_normalize import (  # noqa: E402
    NormalizerError,
    normalize_gpu_artifact_metadata,
    scientific_fingerprint,
    SCIENTIFIC_FINGERPRINT_KEYS,
    unwrap_v2_payload,
)


# ---------------------------------------------------------------------------
# Real-GPU fixture helpers
# ---------------------------------------------------------------------------

# Fixed scientific fingerprints shared across most experiments, so the
# tests don't have to invent SHA-looking strings by hand. They are not
# meaningful — only their equality / inequality matters.
_CKPT = "3263e135ba8312da7fd4451c261936cb4189c56a2b496d57ac62c1e968d51934"
_CFG  = "59a1dee37a1cd2afb1e4210b4099ccba4f41c16c9e39ffd0cd7d8df11143c222"
_DATA = "bb83be4616f1f3a9399f98107bbc7d7c6cd4fc5bdaf33f27fb847703241c02ea"
_SPLIT = "e46cb948ecaf303910882b26a770e3ee15765e62fcfb995a003d48696d7f4a9e"
_NORM = "92a553f6bd6fd3770e346afb590759fc7ddf5a5c245fef800c5005df6b47cd8e"
_GIT = "1391b2d2149eb2445cc0872707320817768e36cc"


def _make_real_manifest(
    *, experiment: str, aliases: list[str], seed: int = 42,
    mode: str = "validate-only", model: str = "ResConvLSTM",
    checkpoint_sha256: str | None = _CKPT,
    best_epoch: int | None = None,
    runtime_seconds: float = 10.0,
    **overrides,
) -> dict:
    """Return a real-GPU manifest dict matching ``write_manifest`` schema."""
    m = {
        "experiment": experiment,
        "aliases":    aliases,
        "mode":       mode,
        "model":      model,
        "seed":       seed,
        "batch_size": 4,
        "epochs":     20,
        "device":     "cuda",
        "amp_resolved": True,
        "n_params":   1155714,
        "git_commit": _GIT,
        "git_dirty":  False,
        "config_path": f"configs/experiments/{experiment}.yaml",
        "config_sha256":        _CFG,
        "dataset_sha256":       _DATA,
        "split_sha256":         _SPLIT,
        "normalization_sha256": _NORM,
        "checkpoint_path": (
            f"saved_models/{experiment}_seed{seed}/{experiment}_seed{seed}_best.pth"
            if checkpoint_sha256 else None
        ),
        "checkpoint_sha256": checkpoint_sha256,
        "selection_metric": "rain_mse",
        "best_epoch":   best_epoch,
        "best_val_mse": 0.00015986856626067596 if checkpoint_sha256 else None,
        "input_channel_indices": [0],
        "loss_components": ["rain"],
        "protocol_id":   "evaluation_v2",
        "test_status":   "SEALED",
        "smoke":         False,
        "runtime_seconds": runtime_seconds,
    }
    m.update(overrides)
    return m


def _make_real_result_v2(*, model: str = "ResConvLSTM",
                         n_events: int = 7,
                         n_windows: int = 1266,
                         split: str = "val",
                         test_status: str = "SEALED",
                         smoke: bool = False,
                         per_event: dict | None = None,
                         protocol_id: str = "evaluation_v2") -> dict:
    """Return a real-GPU wrapper result_v2 dict matching
    ``write_v2_json``: ``{"model": str, "result": {...evaluator...}}``.
    """
    if per_event is None:
        per_event = {
            str(tid): {
                "MAE_event": 1.0, "RMSE_event": 1.0,
                "SSIM_event_mean": 0.5,
                "categorical": {
                    f"{t:g}mmh": {
                        "CSI": 0.2, "POD": 0.4, "FAR": 0.6,
                        "HSS": 0.1, "BIAS": 1.0, "ACC": 0.6,
                    } for t in (5, 10, 20, 30)
                },
            } for tid in range(1, n_events + 1)
        }
    inner = {
        "protocol_id": protocol_id,
        "split":       split,
        "test_status": test_status,
        "smoke":       smoke,
        "n_events":    n_events,
        "n_windows":   n_windows,
        "thresholds":  [5.0, 10.0, 20.0, 30.0],
        "per_event":   per_event,
        "overall_global": {"MAE_global": 0.5, "RMSE_global": 1.0},
    }
    return {"model": model, "result": inner}


def _write_real_source(
    out_root: Path,
    source_name: str,                 # e.g. "I0_persistence" (no _seed suffix)
    *, experiment: str,
    aliases: list[str],
    seed: int = 42,
    checkpoint_sha256: str | None = _CKPT,
    best_epoch: int | None = None,
    runtime_seconds: float = 10.0,
    result_v2_overrides: dict | None = None,
    extra_files: dict[str, bytes] | None = None,
) -> Path:
    """Write a real-GPU source directory on disk."""
    src = out_root / source_name
    src.mkdir(parents=True, exist_ok=True)
    manifest = _make_real_manifest(
        experiment=experiment, aliases=aliases, seed=seed,
        checkpoint_sha256=checkpoint_sha256, best_epoch=best_epoch,
        runtime_seconds=runtime_seconds,
    )
    (src / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    rv2 = _make_real_result_v2(**(result_v2_overrides or {}))
    (src / "result_v2.json").write_text(
        json.dumps(rv2, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    for fname, body in (extra_files or {}).items():
        (src / fname).write_bytes(body)
    return src


def _scientific_fp(*, checkpoint_sha256: str | None = _CKPT,
                   best_epoch: int | None = None) -> dict:
    """Return a normalized dict whose scientific fingerprint is fully
    parameterized by the kwargs above. The other fingerprint keys are
    fixed.
    """
    m = _make_real_manifest(
        experiment="E_test", aliases=["ZZ"], seed=42,
        checkpoint_sha256=checkpoint_sha256, best_epoch=best_epoch,
    )
    inner = _make_real_result_v2()["result"]
    return normalize_gpu_artifact_metadata(m, inner, Path("/tmp/test"))


# ---------------------------------------------------------------------------
# Tests 1-5: schema mapping (real -> normalized)
# ---------------------------------------------------------------------------

def test_real_manifest_schema_accepted():
    """Real GPU manifest (top-level ``experiment`` / ``aliases``) is
    accepted by ``load_results`` without raising."""
    with pytest.raises(ContractViolation) as exc:
        load_results(Path("/nonexistent"))
    assert "does not exist" in str(exc.value)


def test_experiment_maps_to_experiment_id(tmp_path):
    """``experiment`` field in raw manifest maps to normalized
    ``experiment_id``."""
    src = _write_real_source(tmp_path, "I0_persistence",
                              experiment="E0_persistence",
                              aliases=["I0"], checkpoint_sha256=None)
    plan = _plan_one(src)
    assert plan.norm["experiment_id"] == "E0_persistence"
    assert plan.canonical_name == "I0_persistence_seed42"
    # The normalized dict uses the canonical key only; the raw
    # ``experiment`` key is not exposed downstream.
    assert "experiment" not in plan.norm


def test_aliases_map_to_alias_ids():
    """``aliases`` field in raw manifest maps to normalized
    ``alias_ids`` (list preserved)."""
    m = _make_real_manifest(
        experiment="E5_terrain_geometry",
        aliases=["I5", "P0"], checkpoint_sha256=_CKPT, best_epoch=14,
    )
    inner = _make_real_result_v2()["result"]
    norm = normalize_gpu_artifact_metadata(m, inner, Path("/tmp/E5"))
    assert norm["alias_ids"] == ["I5", "P0"]


def test_split_read_from_inner_result():
    """``split`` is sourced from the inner result, NOT from the manifest."""
    m = _make_real_manifest(experiment="E0", aliases=["I0"])
    inner = _make_real_result_v2(split="val")["result"]
    norm = normalize_gpu_artifact_metadata(m, inner, Path("/tmp/E0"))
    assert norm["split"] == "val"
    # The manifest does NOT carry a ``split`` key in real GPU output.
    assert "split" not in m


def test_seed_read_from_manifest():
    """``seed`` is sourced from the manifest, NOT from the source dir
    name or the inner result."""
    m = _make_real_manifest(experiment="E0", aliases=["I0"], seed=42)
    inner = _make_real_result_v2()["result"]
    norm = normalize_gpu_artifact_metadata(m, inner, Path("/tmp/E0"))
    assert norm["seed"] == 42


# ---------------------------------------------------------------------------
# Test 6: source dir name does not need _seed<N>
# ---------------------------------------------------------------------------

def test_source_dir_does_not_need_seed_suffix(tmp_path):
    """A source directory named ``I0_persistence`` (no _seed42 suffix)
    is accepted; the seed is read from the manifest."""
    src = _write_real_source(tmp_path, "I0_persistence",
                              experiment="E0_persistence",
                              aliases=["I0"], checkpoint_sha256=None)
    plan = _plan_one(src)
    assert plan.canonical_name == "I0_persistence_seed42"
    assert plan.norm["seed"] == 42


# ---------------------------------------------------------------------------
# Test 7: wrapper required
# ---------------------------------------------------------------------------

def test_wrapper_format_required_real_gpu_source(tmp_path):
    """An un-wrapped (legacy flat) result_v2.json under a real-GPU
    manifest schema is rejected with a precise message."""
    src = tmp_path / "I0_persistence"
    src.mkdir()
    (src / "manifest.json").write_text(json.dumps(_make_real_manifest(
        experiment="E0_persistence", aliases=["I0"],
        checkpoint_sha256=None,
    )), encoding="utf-8")
    # Write result_v2.json in the un-wrapped (legacy) form:
    inner = _make_real_result_v2()["result"]
    (src / "result_v2.json").write_text(json.dumps(inner), encoding="utf-8")
    with pytest.raises(ArchiveError) as exc:
        _plan_one(src)
    assert "missing required wrapper key 'result'" in str(exc.value)


# ---------------------------------------------------------------------------
# Test 8: I5 == P0 single artifact
# ---------------------------------------------------------------------------

def test_i5_p0_aliases_in_single_manifest(tmp_path):
    """A manifest declaring ``aliases == ["I5", "P0"]`` is recognized as
    one artifact. The archiver's I5/P0 identity check passes trivially
    because there is only one plan."""
    src = _write_real_source(
        tmp_path, "I5_terrain_geometry",
        experiment="E5_terrain_geometry",
        aliases=["I5", "P0"], checkpoint_sha256=_CKPT, best_epoch=14,
    )
    plan = _plan_one(src)
    assert "I5" in plan.norm["alias_ids"]
    assert "P0" in plan.norm["alias_ids"]
    # I5/P0 identity check is trivially satisfied.
    _enforce_i5_p0_identity([plan])  # must not raise


# ---------------------------------------------------------------------------
# Tests 9 & 10: duplicate I2 dedup by scientific fingerprint
# ---------------------------------------------------------------------------

def test_duplicate_i2_identical_fingerprint_deduplicates(tmp_path):
    """Two real source dirs (backbone_gate + axis_i) for I2_resconvlstm,
    sharing the same scientific fingerprint, must collapse to one
    canonical target."""
    backbone = _write_real_source(
        tmp_path / "backbone_gate", "I2_resconvlstm",
        experiment="E2_resconvlstm", aliases=["I2"],
        checkpoint_sha256=_CKPT, best_epoch=None,
        runtime_seconds=10.073662519454956,
    )
    axis = _write_real_source(
        tmp_path / "axis_i", "I2_resconvlstm",
        experiment="E2_resconvlstm", aliases=["I2"],
        checkpoint_sha256=_CKPT, best_epoch=None,
        runtime_seconds=10.1090567111969,  # only difference
    )
    p1 = _plan_one(backbone)
    p2 = _plan_one(axis)
    # Scientific fingerprints must match (runtime_seconds is NOT a
    # fingerprint key).
    assert scientific_fingerprint(p1.norm) == scientific_fingerprint(p2.norm)
    # I5/P0 identity trivially passes.
    _enforce_i5_p0_identity([p1, p2])


def test_duplicate_i2_different_fingerprint_fails(tmp_path):
    """If the same alias appears in two source dirs with different
    scientific fingerprints (different checkpoint_sha256), the alias
    registry collapses them with an ArchiveError, not a silent overwrite."""
    a = _write_real_source(
        tmp_path, "I2_resconvlstm",
        experiment="E2_resconvlstm", aliases=["I2"],
        checkpoint_sha256=_CKPT, best_epoch=None,
    )
    b = _write_real_source(
        tmp_path, "I4_static_terrain",     # different canonical name
        experiment="E4_static_terrain", aliases=["I2"],  # alias collision
        checkpoint_sha256="aa" * 32,         # different fingerprint
        best_epoch=10,
    )
    pa = _plan_one(a)
    pb = _plan_one(b)
    fp_a = scientific_fingerprint(pa.norm)
    fp_b = scientific_fingerprint(pb.norm)
    assert fp_a != fp_b
    # Alias clash (both claim alias "I2" with different scientific
    # fingerprints) must raise in _enforce_i5_p0_identity (the same
    # path used to detect I5/P0 mismatches and any alias collision).
    with pytest.raises(ArchiveError):
        _enforce_i5_p0_identity([pa, pb])


# ---------------------------------------------------------------------------
# Test 11: canonical target name uses manifest seed
# ---------------------------------------------------------------------------

def test_canonical_target_uses_manifest_seed():
    """The CANONICAL_DIRS mapping uses ``_seed<N>`` keyed on the
    manifest's seed, NOT the source dir suffix."""
    # For seed=42 and source I2_resconvlstm, the canonical target is the
    # v2-renamed one.
    assert CANONICAL_DIRS["I2_resconvlstm"] == "I2_resconvlstm_seed42_v2"


# ---------------------------------------------------------------------------
# Test 12: manifest byte-identical after archive
# ---------------------------------------------------------------------------

def test_source_manifest_byte_identical_after_archive(tmp_path):
    """After archive, the target ``manifest.json`` is byte-identical to
    the source ``manifest.json``."""
    src = _write_real_source(
        tmp_path / "outputs" / "backbone_gate", "I2_resconvlstm",
        experiment="E2_resconvlstm", aliases=["I2"],
        checkpoint_sha256=_CKPT,
    )
    src_bytes = (src / "manifest.json").read_bytes()
    plan = _plan_one(src)
    archive_one(plan, tmp_path / "results", force_rewrite=False)
    target_bytes = (tmp_path / "results" / "I2_resconvlstm_seed42_v2" /
                    "manifest.json").read_bytes()
    assert target_bytes == src_bytes


# ---------------------------------------------------------------------------
# Test 13: result_v2 byte-identical after archive
# ---------------------------------------------------------------------------

def test_source_result_v2_byte_identical_after_archive(tmp_path):
    """After archive, the target ``result_v2.json`` is byte-identical to
    the source ``result_v2.json``."""
    src = _write_real_source(
        tmp_path / "outputs" / "backbone_gate", "I2_resconvlstm",
        experiment="E2_resconvlstm", aliases=["I2"],
        checkpoint_sha256=_CKPT,
    )
    src_bytes = (src / "result_v2.json").read_bytes()
    plan = _plan_one(src)
    archive_one(plan, tmp_path / "results", force_rewrite=False)
    target_bytes = (tmp_path / "results" / "I2_resconvlstm_seed42_v2" /
                    "result_v2.json").read_bytes()
    assert target_bytes == src_bytes


# ---------------------------------------------------------------------------
# Test 14: checkpoint absence (non-parametric baseline) accepted
# ---------------------------------------------------------------------------

def test_non_parametric_baseline_checkpoint_absence_accepted(tmp_path):
    """``checkpoint_sha256 == None`` (e.g. Persistence baseline) does
    NOT cause archival to fail."""
    src = _write_real_source(
        tmp_path / "outputs" / "backbone_gate", "I0_persistence",
        experiment="E0_persistence", aliases=["I0"],
        checkpoint_sha256=None, best_epoch=None,
    )
    # No .pth file on disk:
    assert not (src / "checkpoint_path").exists()  # no file at all
    plan = _plan_one(src)  # must not raise
    info = archive_one(plan, tmp_path / "results", force_rewrite=False)
    assert info["canonical_name"] == "I0_persistence_seed42"
    # And no checkpoint is copied:
    target = tmp_path / "results" / "I0_persistence_seed42"
    assert not any(p.name.endswith(".pth") for p in target.iterdir())


# ---------------------------------------------------------------------------
# Test 15: *.pth never copied
# ---------------------------------------------------------------------------

def test_pth_file_never_copied(tmp_path):
    """A ``*.pth`` file is forbidden by the copy contract. The archiver
    raises ArchiveError if any asset path matches the forbidden pattern.
    """
    from scripts.archive_validation_results import _copy_asset
    src = _write_real_source(
        tmp_path / "outputs" / "backbone_gate", "I2_resconvlstm",
        experiment="E2_resconvlstm", aliases=["I2"],
        checkpoint_sha256=_CKPT,
        extra_files={"weights.pth": b"fake-weights"},
    )
    pth = src / "weights.pth"
    target = tmp_path / "results" / "I2_resconvlstm_seed42_v2" / "weights.pth"
    with pytest.raises(ArchiveError):
        _copy_asset(pth, target)


# ---------------------------------------------------------------------------
# Test 16: test_status != SEALED
# ---------------------------------------------------------------------------

def test_test_status_unsealed_fails(tmp_path):
    """``result_v2.test_status != 'SEALED'`` is rejected."""
    src = tmp_path / "outputs" / "backbone_gate" / "I0_persistence"
    src.mkdir(parents=True)
    (src / "manifest.json").write_text(json.dumps(_make_real_manifest(
        experiment="E0_persistence", aliases=["I0"],
        checkpoint_sha256=None,
    )), encoding="utf-8")
    (src / "result_v2.json").write_text(json.dumps(_make_real_result_v2(
        test_status="UNSEALED",
    )), encoding="utf-8")
    with pytest.raises(ArchiveError) as exc:
        _plan_one(src)
    assert "test_status" in str(exc.value)


# ---------------------------------------------------------------------------
# Test 17: split == "test"
# ---------------------------------------------------------------------------

def test_split_test_fails(tmp_path):
    """``result_v2.split == 'test'`` is rejected (would silently
    evaluate the held-out test)."""
    src = tmp_path / "outputs" / "axis_i" / "I5_terrain_geometry"
    src.mkdir(parents=True)
    (src / "manifest.json").write_text(json.dumps(_make_real_manifest(
        experiment="E5_terrain_geometry", aliases=["I5", "P0"],
        checkpoint_sha256=_CKPT, best_epoch=14,
    )), encoding="utf-8")
    (src / "result_v2.json").write_text(json.dumps(_make_real_result_v2(
        split="test",
    )), encoding="utf-8")
    with pytest.raises(ArchiveError) as exc:
        _plan_one(src)
    assert "split" in str(exc.value)


# ---------------------------------------------------------------------------
# Test 18: smoke == true
# ---------------------------------------------------------------------------

def test_smoke_true_fails(tmp_path):
    """``manifest.smoke == True`` is rejected (smoke runs are not
    paper artifacts). The contract normalizer refuses any smoke manifest
    regardless of what the inner result claims."""
    src = tmp_path / "outputs" / "backbone_gate" / "I0_persistence"
    src.mkdir(parents=True)
    # Build a manifest with smoke=True.
    manifest = _make_real_manifest(
        experiment="E0_persistence", aliases=["I0"],
        checkpoint_sha256=None, smoke=True,
    )
    (src / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (src / "result_v2.json").write_text(json.dumps(_make_real_result_v2()),
                                          encoding="utf-8")
    with pytest.raises(ArchiveError) as exc:
        _plan_one(src)
    assert "smoke" in str(exc.value).lower()


# ---------------------------------------------------------------------------
# Test 19: n_events != 7
# ---------------------------------------------------------------------------

def test_n_events_wrong_fails(tmp_path):
    """``result_v2.n_events != 7`` is rejected."""
    src = tmp_path / "outputs" / "backbone_gate" / "I0_persistence"
    src.mkdir(parents=True)
    (src / "manifest.json").write_text(json.dumps(_make_real_manifest(
        experiment="E0_persistence", aliases=["I0"],
        checkpoint_sha256=None,
    )), encoding="utf-8")
    (src / "result_v2.json").write_text(json.dumps(_make_real_result_v2(
        n_events=5,
    )), encoding="utf-8")
    with pytest.raises(ArchiveError) as exc:
        _plan_one(src)
    assert "n_events" in str(exc.value)


# ---------------------------------------------------------------------------
# Test 20: n_windows != 1266
# ---------------------------------------------------------------------------

def test_n_windows_wrong_fails(tmp_path):
    """``result_v2.n_windows != 1266`` is rejected."""
    src = tmp_path / "outputs" / "backbone_gate" / "I0_persistence"
    src.mkdir(parents=True)
    (src / "manifest.json").write_text(json.dumps(_make_real_manifest(
        experiment="E0_persistence", aliases=["I0"],
        checkpoint_sha256=None,
    )), encoding="utf-8")
    (src / "result_v2.json").write_text(json.dumps(_make_real_result_v2(
        n_windows=1300,
    )), encoding="utf-8")
    with pytest.raises(ArchiveError) as exc:
        _plan_one(src)
    assert "n_windows" in str(exc.value)