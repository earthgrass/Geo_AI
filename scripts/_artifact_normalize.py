"""Shared helpers for normalizing real GPU artifact metadata.

The real GPU runner (``scripts/run_experiment.py::write_manifest`` and
``src.evaluation.reporting.write_v2_json``) writes artifacts with the
following schema:

    manifest.json (raw):
        experiment           : str   (e.g. "E2_resconvlstm")
        aliases              : list[str]   (e.g. ["I2"] or ["I5", "P0"])
        mode                 : str   ("train" / "validate-only")
        model                : str
        seed                 : int
        batch_size           : int
        epochs               : int
        device               : str
        amp_resolved         : bool
        n_params             : int
        git_commit           : str
        git_dirty            : bool
        config_path          : str
        config_sha256        : str
        dataset_sha256       : str
        split_sha256         : str
        normalization_sha256 : str
        checkpoint_path      : str | null   (null for non-parametric baselines)
        checkpoint_sha256    : str | null
        selection_metric     : str
        best_epoch           : int | null
        best_val_mse         : float | null
        input_channel_indices: list[int]
        loss_components      : list[str]
        protocol_id          : "evaluation_v2"
        test_status          : "SEALED"
        smoke                : false
        runtime_seconds      : float

    result_v2.json (wrapper from write_v2_json):
        {
          "model": "<model_name>",
          "result": {
            "protocol_id": "evaluation_v2",
            "split": "val",
            "test_status": "SEALED",
            "thresholds": [5.0, 10.0, 20.0, 30.0],
            "n_events": 7,
            "n_windows": 1266,
            ...
            "per_event": {...},
            "overall_global": {...},
            ...
          }
        }

The legacy synthetic schema used internally by the analysis infra (which
mapped directly to columns like ``experiment_id`` / ``alias_ids`` /
``manifest.split``) is NOT the schema the GPU runner writes. We therefore
translate raw -> normalized exactly once at the ingest boundary, and every
downstream consumer sees a stable canonical shape.

Important invariants:

* The original ``manifest.json`` and ``result_v2.json`` bytes are NEVER
  rewritten. ``normalize_gpu_artifact_metadata`` is a pure read-only
  helper.
* ``split`` is read from ``inner_result['split']``, NOT from the manifest
  (the manifest does not carry ``split``).
* ``n_events`` / ``n_windows`` are read from the inner result, NOT from
  the manifest.
* ``checkpoint_sha256`` may be ``None`` for non-parametric baselines
  (e.g. ``Persistence`` / ``I0``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


class NormalizerError(AssertionError):
    """Raised when an artifact cannot be normalized."""


# ---------------------------------------------------------------------------
# Wrapper unwrap (the v2 wrapper is {"model": str, "result": <inner>})
# ---------------------------------------------------------------------------

def unwrap_v2_payload(payload: Any, source: Path,
                       error_cls=NormalizerError) -> Tuple[Dict, Dict]:
    """Unwrap a ``result_v2.json`` payload written by ``write_v2_json``.

    Returns ``(inner_result, wrapper_payload)``. Fails fast on legacy
    un-wrapped format or malformed input.
    """
    if not isinstance(payload, dict):
        raise error_cls(f"{source}: payload is not a JSON object.")
    if "result" not in payload:
        raise error_cls(
            f"{source}: missing required wrapper key 'result'. "
            f"This file must be written by "
            f"src.evaluation.reporting.write_v2_json (which emits a "
            f"{{'model': ..., 'result': ...}} wrapper). Legacy "
            f"un-wrapped format (top-level protocol_id/split/n_events) "
            f"is NOT accepted."
        )
    inner = payload["result"]
    if not isinstance(inner, dict):
        raise error_cls(f"{source}: payload['result'] is not a JSON object.")
    return inner, payload


# ---------------------------------------------------------------------------
# Manifest field map (real schema -> canonical internal field names)
# ---------------------------------------------------------------------------

# These are the canonical names the analysis infra uses downstream. The
# script reads ONLY these names; never raw GPU names downstream.
CANONICAL_MANIFEST_KEYS = (
    "experiment_id",          # <- raw['experiment']
    "alias_ids",              # <- raw['aliases']
    "mode",
    "model",
    "seed",
    "batch_size",
    "epochs",
    "device",
    "amp_resolved",
    "n_params",
    "git_commit",
    "git_dirty",
    "config_path",
    "config_sha256",
    "dataset_sha256",
    "split_sha256",
    "normalization_sha256",
    "checkpoint_path",
    "checkpoint_sha256",      # nullable
    "selection_metric",
    "best_epoch",             # nullable
    "best_val_mse",           # nullable
    "input_channel_indices",
    "loss_components",
    "protocol_id",
    "test_status",
    "smoke",
    "runtime_seconds",
)

# Required keys on the RAW manifest (split is NOT one of these — split
# comes from the inner result). checkpoint_* is allowed to be null because
# non-parametric baselines (e.g. Persistence) have no checkpoint.
_REQUIRED_RAW_MANIFEST_KEYS = (
    "experiment", "aliases", "mode", "model", "seed", "epochs",
    "git_commit", "config_path", "config_sha256",
    "dataset_sha256", "split_sha256", "normalization_sha256",
    "selection_metric", "protocol_id", "test_status", "smoke",
)


# ---------------------------------------------------------------------------
# Normalizer
# ---------------------------------------------------------------------------

def normalize_gpu_artifact_metadata(
    raw_manifest: Dict,
    inner_result: Dict,
    source_dir: Path,
    error_cls=NormalizerError,
) -> Dict:
    """Convert real GPU manifest + inner result into a canonical dict.

    Required keys in the raw manifest are checked for presence; missing
    keys raise ``error_cls`` (defaults to ``NormalizerError``). The
    ``checkpoint_sha256``, ``checkpoint_path``, ``best_epoch`` and
    ``best_val_mse`` fields are allowed to be ``None`` for non-parametric
    baselines.

    Returns a dict whose keys are exactly ``CANONICAL_MANIFEST_KEYS`` plus
    ``split``, ``n_events``, ``n_windows`` (which come from the inner
    result). Callers downstream must use ONLY these canonical keys.
    """
    missing = [k for k in _REQUIRED_RAW_MANIFEST_KEYS if k not in raw_manifest]
    if missing:
        raise error_cls(
            f"{source_dir}: manifest is missing required raw keys: {missing}. "
            f"This manifest was not produced by the real GPU runner."
        )

    # Cross-validate manifest vs inner result for the fields both sides
    # claim. If they disagree, the artifact is internally inconsistent;
    # refuse to silently coerce. The manifest is treated as the runner's
    # claim and the inner result as the evaluator's claim; both must say
    # the same thing.
    for k in ("protocol_id", "test_status"):
        mv = raw_manifest.get(k)
        iv = inner_result.get(k)
        if mv is None or iv is None:
            raise error_cls(
                f"{source_dir}: '{k}' must be present in both manifest and "
                f"inner result_v2; got manifest={mv!r}, inner={iv!r}."
            )
        if mv != iv:
            raise error_cls(
                f"{source_dir}: manifest.{k}={mv!r} disagrees with "
                f"result_v2.{k}={iv!r}. The runner's manifest and the "
                f"evaluator's inner result must agree on '{k}'."
            )

    # Smoke: manifest must be False. (The evaluator never runs a smoke
    # validation on a paper artifact; if it did, both sides would carry
    # ``True``.)
    if raw_manifest.get("smoke") is not False:
        raise error_cls(
            f"{source_dir}: manifest.smoke must equal False; got "
            f"{raw_manifest.get('smoke')!r}. Smoke validation outputs are "
            f"not paper artifacts."
        )

    aliases = list(raw_manifest.get("aliases") or [])
    # Empty aliases are allowed: B1 (TrajGRU) is a backbone sanity
    # baseline and does not participate in the formal I5≡P0 / I2 dup
    # alias registry. The archiver resolves its canonical name from
    # CANONICAL_DIRS via the source dir name; the analyzer just sees
    # it as a row with no formal alias.

    out: Dict[str, Any] = {
        "experiment_id":          raw_manifest["experiment"],
        "alias_ids":              aliases,
        "mode":                   raw_manifest["mode"],
        "model":                  raw_manifest["model"],
        "seed":                   raw_manifest["seed"],
        "batch_size":             raw_manifest["batch_size"],
        "epochs":                 raw_manifest["epochs"],
        "device":                 raw_manifest.get("device"),
        "amp_resolved":           raw_manifest.get("amp_resolved"),
        "n_params":               raw_manifest.get("n_params"),
        "git_commit":             raw_manifest["git_commit"],
        "git_dirty":              raw_manifest.get("git_dirty"),
        "config_path":            raw_manifest["config_path"],
        "config_sha256":          raw_manifest["config_sha256"],
        "dataset_sha256":         raw_manifest["dataset_sha256"],
        "split_sha256":           raw_manifest["split_sha256"],
        "normalization_sha256":   raw_manifest["normalization_sha256"],
        "checkpoint_path":        raw_manifest.get("checkpoint_path"),
        "checkpoint_sha256":      raw_manifest.get("checkpoint_sha256"),
        "selection_metric":       raw_manifest["selection_metric"],
        "best_epoch":             raw_manifest.get("best_epoch"),
        "best_val_mse":           raw_manifest.get("best_val_mse"),
        "input_channel_indices":  list(raw_manifest.get("input_channel_indices") or []),
        "loss_components":        list(raw_manifest.get("loss_components") or []),
        "protocol_id":            raw_manifest["protocol_id"],
        "test_status":            raw_manifest["test_status"],
        "smoke":                  raw_manifest["smoke"],
        "runtime_seconds":        raw_manifest.get("runtime_seconds"),

        # From inner result_v2 payload:
        "split":                  inner_result.get("split"),
        "n_events":               inner_result.get("n_events"),
        "n_windows":              inner_result.get("n_windows"),
        "thresholds":             list(inner_result.get("thresholds") or []),
        "per_event":              inner_result.get("per_event"),
        "overall_global":         inner_result.get("overall_global"),
    }
    return out


# ---------------------------------------------------------------------------
# Scientific fingerprint (for I5=P0 identity and I2 duplicate dedup)
# ---------------------------------------------------------------------------

# These are the fields whose equality defines "same artifact". They DO
# NOT include ``runtime_seconds`` (which is wall-clock dependent) and
# they DO NOT include ``git_dirty`` (which is a transient property of
# the runner's working tree).
SCIENTIFIC_FINGERPRINT_KEYS: Tuple[str, ...] = (
    "checkpoint_sha256",
    "config_sha256",
    "dataset_sha256",
    "split_sha256",
    "normalization_sha256",
    "git_commit",
    "epochs",
    "best_epoch",
)


def scientific_fingerprint(norm: Dict) -> Tuple:
    """Return the scientific fingerprint tuple for a normalized manifest.

    Used to decide whether two artifacts are the SAME scientific object
    (e.g. I5 ≡ P0, or the duplicate I2 reused in both backbone_gate and
    axis_i).
    """
    return tuple(norm.get(k) for k in SCIENTIFIC_FINGERPRINT_KEYS)


# ---------------------------------------------------------------------------
# Minimal raw loader
# ---------------------------------------------------------------------------

def load_raw_manifest_and_result(
    source_dir: Path,
    error_cls=NormalizerError,
) -> Tuple[Dict, Dict, Dict, Dict]:
    """Load and unwrap one GPU artifact directory.

    Returns ``(raw_manifest, wrapper_payload, inner_result, normalized)``.
    Raises ``error_cls`` on any contract failure.
    """
    manifest_path = source_dir / "manifest.json"
    result_v2_path = source_dir / "result_v2.json"
    if not manifest_path.exists():
        raise error_cls(f"{source_dir}: missing manifest.json.")
    if not result_v2_path.exists():
        raise error_cls(f"{source_dir}: missing result_v2.json.")
    try:
        raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise error_cls(f"{source_dir}: manifest.json is not valid JSON: {e}")
    try:
        wrapper = json.loads(result_v2_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise error_cls(f"{source_dir}: result_v2.json is not valid JSON: {e}")
    inner_result, _ = unwrap_v2_payload(wrapper, result_v2_path, error_cls)
    normalized = normalize_gpu_artifact_metadata(
        raw_manifest, inner_result, source_dir, error_cls
    )
    return raw_manifest, wrapper, inner_result, normalized