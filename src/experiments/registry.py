"""Frozen Design-C experiment registry — single source of truth.

Provides:
  - the canonical E->I/P alias mapping (``configs/experiment_aliases_v2.yaml``);
  - strict validation of experiment YAMLs against the frozen common controls
    (docs/MINIMAX_IMPLEMENTATION_SPEC.md §6): seed/batch/epoch/channel/model
    dims/loss weights are enforced, unknown components rejected, P4/P5 refused;
  - a deterministic config fingerprint (SHA-256) for artifact verification.

The alias registry must never contain a runnable P4/P5 row.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EXPERIMENTS_DIR = REPO_ROOT / "configs" / "experiments"
ALIAS_REGISTRY_PATH = REPO_ROOT / "configs" / "experiment_aliases_v2.yaml"

CANONICAL_CHANNEL_RANGE = range(12)
ALLOWED_MODELS = {"Persistence", "PlainConvLSTM", "ResConvLSTM", "TrajGRU"}
ALLOWED_COMPONENTS = {"rain", "smooth", "extreme"}

# Frozen common controls (spec §6). Deviations are rejected for formal mode.
FROZEN_COMMON: Dict[str, Any] = {
    "data.seq_len": 11,
    "data.precip_vmax": 100.0,
    "model.hidden_dims": [64, 128],
    "model.kernel_size": 3,
    "training.batch_size": 4,
    "training.epochs": 20,
    "training.learning_rate": 0.0001,
    "training.weight_decay": 0.0001,
    "training.lr_patience": 10,
    "training.early_stopping_patience": 10,
    "training.grad_clip_norm": 1.0,
    "training.seed": 42,
    "training.use_amp": "auto",
    "training.checkpoint_selection_metric": "rain_mse",
    "physics_loss.enabled": True,
    "physics_loss.lambda_smooth": 0.01,
    "physics_loss.lambda_extreme": 0.5,
    "physics_loss.extreme_threshold": 10.0,
    "physics_loss.orographic.enabled": False,
}

REQUIRED_KEYS = [
    "data.h5_path", "data.split_path", "data.normalization_path",
    "model.name", "model.input_channel_indices",
    "physics_loss.components",
]


# ---------------------------------------------------------------------------
# Alias registry
# ---------------------------------------------------------------------------

def load_alias_registry(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load the frozen alias registry, failing on a runnable P4/P5 row."""
    path = Path(path) if path is not None else ALIAS_REGISTRY_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Alias registry not found at {path}. It is required to resolve "
            "Axis I / Axis II experiment identities."
        )
    with open(path, "r", encoding="utf-8") as f:
        reg = yaml.safe_load(f)

    aliases = reg.get("aliases", {})
    blocked = reg.get("blocked", {})
    for pid in ("P4", "P5"):
        if pid in aliases:
            raise ValueError(
                f"Alias registry must not map {pid} to a runnable config; "
                f"it is blocked ({blocked.get(pid, 'BLOCKED')})."
            )
    return reg


def resolve_alias(alias_or_stem: str, registry: Optional[Dict] = None) -> str:
    """Resolve an Axis alias (I3, P0, ...) or canonical stem to a config stem.

    ``I5`` and ``P0`` both resolve to ``E5_terrain_geometry`` (one artifact).
    """
    reg = registry if registry is not None else load_alias_registry()
    aliases = reg.get("aliases", {})
    if alias_or_stem in aliases:
        return aliases[alias_or_stem]
    # Not an alias: accept a canonical config stem directly.
    return alias_or_stem


def config_path_for(stem: str) -> Path:
    """Return the canonical config path for a config stem."""
    path = EXPERIMENTS_DIR / f"{stem}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Experiment config not found: {path}")
    return path


def aliases_for_stem(stem: str, registry: Optional[Dict] = None) -> List[str]:
    """Return every Axis alias pointing at this canonical config stem."""
    reg = registry if registry is not None else load_alias_registry()
    return sorted(
        a for a, s in reg.get("aliases", {}).items() if s == stem
    )


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

def _get_path(cfg: Dict, dotted: str) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def validate_formal_config(cfg: Dict, config_path: str = "<config>") -> Dict:
    """Validate one experiment config against the frozen matrix.

    Raises ValueError on any deviation. Returns the config unchanged on success.
    """
    for key in REQUIRED_KEYS:
        if _get_path(cfg, key) is None:
            raise ValueError(
                f"{config_path}: missing required field '{key}'."
            )

    # ---- frozen common controls ----
    violations = []
    for dotted, expected in FROZEN_COMMON.items():
        got = _get_path(cfg, dotted)
        if got is None:
            violations.append(f"missing '{dotted}'")
        elif got != expected:
            violations.append(
                f"'{dotted}' = {got!r}, expected {expected!r}"
            )
    if violations:
        raise ValueError(
            f"{config_path}: frozen common controls violated: "
            + "; ".join(violations)
        )

    model_name = cfg["model"]["name"]
    channels = cfg["model"]["input_channel_indices"]
    components = cfg["physics_loss"]["components"]

    # ---- model / channels ----
    if model_name not in ALLOWED_MODELS:
        raise ValueError(
            f"{config_path}: unknown model {model_name!r}; allowed "
            f"{sorted(ALLOWED_MODELS)}."
        )
    if not isinstance(channels, list) or not channels:
        raise ValueError(f"{config_path}: input_channel_indices must be non-empty.")
    if not channels[0] == 0:
        raise ValueError(
            f"{config_path}: precipitation (canonical 0) must be subset "
            "position zero."
        )
    if len(set(channels)) != len(channels):
        raise ValueError(f"{config_path}: duplicate canonical channels {channels}.")
    for c in channels:
        if c not in CANONICAL_CHANNEL_RANGE:
            raise ValueError(
                f"{config_path}: channel {c} outside canonical 0..11."
            )
    if "oro" in components:
        raise ValueError(
            f"{config_path}: orographic component is not part of Design C "
            "(P4/P5 are BLOCKED_BY_ENVIRONMENTAL_WIND_DATA)."
        )
    unknown = set(components) - ALLOWED_COMPONENTS
    if unknown:
        raise ValueError(
            f"{config_path}: unknown loss components {sorted(unknown)}; "
            f"allowed {sorted(ALLOWED_COMPONENTS)}."
        )
    if model_name == "Persistence" and components != ["rain"]:
        raise ValueError(f"{config_path}: Persistence is non-trainable (rain only).")

    # ---- per-experiment loss weights ----
    if "smooth" in components and not components[0] == "rain":
        raise ValueError(f"{config_path}: 'rain' must be the base component.")
    return cfg


def config_fingerprint(cfg: Dict) -> str:
    """Deterministic SHA-256 over the canonicalized config (JSON, sorted)."""
    canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":"),
                           default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def fingerprint_file(path: Path) -> str:
    """SHA-256 of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()
