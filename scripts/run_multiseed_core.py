"""CORE multi-seed runner for Design-C (Phase 1 — multi-seed execution prep).

This script orchestrates the 4-new-seed × 7-CORE-experiment = 28-run matrix
defined in ``docs/MULTISEED_PROTOCOL_FREEZE.md`` §4.1 and the
``Geo_AI_MULTI_AGENT_RESEARCH_ORCHESTRATION.md`` Phase 1 specification.

Scientific contract (every line below is enforced — no exception):
  - ``seed`` is the ONLY scientific field allowed to change across runs.
    All other frozen common controls (``FROZEN_COMMON``) are verified
    against the canonical YAML before each run; only ``training.seed``
    is then overridden.
  - Test split is NEVER loaded. ``evaluate_model_v2`` is always called
    with ``split="val"`` and ``test_status="SEALED"``. The runner
    asserts at import time that the allowed-split allow-list still
    excludes ``"test"``.
  - The runner is resumable: a run directory is "compatible complete"
    iff it carries both ``manifest.json`` and ``result_v2.json`` and
    every contract check below passes. Incomplete runs are reported
    (NEVER silently skipped).
  - I5 and P0 resolve to the same canonical config
    (``E5_terrain_geometry``) and the same per-seed artifact directory.
    They are trained ONCE per seed (the alias ``["I5", "P0"]`` is
    recorded on the manifest).

This file does not modify any frozen scientific artifact. It only
creates per-seed directories under ``<output_root>/seed_<s>/<alias>/``.
By default the runner is invoked as:

    python scripts/run_multiseed_core.py --dry-run
    python scripts/run_multiseed_core.py --verify-only
    python scripts/run_multiseed_core.py --execute   # Phase 2 — NOT used yet

The 28-row execution plan it produces is consumed by
``deliverables/MULTISEED_EXECUTION_PLAN.md`` and by Phase 2 launch.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for p in (REPO_ROOT, str(REPO_ROOT)):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Import the contract we are not allowed to break.
from src.experiments.registry import (  # noqa: E402
    ALIAS_REGISTRY_PATH,
    EXPERIMENTS_DIR,
    FROZEN_COMMON,
    aliases_for_stem,
    config_fingerprint,
    fingerprint_file,
    load_alias_registry,
    resolve_alias,
)
from scripts.run_experiment import (  # noqa: E402
    DEFAULT_H5,
    SPLIT_PATH,
    NORM_PATH,
    load_split,
    make_loader,
    make_model,
    resolve_use_amp,
    write_manifest,
    _git_commit,
    _git_dirty,
)
from src.training.trainer import Trainer, set_seed  # noqa: E402
from src.evaluation.evaluator import evaluate_model_v2  # noqa: E402
from src.evaluation.reporting import (  # noqa: E402
    write_v2_json,
    write_v2_csv,
    write_v2_markdown,
)


# ---------------------------------------------------------------------------
# Frozen scientific contract — read-only references to the protocol freeze.
# ---------------------------------------------------------------------------

#: Multi-seed protocol §2: seeds = {42, 123, 2024, 7, 31415}.
DEFAULT_SEEDS: Tuple[int, ...] = (123, 2024, 7, 31415)
SEED_CANONICAL = 42

#: CORE experiments per MULTISEED_PROTOCOL_FREEZE.md §4.1.
#: I5 ≡ P0 — single artifact. Aliases are recorded on the manifest.
DEFAULT_EXPERIMENTS: Tuple[str, ...] = ("I2", "I3", "I4", "I5", "P1", "P2", "P3")

#: Single allowed evaluator split — test is FORBIDDEN.
ALLOWED_SPLITS = ("train", "val")
EVALUATOR_SPLIT = "val"
EVALUATOR_TEST_STATUS = "SEALED"

#: Evaluator v2 contract (these numbers come from canonical seed-42 audit).
EXPECTED_N_EVENTS = 7
EXPECTED_N_WINDOWS = 1266
EXPECTED_BATCH_SIZE = 4
EXPECTED_EPOCHS = 20

#: v2 categorical thresholds.
THRESHOLDS_MMH = (5.0, 10.0, 20.0, 30.0)


# ---------------------------------------------------------------------------
# Frozen common control enforcement (seed is the only allowed override)
# ---------------------------------------------------------------------------

#: Frozen common fields whose value MUST equal the on-disk YAML.
#: ``training.seed`` is excluded — it is the only scientific override.
FROZEN_COMMON_FIELDS: Tuple[str, ...] = tuple(
    k for k in FROZEN_COMMON.keys() if k != "training.seed"
)


def _get_path(cfg: Dict, dotted: str) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def validate_seed_only_override(cfg: Dict, config_path: str) -> Tuple[bool, List[str]]:
    """Verify every frozen common field except ``training.seed`` matches.

    The multi-seed runner is the only path that overrides ``training.seed``.
    Every other field MUST equal the value frozen in
    ``src.experiments.registry.FROZEN_COMMON``. This is the inline equivalent
    of ``validate_formal_config`` restricted to seed-invariance.
    """
    violations: List[str] = []
    for dotted in FROZEN_COMMON_FIELDS:
        expected = FROZEN_COMMON[dotted]
        got = _get_path(cfg, dotted)
        if got is None:
            violations.append(f"missing '{dotted}'")
        elif got != expected:
            violations.append(
                f"'{dotted}' = {got!r}, expected {expected!r} "
                "(seed-only override forbids this deviation)"
            )
    return (len(violations) == 0), violations


def assert_test_seal() -> None:
    """Hard fail at import time if the test-seal invariants are broken.

    This guards the runner against any future edit to
    ``scripts.evaluate_checkpoint`` that widens the allowed-split allow-list,
    or any edit to ``run_experiment.py`` that drops the ``SEALED`` hardcode.
    """
    if "test" in ALLOWED_SPLITS:
        raise RuntimeError(
            "Test seal broken: 'test' is in ALLOWED_SPLITS. "
            "Refusing to start the multi-seed runner."
        )
    if EVALUATOR_SPLIT == "test":
        raise RuntimeError(
            "Test seal broken: EVALUATOR_SPLIT == 'test'. "
            "Refusing to start the multi-seed runner."
        )
    if EVALUATOR_TEST_STATUS != "SEALED":
        raise RuntimeError(
            "Test seal broken: EVALUATOR_TEST_STATUS != 'SEALED'."
        )


# ---------------------------------------------------------------------------
# Run matrix construction
# ---------------------------------------------------------------------------

def build_run_matrix(
    seeds: Tuple[int, ...],
    experiments: Tuple[str, ...],
    registry: Dict,
) -> List[Dict[str, Any]]:
    """Build the deterministic run matrix from seeds × experiments.

    Each row contains: run_id, seed, experiment_alias, canonical_stem,
    config_path, channels, loss_components, output_dir.
    """
    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        for alias in experiments:
            stem = resolve_alias(alias, registry)
            cfg_path = EXPERIMENTS_DIR / f"{stem}.yaml"
            if not cfg_path.exists():
                raise FileNotFoundError(
                    f"Canonical config not found for alias {alias!r} → "
                    f"{stem!r}: {cfg_path}"
                )
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
            channels = list(cfg["model"]["input_channel_indices"])
            components = list(cfg["physics_loss"]["components"])
            model_name = cfg["model"]["name"]
            all_aliases = aliases_for_stem(stem, registry)
            rows.append({
                "run_id": f"ms_seed{seed}_{alias}",
                "seed": int(seed),
                "experiment_alias": alias,
                "canonical_stem": stem,
                "config_path": str(cfg_path),
                "config_sha256": config_fingerprint(cfg),
                "model_name": model_name,
                "channels": channels,
                "loss_components": components,
                "aliases_on_manifest": all_aliases,
            })
    return rows


# ---------------------------------------------------------------------------
# Compatibility check (skip-completed decision)
# ---------------------------------------------------------------------------

def check_compatibility(out_dir: Path, row: Dict[str, Any]) -> Dict[str, Any]:
    """Decide whether ``out_dir`` holds a compatible completed run.

    Returns a structured dict with status ∈ {COMPLETE, INCOMPLETE, MISSING}
    and the list of failed checks. NEVER silently skips — an INCOMPLETE
    result is reported as such.
    """
    manifest_path = out_dir / "manifest.json"
    result_path = out_dir / "result_v2.json"

    if not manifest_path.exists() or not result_path.exists():
        return {
            "status": "MISSING",
            "reason": "manifest.json or result_v2.json not present",
            "failed_checks": ["manifest_or_result_missing"],
            "out_dir": str(out_dir),
        }

    try:
        man = json.loads(manifest_path.read_text(encoding="utf-8"))
        res = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "INCOMPLETE",
            "reason": f"json parse error: {type(exc).__name__}: {exc}",
            "failed_checks": ["json_parse_error"],
            "out_dir": str(out_dir),
        }

    inner = res.get("result", res)
    failures: List[str] = []

    # manifest contract
    if man.get("experiment") != row["canonical_stem"]:
        failures.append("manifest.experiment != canonical_stem")
    if man.get("seed") != row["seed"]:
        failures.append(f"manifest.seed != {row['seed']}")
    if man.get("batch_size") != EXPECTED_BATCH_SIZE:
        failures.append("manifest.batch_size != 4")
    if man.get("epochs") != EXPECTED_EPOCHS:
        failures.append("manifest.epochs != 20")
    if man.get("test_status") != "SEALED":
        failures.append("manifest.test_status != SEALED")
    if man.get("smoke", False) is not False:
        failures.append("manifest.smoke != false")
    if man.get("input_channel_indices") != row["channels"]:
        failures.append("manifest.channels != expected channels")
    if man.get("loss_components") != row["loss_components"]:
        failures.append("manifest.loss_components != expected components")
    if man.get("protocol_id") != "evaluation_v2":
        failures.append("manifest.protocol_id != evaluation_v2")
    if man.get("model") != row["model_name"]:
        failures.append("manifest.model != expected model")

    # inner v2 contract
    if inner.get("protocol_id") != "evaluation_v2":
        failures.append("result_v2.protocol_id != evaluation_v2")
    if inner.get("test_status") != "SEALED":
        failures.append("result_v2.test_status != SEALED")
    if inner.get("split") != "val":
        failures.append("result_v2.split != val")
    if inner.get("n_events") != EXPECTED_N_EVENTS:
        failures.append(
            f"result_v2.n_events != {EXPECTED_N_EVENTS} (got "
            f"{inner.get('n_events')!r})"
        )
    if inner.get("n_windows") != EXPECTED_N_WINDOWS:
        failures.append(
            f"result_v2.n_windows != {EXPECTED_N_WINDOWS} (got "
            f"{inner.get('n_windows')!r})"
        )

    if failures:
        return {
            "status": "INCOMPLETE",
            "reason": "contract checks failed",
            "failed_checks": failures,
            "out_dir": str(out_dir),
        }
    return {
        "status": "COMPLETE",
        "reason": "all contract checks passed",
        "failed_checks": [],
        "out_dir": str(out_dir),
        "manifest_seed": man.get("seed"),
        "runtime_seconds": man.get("runtime_seconds"),
    }


# ---------------------------------------------------------------------------
# Per-run training and evaluation (used by --execute; NOT by Phase 1)
# ---------------------------------------------------------------------------

def run_one_row(row: Dict[str, Any], output_root: Path,
                num_workers: Optional[int]) -> Dict[str, Any]:
    """Train + validate one row. Used only after Phase 1 sign-off.

    Always writes 4 artifacts: manifest.json, result_v2.json,
    metrics_v2.csv, validation.md. Test split is never constructed.
    """
    out_dir = output_root / f"seed_{row['seed']}" / row["experiment_alias"]
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out_dir / "models"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Load canonical YAML; verify seed is the only allowed override.
    cfg = yaml.safe_load(Path(row["config_path"]).read_text(encoding="utf-8"))
    ok, violations = validate_seed_only_override(cfg, row["config_path"])
    if not ok:
        raise RuntimeError(
            f"Row {row['run_id']}: frozen common controls violated "
            f"(seed-only override forbids these deviations): "
            + "; ".join(violations)
        )
    # The single allowed scientific override.
    cfg["training"]["seed"] = int(row["seed"])

    set_seed(int(row["seed"]))

    import torch
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    workers = num_workers if num_workers is not None else (4 if cuda else 0)
    use_amp = resolve_use_amp(cfg["training"].get("use_amp"), cuda)

    train_ids, val_ids, _test_ids = load_split(cfg["data"]["split_path"])
    stats = json.load(open(cfg["data"]["normalization_path"], encoding="utf-8"))
    h5_path = cfg["data"]["h5_path"]
    precip_vmax = float(cfg["data"]["precip_vmax"])
    channel_indices = list(cfg["model"]["input_channel_indices"])

    _, val_loader = make_loader(
        h5_path, val_ids, channel_indices, stats, EXPECTED_BATCH_SIZE,
        workers, shuffle=False, pin_memory=cuda,
    )

    model = make_model(
        cfg["model"]["name"], channel_indices,
        list(cfg["model"]["hidden_dims"]), cfg["model"]["kernel_size"],
    )
    is_module = isinstance(model, torch.nn.Module)
    n_params = sum(x.numel() for x in model.parameters()) if is_module else 0
    if is_module:
        model = model.to(device)

    trainer_config = {
        "model_name": cfg["model"]["name"],
        "use_physics_loss": True,
        "normalize_precip": False,
        "precip_vmax": precip_vmax,
        "physics_loss": {
            "lambda_smooth": cfg["physics_loss"]["lambda_smooth"],
            "lambda_extreme": cfg["physics_loss"]["lambda_extreme"],
            "extreme_threshold": (
                cfg["physics_loss"]["extreme_threshold"] / precip_vmax
            ),
            "components": list(cfg["physics_loss"]["components"]),
        },
        "learning_rate": cfg["training"]["learning_rate"],
        "weight_decay": cfg["training"]["weight_decay"],
        "lr_patience": cfg["training"]["lr_patience"],
        "early_stopping_patience": cfg["training"]["early_stopping_patience"],
        "grad_clip_norm": cfg["training"]["grad_clip_norm"],
        "use_amp": use_amp,
        "checkpoint_selection_metric":
            cfg["training"]["checkpoint_selection_metric"],
        "checkpoint_dir": str(checkpoint_dir),
        "log_dir": str(log_dir),
    }
    _, train_loader = make_loader(
        h5_path, train_ids, channel_indices, stats, EXPECTED_BATCH_SIZE,
        workers, shuffle=True, pin_memory=cuda,
    )

    t0 = time.time()
    trainer = Trainer(model, train_loader, val_loader, trainer_config)
    trainer.train(epochs=EXPECTED_EPOCHS)
    train_runtime_s = time.time() - t0
    best_epoch = trainer.best_epoch + 1
    best_val_mse = trainer.best_val_mse
    checkpoint_path = str(checkpoint_dir / f"{cfg['model']['name']}_best.pth")
    model = model.to(device)

    t0 = time.time()
    result = evaluate_model_v2(
        model, val_loader, device, precip_vmax=precip_vmax,
        thresholds=list(THRESHOLDS_MMH),
        channel_indices=channel_indices,
        split=EVALUATOR_SPLIT, test_status=EVALUATOR_TEST_STATUS,
    )
    eval_runtime_s = time.time() - t0
    total_runtime_s = train_runtime_s + eval_runtime_s

    write_v2_json(result, out_dir / "result_v2.json",
                  model_name=cfg["model"]["name"])
    write_v2_csv(result, out_dir / "metrics_v2.csv",
                 model_name=cfg["model"]["name"])
    write_v2_markdown(
        result, out_dir / "validation.md", model_name=cfg["model"]["name"],
        header={"mode": "train", "smoke": False,
                "n_val_windows": result["n_windows"]},
    )

    hashes = {
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "config_sha256": config_fingerprint(cfg),
        "dataset_sha256": fingerprint_file(Path(h5_path)),
        "split_sha256": fingerprint_file(Path(cfg["data"]["split_path"])),
        "normalization_sha256": fingerprint_file(
            Path(cfg["data"]["normalization_path"])),
        "checkpoint_sha256": fingerprint_file(Path(checkpoint_path)),
    }
    write_manifest(
        out_dir, model_name=cfg["model"]["name"],
        config_path=Path(row["config_path"]), cfg=cfg, result=result,
        device=device, seed=int(row["seed"]),
        batch_size=EXPECTED_BATCH_SIZE, epochs=EXPECTED_EPOCHS,
        use_amp=use_amp, n_params=n_params, mode="train",
        aliases=row["aliases_on_manifest"], checkpoint_path=checkpoint_path,
        smoke=False, best_epoch=best_epoch, best_val_mse=best_val_mse,
        runtime_s=total_runtime_s, hashes=hashes,
    )
    return {
        "run_id": row["run_id"],
        "status": "EXECUTED",
        "out_dir": str(out_dir),
        "runtime_seconds": total_runtime_s,
        "best_epoch": best_epoch,
        "best_val_mse": best_val_mse,
    }


# ---------------------------------------------------------------------------
# Plan writers (dry-run / verify-only output)
# ---------------------------------------------------------------------------

def write_plan_artifacts(
    rows: List[Dict[str, Any]],
    statuses: List[Dict[str, Any]],
    output_root: Path,
) -> Dict[str, Path]:
    """Write the 28-row execution plan as JSON, CSV, and Markdown.

    The Markdown is the file consumed by
    ``deliverables/MULTISEED_EXECUTION_PLAN.md`` (Phase 1 deliverable).
    """
    plan_dir = output_root / "_plan"
    plan_dir.mkdir(parents=True, exist_ok=True)

    # JSON — full machine-readable matrix
    plan_json = plan_dir / "matrix.json"
    payload = {
        "seeds": sorted({r["seed"] for r in rows}),
        "experiments": sorted({r["experiment_alias"] for r in rows}),
        "n_rows": len(rows),
        "frozen_configs_sha256": {
            "configs/experiment_aliases_v2.yaml":
                fingerprint_file(ALIAS_REGISTRY_PATH),
            "configs/splits_v1.yaml":
                fingerprint_file(Path(SPLIT_PATH)),
            "configs/normalization_v1.json":
                fingerprint_file(Path(NORM_PATH)),
            "configs/evaluation_thresholds_v1.json": (
                fingerprint_file(Path(REPO_ROOT /
                                     "configs/evaluation_thresholds_v1.json"))
                if (REPO_ROOT / "configs/evaluation_thresholds_v1.json").exists()
                else None
            ),
        },
        "rows": [
            {**row, "compatibility": statuses[i]}
            for i, row in enumerate(rows)
        ],
    }
    plan_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # CSV — 28-row table
    plan_csv = plan_dir / "matrix.csv"
    with plan_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "RUN_ID", "SEED", "EXPERIMENT", "CANONICAL_STEM",
            "CONFIG_PATH", "CHANNELS", "LOSS", "OUTPUT_DIR",
            "STATUS", "REASON",
        ])
        for row, status in zip(rows, statuses):
            w.writerow([
                row["run_id"], row["seed"], row["experiment_alias"],
                row["canonical_stem"], row["config_path"],
                "[" + ",".join(str(c) for c in row["channels"]) + "]",
                "[" + ",".join(row["loss_components"]) + "]",
                str(output_root / f"seed_{row['seed']}"
                    / row["experiment_alias"]),
                status["status"], status["reason"],
            ])

    # Markdown — human-readable
    plan_md = plan_dir / "matrix.md"
    lines: List[str] = []
    lines.append("# Multi-Seed CORE Execution Plan")
    lines.append("")
    lines.append(f"- Seeds: `{sorted({r['seed'] for r in rows})}`")
    lines.append(
        f"- Experiments: `{sorted({r['experiment_alias'] for r in rows})}`"
    )
    lines.append(f"- Rows: **{len(rows)}**")
    n_complete = sum(1 for s in statuses if s["status"] == "COMPLETE")
    n_incomplete = sum(1 for s in statuses if s["status"] == "INCOMPLETE")
    n_missing = sum(1 for s in statuses if s["status"] == "MISSING")
    lines.append(
        f"- Status: COMPLETE={n_complete}, "
        f"INCOMPLETE={n_incomplete}, MISSING={n_missing}"
    )
    lines.append("")
    lines.append("## Frozen config SHA256")
    lines.append("")
    for k, v in payload["frozen_configs_sha256"].items():
        if v:
            lines.append(f"- `{k}`: `{v}`")
    lines.append("")
    lines.append("## 28-row matrix")
    lines.append("")
    lines.append(
        "| RUN_ID | SEED | EXPERIMENT | CHANNELS | LOSS | STATUS |"
    )
    lines.append("|---|---|---|---|---|---|")
    for row, status in zip(rows, statuses):
        lines.append(
            f"| `{row['run_id']}` | {row['seed']} | "
            f"{row['experiment_alias']} ({row['canonical_stem']}) | "
            f"`{row['channels']}` | `{row['loss_components']}` | "
            f"{status['status']} |"
        )
    plan_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "json": plan_json,
        "csv": plan_csv,
        "markdown": plan_md,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_csv_ints(s: str) -> Tuple[int, ...]:
    return tuple(int(x) for x in s.split(",") if x.strip())


def parse_csv_strs(s: str) -> Tuple[str, ...]:
    return tuple(x.strip() for x in s.split(",") if x.strip())


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "CORE multi-seed runner for Design-C "
            "(docs/MULTISEED_PROTOCOL_FREEZE.md §4.1). "
            "Phase 1 default is --dry-run (no GPU, no training)."
        ),
    )
    parser.add_argument(
        "--seeds", type=parse_csv_ints, default=DEFAULT_SEEDS,
        help=(
            "Comma-separated seeds to run "
            f"(default: {','.join(str(s) for s in DEFAULT_SEEDS)})."
        ),
    )
    parser.add_argument(
        "--experiments", type=parse_csv_strs, default=DEFAULT_EXPERIMENTS,
        help=(
            "Comma-separated Axis aliases "
            f"(default: {','.join(DEFAULT_EXPERIMENTS)})."
        ),
    )
    parser.add_argument(
        "--output-root", default="outputs/multiseed",
        help="Output root for per-seed dirs (default: outputs/multiseed).",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run", action="store_true", default=True,
        help=(
            "Enumerate the run matrix, write plan JSON/CSV/Markdown, "
            "and exit. No GPU, no training. (DEFAULT.)"
        ),
    )
    mode.add_argument(
        "--verify-only", action="store_true",
        help=(
            "Same as --dry-run but ALSO call check_compatibility on every "
            "existing per-seed dir. No GPU, no training."
        ),
    )
    mode.add_argument(
        "--execute", action="store_true",
        help=(
            "Actually train + validate each row. NOT FOR PHASE 1 — requires "
            "explicit human sign-off on deliverables/MULTISEED_EXECUTION_PLAN.md."
        ),
    )
    parser.add_argument(
        "--num-workers", type=int, default=None,
        help="DataLoader workers (used by --execute only).",
    )
    args = parser.parse_args()

    # Hard test-seal guard at every entry point.
    assert_test_seal()

    if SEED_CANONICAL in args.seeds:
        print(
            "[guard] NOTE: canonical seed 42 is in --seeds; Phase 2 will "
            "skip those rows because seed-42 artifacts already exist.",
            file=sys.stderr,
        )

    registry = load_alias_registry()
    rows = build_run_matrix(args.seeds, args.experiments, registry)

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    # ---- compatibility scan ----
    statuses: List[Dict[str, Any]] = []
    for row in rows:
        out_dir = output_root / f"seed_{row['seed']}" / row["experiment_alias"]
        status = check_compatibility(out_dir, row)
        statuses.append(status)

    # ---- write plan artifacts ----
    plan_paths = write_plan_artifacts(rows, statuses, output_root)
    print(f"[plan] wrote {plan_paths['json']}")
    print(f"[plan] wrote {plan_paths['csv']}")
    print(f"[plan] wrote {plan_paths['markdown']}")

    # ---- stdout summary ----
    print("")
    print("=" * 70)
    print("Multi-Seed CORE Execution Plan (Phase 1 — Dry Run)")
    print("=" * 70)
    n_complete = sum(1 for s in statuses if s["status"] == "COMPLETE")
    n_incomplete = sum(1 for s in statuses if s["status"] == "INCOMPLETE")
    n_missing = sum(1 for s in statuses if s["status"] == "MISSING")
    print(
        f"Rows: {len(rows)}  |  "
        f"COMPLETE: {n_complete}  |  "
        f"INCOMPLETE: {n_incomplete}  |  "
        f"MISSING: {n_missing}"
    )
    print("")
    print("Per-row status:")
    for row, status in zip(rows, statuses):
        marker = {
            "COMPLETE": "✓", "MISSING": "·", "INCOMPLETE": "!"
        }[status["status"]]
        line = (
            f"  {marker} {row['run_id']:<24s} "
            f"seed={row['seed']:<6d}  "
            f"stem={row['canonical_stem']:<32s}  "
            f"status={status['status']}"
        )
        print(line)
        if status["status"] == "INCOMPLETE":
            for f in status["failed_checks"]:
                print(f"        ! {f}")

    if args.execute:
        # Phase 2 path — explicitly gated by the explicit flag.
        if any(s["status"] == "INCOMPLETE" for s in statuses):
            print(
                "[execute] refusing to start: INCOMPLETE runs detected. "
                "Resolve them first or pass --force (NOT YET IMPLEMENTED).",
                file=sys.stderr,
            )
            return 2
        print("")
        print("=" * 70)
        print("Multi-Seed CORE Execution (Phase 2)")
        print("=" * 70)
        for row, status in zip(rows, statuses):
            if status["status"] == "COMPLETE":
                print(f"  · SKIP {row['run_id']} (compatible complete)")
                continue
            print(f"  → RUN  {row['run_id']}")
            run_one_row(row, output_root, args.num_workers)
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())