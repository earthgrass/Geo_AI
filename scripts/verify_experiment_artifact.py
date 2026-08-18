"""Read-only artifact compatibility verifier for frozen Design-C checkpoints.

Usage:
    python scripts/verify_experiment_artifact.py \\
        --experiment I2 \\
        --checkpoint saved_models/E2_resconvlstm_seed42/E2_resconvlstm_seed42_best.pth \\
        --manifest results/E2_resconvlstm_seed42/manifest.json

Checks (never trains, never copies):
  - alias resolves to the canonical config (I5 and P0 -> E5_terrain_geometry);
  - config passes frozen common-control validation (seed/channels/epochs/...);
  - checkpoint weights strictly load into the config-built model;
  - manifest fields (seed/batch/epochs/channels/model/test_status) match config;
  - checkpoint SHA-256 matches the manifest when the manifest records one;
  - a smoke artifact is rejected.

Exit code 0 = compatible; nonzero = BLOCKED with a printed reason.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for p in (REPO_ROOT, str(REPO_ROOT)):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.experiments.registry import (
    config_fingerprint,
    config_path_for,
    fingerprint_file,
    load_alias_registry,
    resolve_alias,
    validate_formal_config,
)
from scripts.run_experiment import make_model

FROZEN_SEED = 42
FROZEN_BATCH = 4
FROZEN_EPOCHS = 20


def _fmt_value(v):
    if isinstance(v, float) and v != v:  # NaN
        return "nan"
    return v


def main():
    p = argparse.ArgumentParser(description="Verify a Design-C artifact.")
    p.add_argument("--experiment", required=True,
                   help="Axis alias or canonical config stem (e.g. I2, I5, P0).")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", default=None)
    args = p.parse_args()

    registry = load_alias_registry()
    stem = resolve_alias(args.experiment, registry)
    aliases = sorted(a for a, s in registry.get("aliases", {}).items() if s == stem)

    checks: list = []
    ok = True

    def check(name: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        checks.append({"check": name, "ok": bool(passed), "detail": detail})
        if not passed:
            ok = False

    # 1. Alias resolution / identity
    check("alias_resolves", True, f"{args.experiment} -> {stem} (aliases={aliases})")
    if stem == "E5_terrain_geometry":
        i5 = resolve_alias("I5", registry)
        p0 = resolve_alias("P0", registry)
        check("i5_p0_same_identity", i5 == p0 == "E5_terrain_geometry",
              f"I5 -> {i5}, P0 -> {p0}")

    # 2. Config exists + frozen validation + fingerprint
    config_path = config_path_for(stem)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    try:
        validate_formal_config(cfg, str(config_path))
        check("config_frozen_controls", True,
              f"config_sha256={config_fingerprint(cfg)}")
    except ValueError as exc:
        check("config_frozen_controls", False, str(exc))

    # 3. Checkpoint exists
    ckpt_path = Path(args.checkpoint)
    check("checkpoint_exists", ckpt_path.exists(), str(ckpt_path))
    if not ckpt_path.exists():
        _finish(checks, config_path=config_path, stem=stem)
        return 1

    # 4. Strict state-dict load into the config-built model
    model_name = cfg["model"]["name"]
    if model_name == "Persistence":
        check("persistence_has_no_checkpoint", False,
              "Persistence is non-trainable; a checkpoint is not expected.")
        _finish(checks, config_path=config_path, stem=stem)
        return 1
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        model = make_model(
            model_name,
            list(cfg["model"]["input_channel_indices"]),
            list(cfg["model"]["hidden_dims"]),
            cfg["model"]["kernel_size"],
        )
        model.load_state_dict(state, strict=True)
        check("state_dict_loads", True,
              f"model={model_name} "
              f"params={sum(x.numel() for x in model.parameters()):,}")
    except Exception as exc:  # noqa: BLE001
        check("state_dict_loads", False, f"{type(exc).__name__}: {exc}")

    # 5. Manifest cross-check (when supplied)
    if args.manifest:
        manifest_path = Path(args.manifest)
        check("manifest_exists", manifest_path.exists(), str(manifest_path))
        if manifest_path.exists():
            man = json.loads(manifest_path.read_text(encoding="utf-8"))
            if man.get("smoke"):
                check("not_smoke_artifact", False,
                      "manifest declares smoke=True (non-formal).")
            else:
                check("not_smoke_artifact", True)
            fields = {
                "model": (man.get("model"), model_name),
                "input_channel_indices": (
                    man.get("input_channel_indices"),
                    list(cfg["model"]["input_channel_indices"])),
                "seed": (man.get("seed"), FROZEN_SEED),
                "batch_size": (
                    man.get("batch_size") or man.get("max_batch"),
                    FROZEN_BATCH),
            }
            man_epochs = man.get("max_epochs", man.get("epochs"))
            fields["epochs"] = (man_epochs, FROZEN_EPOCHS)
            for name, (got, expected) in fields.items():
                check(f"manifest.{name}", got == expected,
                      f"{got!r} == {expected!r}")
            ts = man.get("test_status")
            check("manifest.test_status", str(ts).upper() == "SEALED",
                  f"test_status={ts}")
            man_ckpt_hash = man.get("checkpoint_sha256")
            if man_ckpt_hash:
                actual = fingerprint_file(ckpt_path)
                check("checkpoint_hash_matches", actual == man_ckpt_hash,
                      f"{actual[:12]}... == {man_ckpt_hash[:12]}...")
    else:
        check("manifest_crosscheck_skipped", True,
              "no --manifest supplied; config + state-dict checks only")

    _finish(checks, config_path=config_path, stem=stem)
    return 0 if ok else 1


def _finish(checks, config_path, stem) -> None:
    report = {
        "experiment": stem,
        "compatible": all(c["ok"] for c in checks),
        "checks": checks,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("COMPATIBLE" if report["compatible"] else "INCOMPATIBLE")


if __name__ == "__main__":
    sys.exit(main())
