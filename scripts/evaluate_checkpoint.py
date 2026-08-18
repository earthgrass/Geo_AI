"""Re-evaluate an existing trained checkpoint with evaluator v2 (no retraining).

Usage:
    python scripts/evaluate_checkpoint.py \
        --config configs/experiments/E2_resconvlstm.yaml \
        --checkpoint saved_models/E2_resconvlstm_seed42/E2_resconvlstm_seed42_best.pth \
        --split val --out results/I2_resconvlstm_seed42_eval_v2

Splits:
    --split train | val   allowed (evaluation only; never trains).
    --split test          REFUSED while the test seal is active.

The test split (events 2306/2310/2402/2418) has no evaluation path through
this tool. It reads the frozen dataset, applies the configured channel
semantics, runs evaluator v2, and writes structured v2 artifacts + a manifest.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for p in (REPO_ROOT, str(REPO_ROOT)):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from scripts.run_experiment import (
    make_loader,
    make_model,
    resolve_use_amp,
    load_split,
    _load_checkpoint_weights,
    write_manifest,
    _git_commit,
    _git_dirty,
)
from src.evaluation.evaluator import evaluate_model_v2
from src.evaluation.reporting import write_v2_json, write_v2_csv, write_v2_markdown
from src.experiments.registry import (
    aliases_for_stem,
    config_fingerprint,
    fingerprint_file,
    load_alias_registry,
    validate_formal_config,
)

ALLOWED_SPLITS = ("train", "val")
SEALED_TEST_IDS = "2306,2310,2402,2418"


def check_split_allowed(split: str) -> None:
    """Refuse any non-train/val split (the sealed test events) while sealed."""
    if split not in ALLOWED_SPLITS:
        raise SystemExit(
            f"--split {split!r} is REFUSED. Test evaluation is SEALED "
            f"(events {SEALED_TEST_IDS}). Only train/val may be evaluated "
            "through this tool."
        )


def main():
    p = argparse.ArgumentParser(description="Evaluate a checkpoint with v2.")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="val",
                   help="One of train|val. 'test' is refused while sealed.")
    p.add_argument("--out", required=True)
    p.add_argument("--num-workers", type=int, default=None, help="Infra only.")
    args = p.parse_args()

    check_split_allowed(args.split)

    config_path = Path(args.config)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_formal_config(cfg, str(config_path))

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    channel_indices = list(model_cfg["input_channel_indices"])
    model_name = model_cfg["name"]
    precip_vmax = float(data_cfg["precip_vmax"])

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    num_workers = (args.num_workers if args.num_workers is not None
                   else (4 if cuda else 0))
    use_amp = resolve_use_amp(train_cfg.get("use_amp"), cuda)

    train_ids, val_ids, _test_ids = load_split(data_cfg["split_path"])
    ids = train_ids if args.split == "train" else val_ids
    stats = json.load(open(data_cfg["normalization_path"], encoding="utf-8"))
    registry = load_alias_registry()
    aliases = aliases_for_stem(config_path.stem, registry)

    ds, loader = make_loader(
        data_cfg["h5_path"], ids, channel_indices, stats,
        batch_size=train_cfg["batch_size"], num_workers=num_workers,
        shuffle=False, pin_memory=cuda)

    model = make_model(
        model_name, channel_indices, list(model_cfg["hidden_dims"]),
        model_cfg["kernel_size"])
    is_module = isinstance(model, torch.nn.Module)
    n_params = sum(x.numel() for x in model.parameters()) if is_module else 0
    if is_module:
        model = model.to(device)

    ckpt = None
    if model_name != "Persistence":
        if not Path(args.checkpoint).exists():
            raise SystemExit(
                f"checkpoint not found: {args.checkpoint}")
        ckpt = _load_checkpoint_weights(args.checkpoint, model, device)
        print(f"[eval] loaded {args.checkpoint}")

    t0 = time.time()
    result = evaluate_model_v2(
        model, loader, device, precip_vmax=precip_vmax,
        thresholds=[5.0, 10.0, 20.0, 30.0],
        channel_indices=channel_indices,
        split=args.split, test_status="SEALED",
    )
    runtime_s = time.time() - t0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_v2_json(result, out_dir / "result_v2.json", model_name=model_name)
    write_v2_csv(result, out_dir / "metrics_v2.csv", model_name=model_name)
    write_v2_markdown(
        result, out_dir / "evaluation.md", model_name=model_name,
        header={"mode": "checkpoint-eval", "split": args.split,
                "n_windows": result["n_windows"]})

    hashes = {
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "config_sha256": config_fingerprint(cfg),
        "dataset_sha256": fingerprint_file(Path(data_cfg["h5_path"])),
        "split_sha256": fingerprint_file(Path(data_cfg["split_path"])),
        "normalization_sha256": fingerprint_file(
            Path(data_cfg["normalization_path"])),
        "checkpoint_sha256": (
            fingerprint_file(Path(args.checkpoint))
            if Path(args.checkpoint).exists() else None),
    }
    write_manifest(
        out_dir, model_name=model_name, config_path=config_path, cfg=cfg,
        result=result, device=device, seed=train_cfg["seed"],
        batch_size=train_cfg["batch_size"], epochs=train_cfg["epochs"],
        use_amp=use_amp, n_params=n_params,
        mode=f"checkpoint-eval[{args.split}]", aliases=aliases,
        checkpoint_path=args.checkpoint, smoke=False,
        best_epoch=(ckpt or {}).get("best_epoch"),
        best_val_mse=(ckpt or {}).get("best_val_mse",
                                      (ckpt or {}).get("best_val_loss")),
        runtime_s=runtime_s, hashes=hashes)

    og = result["overall_global"]
    print(f"[eval] {config_path.stem} ({args.split}) "
          f"MAE_global={og['MAE_global']:.6f} "
          f"RMSE_global={og['RMSE_global']:.6f} "
          f"windows={result['n_windows']} events={result['n_events']}")
    print(f"[eval] wrote {out_dir / 'result_v2.json'} "
          f"(protocol_id={result['protocol_id']}, test_status=SEALED)")


if __name__ == "__main__":
    main()
