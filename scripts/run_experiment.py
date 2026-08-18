"""Run one frozen Design-C experiment (train or validate-only; test SEALED).

Reads the event-level split (configs/splits_v1.yaml), the frozen train-only
normalization (configs/normalization_v1.json), and one experiment config
(configs/experiments/). Formal mode uses ONLY the YAML values — CLI overrides
of seed/batch/epoch/channels/model-dims/loss-weights/thresholds are prohibited
(docs/MINIMAX_IMPLEMENTATION_SPEC.md §6).

Modes:
  --mode train         Train the model, then evaluate VALIDATION with evaluator v2.
  --mode validate-only Evaluate an existing checkpoint on VALIDATION with v2
                       (requires --checkpoint; never retrains).

Test sealing:
  There is no test path and no allow-test-eval convenience flag. Test IDs are
  never loaded or evaluated while the seal is active.

Usage:
    python scripts/run_experiment.py --config configs/experiments/E3_resconvlstm_cma.yaml
    python scripts/run_experiment.py --config configs/experiments/E2_resconvlstm.yaml \
        --mode validate-only --checkpoint saved_models/E2_resconvlstm_seed42/E2_resconvlstm_seed42_best.pth \
        --out outputs/revalidation_v2/E2_resconvlstm
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
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

from src.data.dataset import TyphoonDataset
from src.data.transforms import ChannelNormalize
from src.models.baselines import PersistenceBaseline, PlainConvLSTM, ResConvLSTM
from src.models.trajgru import TrajGRU
from src.training.trainer import Trainer, set_seed
from src.evaluation.evaluator import evaluate_model_v2
from src.evaluation.reporting import write_v2_json, write_v2_csv, write_v2_markdown
from src.experiments.registry import (
    ALIAS_REGISTRY_PATH,
    aliases_for_stem,
    config_fingerprint,
    fingerprint_file,
    load_alias_registry,
    resolve_alias,
    validate_formal_config,
)
from torch.utils.data import DataLoader

DEFAULT_H5 = "ConvLSTM_Dataset_128.h5"
SPLIT_PATH = "configs/splits_v1.yaml"
NORM_PATH = "configs/normalization_v1.json"


# ---------------------------------------------------------------------------
# Data / model factories
# ---------------------------------------------------------------------------

def load_split(path=SPLIT_PATH):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["train"], cfg["val"], cfg["test"]


def make_model(model_name, channel_indices, hidden_dims, kernel_size):
    n_ch = len(channel_indices)
    if model_name == "Persistence":
        return PersistenceBaseline(0)
    if model_name == "PlainConvLSTM":
        return PlainConvLSTM(hidden_dims=list(hidden_dims), kernel_size=kernel_size)
    if model_name == "ResConvLSTM":
        return ResConvLSTM(
            input_channels=n_ch, hidden_dims=list(hidden_dims),
            kernel_size=kernel_size)
    if model_name == "TrajGRU":
        return TrajGRU(
            input_channels=n_ch, hidden_dims=list(hidden_dims),
            kernel_size=kernel_size)
    raise ValueError(f"Unknown model: {model_name}")


def make_loader(h5, ids, channel_indices, stats, batch_size, num_workers,
                shuffle, pin_memory):
    transform = ChannelNormalize(
        stats, channel_indices=channel_indices, precip_vmax=100.0)
    ds = TyphoonDataset(
        h5, typhoon_ids=ids, channel_indices=channel_indices, transform=transform)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
        drop_last=(shuffle is True),
    )
    return ds, loader


def resolve_use_amp(yaml_use_amp, cuda_available):
    """Formal AMP resolution: ``auto`` (or None) -> on when CUDA, else off."""
    if yaml_use_amp == "auto" or yaml_use_amp is None:
        return bool(cuda_available)
    return bool(yaml_use_amp)


def _git_commit():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            check=True, cwd=REPO_ROOT)
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _git_dirty():
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True,
            check=True, cwd=REPO_ROOT)
        return len(out.stdout.strip()) > 0
    except Exception:
        return None


def _load_checkpoint_weights(checkpoint_path, model, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    return ckpt


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def write_manifest(out_dir, *, model_name, config_path, cfg, result,
                   device, seed, batch_size, epochs, use_amp, n_params,
                   mode, aliases, checkpoint_path, smoke, best_epoch,
                   best_val_mse, runtime_s, hashes):
    manifest = {
        "experiment": config_path.stem,
        "aliases": aliases,
        "mode": mode,
        "model": model_name,
        "seed": seed,
        "batch_size": batch_size,
        "epochs": epochs,
        "device": str(device),
        "amp_resolved": use_amp,
        "n_params": n_params,
        "git_commit": hashes["git_commit"],
        "git_dirty": hashes["git_dirty"],
        "config_path": str(config_path),
        "config_sha256": hashes["config_sha256"],
        "dataset_sha256": hashes["dataset_sha256"],
        "split_sha256": hashes["split_sha256"],
        "normalization_sha256": hashes["normalization_sha256"],
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": hashes.get("checkpoint_sha256"),
        "selection_metric": cfg["training"]["checkpoint_selection_metric"],
        "best_epoch": best_epoch,
        "best_val_mse": best_val_mse,
        "input_channel_indices": cfg["model"]["input_channel_indices"],
        "loss_components": cfg["physics_loss"]["components"],
        "protocol_id": result.get("protocol_id"),
        "test_status": result.get("test_status"),
        "smoke": bool(smoke),
        "runtime_seconds": runtime_s,
    }
    path = Path(out_dir) / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Run a frozen Design-C experiment (train or validate-only; "
                    "test SEALED).")
    p.add_argument("--config", required=True)
    p.add_argument("--mode", choices=["train", "validate-only"], default="train")
    p.add_argument("--checkpoint", default=None,
                   help="Checkpoint path (required for --mode validate-only).")
    p.add_argument("--out", default="outputs/experiments")
    p.add_argument("--num-workers", type=int, default=None, help="Infra only.")
    p.add_argument("--smoke", action="store_true",
                   help="Cap samples; artifacts marked non-formal.")
    args = p.parse_args()

    # ---- config ----
    config_path = Path(args.config)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_formal_config(cfg, str(config_path))

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    phy_cfg = cfg["physics_loss"]
    out_cfg = cfg.get("output", {})

    model_name = model_cfg["name"]
    channel_indices = list(model_cfg["input_channel_indices"])
    hidden_dims = list(model_cfg["hidden_dims"])
    kernel_size = model_cfg["kernel_size"]
    seed = train_cfg["seed"]
    batch_size = train_cfg["batch_size"]
    epochs = train_cfg["epochs"]
    components = list(phy_cfg["components"])
    smoke = args.smoke

    h5_path = data_cfg["h5_path"]
    split_path = data_cfg["split_path"]
    norm_path = data_cfg["normalization_path"]
    precip_vmax = float(data_cfg["precip_vmax"])

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    num_workers = (args.num_workers if args.num_workers is not None
                   else (4 if cuda else 0))
    use_amp = resolve_use_amp(train_cfg.get("use_amp"), cuda)

    # ---- split / normalization / aliases ----
    train_ids, val_ids, _test_ids = load_split(split_path)
    stats = json.load(open(norm_path, encoding="utf-8"))
    registry = load_alias_registry()
    stem = config_path.stem
    aliases = aliases_for_stem(stem, registry)
    set_seed(seed)

    print(f"[run] mode={args.mode} model={model_name} channels={channel_indices} "
          f"device={device} seed={seed} batch={batch_size} epochs={epochs} "
          f"workers={num_workers} amp={use_amp} aliases={aliases or '-'} "
          f"smoke={smoke}")

    # ---- loaders (validation always; train only when training) ----
    val_ds, val_loader = make_loader(
        h5_path, val_ids, channel_indices, stats, batch_size, num_workers,
        shuffle=False, pin_memory=cuda)
    if smoke:
        val_ds.indices = val_ds.indices[:16]

    model = make_model(model_name, channel_indices, hidden_dims, kernel_size)
    is_module = isinstance(model, torch.nn.Module)
    n_params = sum(x.numel() for x in model.parameters()) if is_module else 0
    if is_module:
        model = model.to(device)

    checkpoint_path = args.checkpoint
    best_epoch = None
    best_val_mse = None
    runtime_s = 0.0

    if args.mode == "validate-only":
        if model_name == "Persistence":
            print("[run] Persistence needs no checkpoint (non-trainable).")
        elif not checkpoint_path or not Path(checkpoint_path).exists():
            raise SystemExit(
                "--mode validate-only requires an existing --checkpoint path.")
        else:
            _load_checkpoint_weights(checkpoint_path, model, device)
            ckpt = torch.load(checkpoint_path, map_location=device)
            best_epoch = ckpt.get("best_epoch")
            best_val_mse = ckpt.get("best_val_mse", ckpt.get("best_val_loss"))
            print(f"[run] loaded checkpoint {checkpoint_path} "
                  f"(best_epoch={best_epoch})")
    else:
        # ---- training (never for E2/I2; enforced by the GPU gates) ----
        if model_name == "Persistence":
            print("[run] Persistence is non-trainable; validation-only.")
        else:
            trainer_config = {
                "model_name": model_name,
                "use_physics_loss": True,
                "normalize_precip": False,
                "precip_vmax": precip_vmax,
                "physics_loss": {
                    "lambda_smooth": phy_cfg["lambda_smooth"],
                    "lambda_extreme": phy_cfg["lambda_extreme"],
                    # Training operates in normalized precipitation space, so the
                    # frozen 10 mm/h threshold becomes 0.1.
                    "extreme_threshold": phy_cfg["extreme_threshold"] / precip_vmax,
                    "components": components,
                },
                "learning_rate": train_cfg["learning_rate"],
                "weight_decay": train_cfg["weight_decay"],
                "lr_patience": train_cfg["lr_patience"],
                "early_stopping_patience": train_cfg["early_stopping_patience"],
                "grad_clip_norm": train_cfg["grad_clip_norm"],
                "use_amp": use_amp,
                "checkpoint_selection_metric": train_cfg["checkpoint_selection_metric"],
                "checkpoint_dir": out_cfg.get("checkpoint_dir", "outputs/models"),
                "log_dir": out_cfg.get("log_dir", "outputs/logs"),
            }
            train_ds, train_loader = make_loader(
                h5_path, train_ids, channel_indices, stats, batch_size,
                num_workers, shuffle=True, pin_memory=cuda)
            if smoke:
                train_ds.indices = train_ds.indices[:16]
                print(f"[smoke] capped train={len(train_ds.indices)} "
                      f"val={len(val_ds.indices)}")

            t0 = time.time()
            trainer = Trainer(model, train_loader, val_loader, trainer_config)
            trainer.train(epochs=epochs)
            runtime_s = time.time() - t0
            best_epoch = trainer.best_epoch + 1
            best_val_mse = trainer.best_val_mse
            checkpoint_path = str(
                Path(trainer_config["checkpoint_dir"]) / f"{model_name}_best.pth")
            model = model.to(device)

    # ---- validation evaluation (evaluator v2; NO test path) ----
    t0 = time.time()
    result = evaluate_model_v2(
        model, val_loader, device, precip_vmax=precip_vmax,
        thresholds=[5.0, 10.0, 20.0, 30.0],
        channel_indices=channel_indices,
        split="val", test_status="SEALED",
    )
    runtime_s += time.time() - t0

    # ---- write v2 artifacts ----
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_v2_json(result, out_dir / "result_v2.json", model_name=model_name)
    write_v2_csv(result, out_dir / "metrics_v2.csv", model_name=model_name)
    write_v2_markdown(
        result, out_dir / "validation.md", model_name=model_name,
        header={"mode": args.mode, "smoke": smoke,
                "n_val_windows": result["n_windows"]})

    hashes = {
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "config_sha256": config_fingerprint(cfg),
        "dataset_sha256": fingerprint_file(Path(h5_path)) if not smoke else None,
        "split_sha256": fingerprint_file(Path(split_path)),
        "normalization_sha256": fingerprint_file(Path(norm_path)),
        "checkpoint_sha256": (
            fingerprint_file(Path(checkpoint_path))
            if checkpoint_path and Path(checkpoint_path).exists() else None),
    }
    write_manifest(
        out_dir, model_name=model_name, config_path=config_path, cfg=cfg,
        result=result, device=device, seed=seed, batch_size=batch_size,
        epochs=epochs, use_amp=use_amp, n_params=n_params, mode=args.mode,
        aliases=aliases, checkpoint_path=checkpoint_path, smoke=smoke,
        best_epoch=best_epoch, best_val_mse=best_val_mse, runtime_s=runtime_s,
        hashes=hashes)

    og = result["overall_global"]
    print(f"[run] wrote {out_dir / 'result_v2.json'}, "
          f"{out_dir / 'validation.md'}")
    print(f"[run] v2 MAE_global={og['MAE_global']:.6f} "
          f"RMSE_global={og['RMSE_global']:.6f} "
          f"windows={result['n_windows']} events={result['n_events']}")
    print("[run] TEST SET SEALED — no test loader, inference, or metrics.")


if __name__ == "__main__":
    main()
