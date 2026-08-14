"""Run one frozen experiment (train + validation ONLY; test is sealed by default).

Reads the event-level split (configs/splits_v1.yaml), the frozen normalization
(configs/normalization_v1.json), and one experiment config (configs/experiments/).

Test-set evaluation is REFUSED unless ``--allow-test-eval`` is explicitly passed
(default False). This enforces the highest-priority research rule: no peeking at
test performance during model development / selection.

Usage:
    python scripts/run_experiment.py --config configs/experiments/05_resconvlstm_terrain.yaml \
        --epochs 100 --out outputs/exp/E5
"""

from __future__ import annotations

import argparse
import json
import os
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
from src.evaluation.evaluator import evaluate_model
from torch.utils.data import DataLoader


def load_split(path="configs/splits_v1.yaml"):
    cfg = yaml.safe_load(open(path, encoding="utf-8"))
    return cfg["train"], cfg["val"], cfg["test"]


def make_model(model_name: str, channel_indices, hidden_dims=(64, 128)):
    n_ch = len(channel_indices)
    if model_name == "PlainConvLSTM":
        return PlainConvLSTM(hidden_dims=list(hidden_dims))
    if model_name == "ResConvLSTM":
        return ResConvLSTM(input_channels=n_ch, hidden_dims=list(hidden_dims))
    if model_name == "TrajGRU":
        return TrajGRU(input_channels=n_ch, hidden_dims=list(hidden_dims))
    raise ValueError(f"Unknown model: {model_name}")


def make_loaders(h5, ids, channel_indices, stats, batch_size, num_workers):
    transform = ChannelNormalize(stats, channel_indices=channel_indices, precip_vmax=100.0)
    ds = TyphoonDataset(h5, typhoon_ids=ids, channel_indices=channel_indices, transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return ds, loader


def main():
    p = argparse.ArgumentParser(description="Run a frozen validation-stage experiment.")
    p.add_argument("--config", required=True)
    p.add_argument("--h5", default="ConvLSTM_Dataset_128.h5")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--out", default="outputs/experiments")
    p.add_argument("--allow-test-eval", action="store_true",
                   help="EXPLICITLY unseal test evaluation (research rule: default OFF).")
    p.add_argument("--max-samples", type=int, default=0,
                   help="Smoke-test cap: limit train/val samples to this count (0 = full).")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    model_cfg = cfg["model"]
    train_cfg = cfg.get("training", {})
    phy_cfg = cfg.get("physics_loss", {})
    model_name = model_cfg["name"]
    channel_indices = model_cfg["input_channel_indices"]
    seed = train_cfg.get("seed", 42)

    train_ids, val_ids, test_ids = load_split()
    stats = json.load(open("configs/normalization_v1.json", encoding="utf-8"))
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[run] model={model_name} channels={channel_indices} device={device} seed={seed}")

    # ---- E0 Persistence: no training, validation eval only ----
    if model_name == "Persistence":
        _, val_loader = make_loaders(args.h5, val_ids, channel_indices, stats,
                                     args.batch_size, args.num_workers)
        model = PersistenceBaseline(0)
        res = evaluate_model(model, val_loader, device, channel_indices=channel_indices)
        _write_results(args.out, model_name, res, device, "Persistence (no training)")
        _seal_check(args, test_ids, channel_indices, stats)
        return

    # ---- Training experiments ----
    train_ds, train_loader = make_loaders(args.h5, train_ids, channel_indices, stats,
                                          args.batch_size, args.num_workers)
    val_ds, val_loader = make_loaders(args.h5, val_ids, channel_indices, stats,
                                      args.batch_size, args.num_workers)
    if args.max_samples > 0:
        train_ds.indices = train_ds.indices[:args.max_samples]
        val_ds.indices = val_ds.indices[:args.max_samples]
        print(f"[smoke] capped train={len(train_ds.indices)} val={len(val_ds.indices)}")

    model = make_model(model_name, channel_indices)
    n_params = sum(x.numel() for x in model.parameters())
    print(f"[run] params={n_params:,}")

    components = phy_cfg.get("components", ["rain"])
    trainer_config = {
        "model_name": model_name,
        "use_physics_loss": True,
        "normalize_precip": False,  # ChannelNormalize already normalizes; loss stays in mm/h-space of normalized P
        "precip_vmax": 100.0,
        "physics_loss": {
            "lambda_extreme": phy_cfg.get("lambda_extreme", 0.5),
            "extreme_threshold": phy_cfg.get("extreme_threshold", 10.0) / 100.0,
            "components": components,
        },
        "learning_rate": train_cfg.get("learning_rate", 1e-4),
        "weight_decay": train_cfg.get("weight_decay", 1e-4),
        "lr_patience": train_cfg.get("lr_patience", 10),
        "early_stopping_patience": train_cfg.get("early_stopping_patience", 10),
        "grad_clip_norm": train_cfg.get("grad_clip_norm", 1.0),
        "use_amp": False,  # CPU-safe
        "checkpoint_dir": os.path.join(args.out, "models"),
        "log_dir": os.path.join(args.out, "logs"),
    }

    t0 = time.time()
    trainer = Trainer(model, train_loader, val_loader, trainer_config)
    trainer.train(epochs=args.epochs)
    runtime_s = time.time() - t0

    res = evaluate_model(model, val_loader, device, channel_indices=channel_indices)
    _write_results(args.out, model_name, res, device, "trained",
                   runtime_s=runtime_s, n_params=n_params,
                   best_val_loss=trainer.best_val_loss)

    _seal_check(args, test_ids, channel_indices, stats)


def _seal_check(args, test_ids, channel_indices, stats):
    if args.allow_test_eval:
        _, test_loader = make_loaders(args.h5, test_ids, channel_indices, stats,
                                      args.batch_size, args.num_workers)
        # NOTE: test evaluation is intentionally NOT wired to model selection;
        # it is only printed when the researcher explicitly unseals it.
        print("[test] TEST SET EVALUATION ALLOWED BY EXPLICIT FLAG (not used for model selection).")
    else:
        print("[test] TEST SET SEALED — final test evaluation refused (use --allow-test-eval explicitly).")


def _write_results(out_dir, model_name, res, device, mode, **extra):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    overall = res["overall"]
    event = res["per_event"]
    rows = []
    rows.append("## Experiment: " + model_name)
    rows.append(f"- mode: {mode}")
    rows.append(f"- device: {device}")
    rows.append(f"- n_val_windows: {len(res['per_window'])}")
    for k, v in extra.items():
        rows.append(f"- {k}: {v}")
    rows.append("")
    rows.append("### Overall validation metrics")
    for k, v in sorted(overall.items()):
        rows.append(f"- {k}: {v:.5f}")
    rows.append("")
    rows.append("### Per-event validation metrics (typhoon_id: MAE/RMSE/CSI@10/CSI@20)")
    for tid in sorted(event.keys()):
        m = event[tid]
        rows.append(
            f"- {tid}: MAE={m.get('MAE', float('nan')):.4f} RMSE={m.get('RMSE', float('nan')):.4f} "
            f"CSI_10={m.get('CSI_10.0mmh', float('nan')):.4f} CSI_20={m.get('CSI_20.0mmh', float('nan')):.4f}"
        )
    (out / f"{model_name}_validation.md").write_text("\n".join(rows), encoding="utf-8")
    print(f"[run] wrote {out / (model_name + '_validation.md')}")


if __name__ == "__main__":
    main()
