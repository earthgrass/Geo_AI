"""Evaluate PI-ResConvLSTM and baselines on a year-based split.

This script turns trained checkpoints into paper-ready metric tables.
It intentionally supports the persistence baseline without requiring a
checkpoint, because persistence is the minimum nowcasting benchmark.

Examples:
    python -m src.evaluation.evaluate_models --h5 ConvLSTM_Dataset_128.h5 --baseline persistence
    python -m src.evaluation.evaluate_models --checkpoint outputs/models/PI-ResConvLSTM_best.pth
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.dataset import TyphoonDataset
from ..data.transforms import Compose, MinMaxNormalize
from ..models.baselines import PersistenceBaseline
from ..models.pi_res_convlstm import PIResConvLSTM
from .metrics import compute_all_metrics


def denormalize(array: np.ndarray, vmax: float) -> np.ndarray:
    return array * vmax


def evaluate_persistence(
    loader: DataLoader,
    precip_vmax: float,
    precip_channel_idx: int = 0,
) -> Dict[str, float]:
    model = PersistenceBaseline(precip_channel_idx=precip_channel_idx)
    all_pred, all_true = [], []
    for X, Y, _meta in loader:
        pred = model(X).numpy()[:, 0]
        true = Y.numpy()[:, 0]
        all_pred.append(denormalize(pred, precip_vmax))
        all_true.append(denormalize(true, precip_vmax))
    return compute_all_metrics(np.concatenate(all_pred), np.concatenate(all_true))


@torch.no_grad()
def evaluate_checkpoint(
    loader: DataLoader,
    checkpoint_path: str,
    precip_vmax: float,
    input_channels: int = 4,
    hidden_dims: List[int] | None = None,
    precip_channel_idx: int = 0,
) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PIResConvLSTM(
        input_channels=input_channels,
        hidden_dims=hidden_dims or [64, 128],
        precip_channel_idx=precip_channel_idx,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.eval()

    all_pred, all_true = [], []
    for X, Y, meta in loader:
        X = X.to(device)
        P_prev = meta["P_prev"].to(device)
        delta = model(X)
        pred = torch.relu(P_prev + delta).cpu().numpy()[:, 0]
        true = Y.numpy()[:, 0]
        all_pred.append(denormalize(pred, precip_vmax))
        all_true.append(denormalize(true, precip_vmax))

    return compute_all_metrics(np.concatenate(all_pred), np.concatenate(all_true))


def create_eval_loader(args: argparse.Namespace) -> DataLoader:
    transform = Compose([
        MinMaxNormalize(vmin=0.0, vmax=args.precip_vmax, channel_idx=args.precip_channel_idx)
    ]) if args.normalize else None

    ds = TyphoonDataset(
        args.h5,
        seq_len=args.seq_len,
        precip_channel_idx=args.precip_channel_idx,
        split_years=(args.start_year, args.end_year),
        transform=transform,
    )
    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def write_metrics_csv(metrics: Dict[str, float], output_path: str, model_name: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "metric", "value"])
        for key, value in sorted(metrics.items()):
            writer.writerow([model_name, key, value])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate precipitation nowcasting models")
    parser.add_argument("--h5", default="ConvLSTM_Dataset_128.h5")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--baseline", choices=["persistence"], default=None)
    parser.add_argument("--output", default="outputs/evaluation/metrics.csv")
    parser.add_argument("--seq_len", type=int, default=11)
    parser.add_argument("--input_channels", type=int, default=4)
    parser.add_argument("--precip_channel_idx", type=int, default=0)
    parser.add_argument("--precip_vmax", type=float, default=100.0)
    parser.add_argument("--normalize", dest="normalize", action="store_true", default=True)
    parser.add_argument("--no_normalize", dest="normalize", action="store_false")
    parser.add_argument("--start_year", type=int, default=2024)
    parser.add_argument("--end_year", type=int, default=2024)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loader = create_eval_loader(args)

    if args.baseline == "persistence":
        metrics = evaluate_persistence(loader, args.precip_vmax, args.precip_channel_idx)
        model_name = "Persistence"
    elif args.checkpoint:
        metrics = evaluate_checkpoint(
            loader,
            args.checkpoint,
            args.precip_vmax,
            input_channels=args.input_channels,
            precip_channel_idx=args.precip_channel_idx,
        )
        model_name = Path(args.checkpoint).stem
    else:
        raise SystemExit("Provide --baseline persistence or --checkpoint path.")

    write_metrics_csv(metrics, args.output, model_name)
    print(f"Wrote metrics to {args.output}")


if __name__ == "__main__":
    main()
