"""Training orchestrator for PI-ResConvLSTM and baseline models.

Supports:
    - Temporal (year-based) split — no data leakage
    - Multiple baselines with shared infrastructure
    - Physics-informed loss with component-level ablation
    - Automatic Mixed Precision (AMP)
    - Early stopping, LR scheduling, checkpoint management
    - TensorBoard logging
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import os
import json
from typing import Dict, Optional, Tuple, Any

from ..data.dataset import TyphoonDataset
from ..data.transforms import Compose, MinMaxNormalize
from ..models.pi_res_convlstm import PIResConvLSTM
from ..models.baselines import PlainConvLSTM, ResConvLSTM
from ..models.trajgru import TrajGRU
from .physics_loss import PhysicsInformedLoss


# ---------------------------------------------------------------------------
# Helper: set all random seeds
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def create_dataloaders(
    h5_path: str,
    seq_len: int = 11,
    precip_channel_idx: int = 0,
    train_years: Tuple[int, int] = (2014, 2022),
    val_years: Tuple[int, int] = (2023, 2023),
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True,
    normalize: bool = True,
    precip_max: float = 100.0,
) -> Tuple[DataLoader, DataLoader]:
    """Create train/val dataloaders with temporal split.

    Returns:
        train_loader, val_loader, test_loader (None if test_years=None)
    """
    transforms = None
    if normalize:
        transforms = Compose([
            MinMaxNormalize(vmin=0.0, vmax=precip_max, channel_idx=precip_channel_idx),
        ])

    train_ds = TyphoonDataset(
        h5_path, seq_len=seq_len,
        precip_channel_idx=precip_channel_idx,
        split_years=train_years,
        transform=transforms,
    )
    val_ds = TyphoonDataset(
        h5_path, seq_len=seq_len,
        precip_channel_idx=precip_channel_idx,
        split_years=val_years,
        transform=transforms,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def create_model(name: str, **kwargs) -> nn.Module:
    """Create model by name.

    Args:
        name: One of "PI-ResConvLSTM", "ResConvLSTM", "PlainConvLSTM".
        **kwargs: Passed to model constructor.

    Returns:
        Model instance.
    """
    if name == "PI-ResConvLSTM":
        return PIResConvLSTM(**kwargs)
    elif name == "ResConvLSTM":
        allowed = {
            'input_channels', 'precip_channel_idx', 'hidden_dims', 'kernel_size'
        }
        return ResConvLSTM(**{k: v for k, v in kwargs.items() if k in allowed})
    elif name == "PlainConvLSTM":
        allowed = {'hidden_dims', 'kernel_size'}
        return PlainConvLSTM(**{k: v for k, v in kwargs.items() if k in allowed})
    elif name == "TrajGRU":
        allowed = {'input_channels', 'hidden_dims', 'kernel_size'}
        return TrajGRU(**{k: v for k, v in kwargs.items() if k in allowed})
    else:
        raise ValueError(f"Unknown model: {name}. "
                         f"Choose from: PI-ResConvLSTM, ResConvLSTM, "
                         f"PlainConvLSTM, TrajGRU")


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Unified trainer for PI-ResConvLSTM and baselines.

    Args:
        model: Model instance.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        config: Training configuration dict.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.model = self.model.to(self.device)

        # Loss function
        use_physics = config.get('use_physics_loss', True)
        self.oro_config = {'enabled': False}
        if use_physics:
            phy_cfg = config.get('physics_loss', {})
            extreme_threshold = phy_cfg.get('extreme_threshold', 10.0)
            if config.get('normalize_precip', False):
                extreme_threshold = extreme_threshold / max(
                    float(config.get('precip_vmax', 100.0)), 1e-8
                )
            self.oro_config = dict(phy_cfg.get('orographic', {'enabled': False}))
            self.criterion = PhysicsInformedLoss(
                lambda_smooth=phy_cfg.get('lambda_smooth', 0.01),
                lambda_extreme=phy_cfg.get('lambda_extreme', 0.5),
                lambda_oro=phy_cfg.get('lambda_oro', 0.1),
                extreme_threshold=extreme_threshold,
                orographic_corr_weight=phy_cfg.get('orographic_corr_weight', True),
                components=phy_cfg.get('components', None),
                oro_config=self.oro_config,
            )
        else:
            self.criterion = nn.MSELoss()
        self.criterion = self.criterion.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 1e-4),
        )

        # Scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5,
            patience=config.get('lr_patience', 10),
        )

        # AMP (only on CUDA; the scaler is device-specific).
        self.use_amp = bool(config.get('use_amp', True)) and torch.cuda.is_available()
        self.scaler = GradScaler('cuda', enabled=self.use_amp)

        # Early stopping
        self.patience = config.get('early_stopping_patience', 20)
        self.grad_clip = config.get('grad_clip_norm', 1.0)

        # Logging
        self.checkpoint_dir = config.get('checkpoint_dir', 'outputs/models')
        self.log_dir = config.get('log_dir', 'outputs/logs')
        self.model_name = config.get('model_name', 'model')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # Frozen checkpoint-selection rule: minimum validation base (rain) MSE.
        # Auxiliary-loss totals are logged but NEVER select the epoch.
        self.selection_metric = config.get(
            'checkpoint_selection_metric', 'rain_mse')
        if self.selection_metric not in ('rain_mse', 'val_base_mse'):
            raise ValueError(
                f"checkpoint_selection_metric must be 'rain_mse' (base MSE), "
                f"got {self.selection_metric!r}."
            )
        self.best_val_mse = float('inf')   # selection signal (base rain MSE)
        self.best_val_total = float('inf')  # composite val total (logged only)
        self.best_val_loss = self.best_val_mse  # alias = selection value
        self.best_epoch = -1                 # 0-indexed best epoch
        self.patience_counter = 0
        self.current_epoch = 0

        # Summary
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[Trainer] Model: {self.model_name} | "
              f"Params: {n_params:,} | Device: {self.device}")
        print(f"[Trainer] Train: {len(train_loader.dataset)} samples | "
              f"Val: {len(val_loader.dataset)} samples")

    def _predict_precipitation(
        self,
        X: torch.Tensor,
        P_prev: torch.Tensor,
    ) -> torch.Tensor:
        """Return absolute precipitation prediction for every model family.

        Frozen prediction contract (docs/RESEARCH_DESIGN_C_FREEZE.md §13):
        - PlainConvLSTM and TrajGRU are ABSOLUTE-output models: their forward
          already applies a ReLU precipitation head, so the output IS the
          precipitation field. Adding ``P_prev`` again would double-count.
        - ResConvLSTM / PI-ResConvLSTM output a residual delta; the absolute
          prediction is ``ReLU(P_prev + delta)``.

        Both branches therefore return the SAME absolute tensor the evaluator
        consumes (evaluator.predict_absolute), so train/validation/evaluation
        semantics never diverge.
        """
        if isinstance(self.model, (PlainConvLSTM, TrajGRU)):
            # Precipitation-only absolute-output models.
            return self.model(X[:, :, 0:1, :, :])

        delta_p = self.model(X)
        return torch.relu(P_prev + delta_p)

    def _build_physics_aux(
        self,
        X: torch.Tensor,
        P_prev: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Build auxiliary physics tensors consumed by PhysicsInformedLoss.

        The orographic uplift field O = u·dh/dx + v·dh/dy is only computed when
        the orographic term is explicitly enabled AND its wind channels are
        configured. Channel indices come from configuration, never hard-coded.
        """
        aux = {'P_prev': P_prev}

        oro_cfg = self.oro_config
        if not oro_cfg.get('enabled', False):
            return aux

        u_ch = oro_cfg.get('u_channel')
        v_ch = oro_cfg.get('v_channel')
        dhx_ch = oro_cfg.get('dh_dx_channel', 9)
        dhy_ch = oro_cfg.get('dh_dy_channel', 10)

        if u_ch is None or v_ch is None:
            raise RuntimeError(
                "Orographic term is enabled but u_channel/v_channel are not "
                "configured. Environmental wind channels are required."
            )

        n_channels = X.shape[2]
        for name, c in (("u_channel", u_ch), ("v_channel", v_ch),
                        ("dh_dx_channel", dhx_ch), ("dh_dy_channel", dhy_ch)):
            if c >= n_channels:
                raise RuntimeError(
                    f"orographic {name}={c} exceeds input channel count {n_channels}."
                )

        last = X[:, -1, :, :, :]
        u = last[:, u_ch:u_ch + 1, :, :]
        v = last[:, v_ch:v_ch + 1, :, :]
        dh_dx = last[:, dhx_ch:dhx_ch + 1, :, :]
        dh_dy = last[:, dhy_ch:dhy_ch + 1, :, :]
        aux['oro_lift'] = u * dh_dx + v * dh_dy

        return aux

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train_epoch(self) -> Dict[str, float]:
        """Run one training epoch.

        Returns:
            Dict of average loss components.
        """
        self.model.train()
        total_loss = 0.0
        loss_components = {}
        n_samples = 0

        for X, Y, meta in self.train_loader:
            X = X.to(self.device, non_blocking=self.device.type == 'cuda')
            Y = Y.to(self.device, non_blocking=self.device.type == 'cuda')
            P_prev = meta['P_prev'].to(self.device, non_blocking=self.device.type == 'cuda')

            self.optimizer.zero_grad()

            with autocast("cuda", enabled=self.use_amp):
                P_hat = self._predict_precipitation(X, P_prev)

                # Compute loss
                if isinstance(self.criterion, PhysicsInformedLoss):
                    aux = self._build_physics_aux(X, P_prev)
                    losses = self.criterion(P_hat, Y, aux)
                    loss = losses['total']
                else:
                    loss = self.criterion(P_hat, Y)
                    losses = {'total': loss, 'rain': loss}

            # Backward
            self.scaler.scale(loss).backward()

            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Accumulate
            batch_size = X.size(0)
            n_samples += batch_size
            total_loss += loss.item() * batch_size
            for k, v in losses.items():
                if isinstance(v, torch.Tensor):
                    loss_components[k] = (
                        loss_components.get(k, 0.0) + v.item() * batch_size
                    )
                else:
                    loss_components[k] = loss_components.get(k, 0.0) + v * batch_size

        return {k: v / n_samples for k, v in loss_components.items()}

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run one validation epoch.

        Returns:
            Dict of average loss components.
        """
        self.model.eval()
        total_loss = 0.0
        loss_components = {}
        n_samples = 0

        for X, Y, meta in self.val_loader:
            X = X.to(self.device, non_blocking=self.device.type == 'cuda')
            Y = Y.to(self.device, non_blocking=self.device.type == 'cuda')
            P_prev = meta['P_prev'].to(self.device, non_blocking=self.device.type == 'cuda')

            with autocast("cuda", enabled=self.use_amp):
                P_hat = self._predict_precipitation(X, P_prev)

                if isinstance(self.criterion, PhysicsInformedLoss):
                    aux = self._build_physics_aux(X, P_prev)
                    losses = self.criterion(P_hat, Y, aux)
                else:
                    loss = self.criterion(P_hat, Y)
                    losses = {'total': loss, 'rain': loss}

            batch_size = X.size(0)
            n_samples += batch_size
            total_loss += losses['total'].item() * batch_size if isinstance(
                losses['total'], torch.Tensor
            ) else losses['total'] * batch_size

            for k, v in losses.items():
                if isinstance(v, torch.Tensor):
                    loss_components[k] = (
                        loss_components.get(k, 0.0) + v.item() * batch_size
                    )
                else:
                    loss_components[k] = loss_components.get(k, 0.0) + v * batch_size

        return {k: v / n_samples for k, v in loss_components.items()}

    # ------------------------------------------------------------------
    # Main training
    # ------------------------------------------------------------------

    def train(self, epochs: int) -> nn.Module:
        """Run full training loop.

        Args:
            epochs: Maximum number of epochs.

        Returns:
            Trained model (with best weights loaded).
        """
        history = {'train': [], 'val': []}
        print(f"\n{'='*55}")
        print(f"Training {self.model_name} for {epochs} epochs")
        print(f"{'='*55}")

        for epoch in range(epochs):
            self.current_epoch = epoch
            t0 = time.time()

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Selection signal: unweighted validation base (rain) MSE.
            val_rain = val_metrics.get(
                'rain', val_metrics.get('loss', float('inf')))
            val_total = val_metrics.get('total', val_rain)

            # LR scheduling tracks the same base-MSE selection signal so the
            # reduction never responds to a composite loss the design freezes
            # out of selection.
            self.scheduler.step(val_rain)

            # Early stopping check (strict <: earliest epoch wins an exact tie).
            is_best = val_rain < self.best_val_mse
            if is_best:
                self.best_val_mse = val_rain
                self.best_val_total = val_total
                self.best_epoch = epoch
                self.patience_counter = 0
                self._save_checkpoint('best')
            else:
                self.patience_counter += 1

            # Periodic checkpoint
            if epoch % 10 == 0:
                self._save_checkpoint(f'epoch_{epoch:03d}')

            # Log
            history['train'].append(train_metrics)
            history['val'].append(val_metrics)

            elapsed = time.time() - t0
            lr = self.optimizer.param_groups[0]['lr']
            print(
                f"Epoch {epoch+1:3d}/{epochs} | "
                f"Train: {train_metrics.get('total', 0):.4f} | "
                f"ValTotal: {val_total:.4f} | "
                f"ValMSE: {val_rain:.6f} | "
                f"BestMSE: {self.best_val_mse:.6f} | "
                f"LR: {lr:.2e} | "
                f"Time: {elapsed:.1f}s"
            )

            if self.patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch+1} "
                      f"(patience={self.patience})")
                break

        # Load best weights
        self._load_best()

        # Save history with explicit selection metadata.
        history['selection_metric'] = self.selection_metric
        history['best_val_mse'] = self.best_val_mse
        history['best_val_total'] = self.best_val_total
        history['best_epoch'] = self.best_epoch + 1  # 1-indexed
        with open(os.path.join(self.log_dir, 'history.json'), 'w') as f:
            json.dump(history, f, indent=2)

        print(
            f"\nTraining complete. Best val base MSE: {self.best_val_mse:.8f} "
            f"(epoch {self.best_epoch + 1}, selection_metric="
            f"{self.selection_metric}, best val total: {self.best_val_total:.8f})"
        )
        return self.model

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, tag: str):
        """Save model checkpoint."""
        path = os.path.join(
            self.checkpoint_dir, f"{self.model_name}_{tag}.pth"
        )
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'selection_metric': self.selection_metric,
            'best_val_mse': self.best_val_mse,
            'best_val_total': self.best_val_total,
            'best_epoch': self.best_epoch + 1,  # 1-indexed
            'config': self.config,
        }, path)

    def _load_best(self):
        """Load best model weights."""
        path = os.path.join(
            self.checkpoint_dir, f"{self.model_name}_best.pth"
        )
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    h5_path: str,
    output_dir: str = "outputs",
    models_to_run: list = None,
    use_physics_loss: bool = True,
    epochs: int = 100,
    batch_size: int = 8,
    seed: int = 42,
    **kwargs,
):
    """Run full benchmark across all baseline models.

    This trains and evaluates:
        1. Persistence (no training needed)
        2. PlainConvLSTM
        3. ResConvLSTM (+ DEM channels)
        4. PI-ResConvLSTM (with physics loss)
        5. PIA-ResConvLSTM (with Channel Attention, optional)

    Args:
        h5_path: Path to HDF5 dataset.
        output_dir: Output directory for checkpoints and logs.
        models_to_run: List of model names to run. None = all.
        use_physics_loss: Enable physics loss for PI-ResConvLSTM.
        epochs: Max epochs per model.
        batch_size: Batch size.
        seed: Random seed.

    Returns:
        Dict mapping model name to validation metrics.
    """
    set_seed(seed)

    if models_to_run is None:
        models_to_run = [
            "PlainConvLSTM",
            "ResConvLSTM",
            "PI-ResConvLSTM",
        ]

    # Shared model kwargs
    model_kwargs = {
        'input_channels': kwargs.get('input_channels', 4),
        'precip_channel_idx': kwargs.get('precip_channel_idx', 0),
        'hidden_dims': kwargs.get('hidden_dims', [64, 128]),
    }

    results = {}

    for model_name in models_to_run:
        print(f"\n{'#'*55}")
        print(f"# Benchmark: {model_name}")
        print(f"{'#'*55}")

        # Create dataloaders
        train_loader, val_loader = create_dataloaders(
            h5_path=h5_path,
            batch_size=batch_size,
            num_workers=kwargs.get('num_workers', 4),
        )

        # Create model
        model = create_model(model_name, **model_kwargs)

        # Config
        config = {
            'model_name': model_name,
            'use_physics_loss': use_physics_loss and model_name == "PI-ResConvLSTM",
            'normalize_precip': kwargs.get('normalize', True),
            'precip_vmax': kwargs.get('precip_vmax', 100.0),
            'physics_loss': {
                'lambda_oro': 0.1,
                'lambda_smooth': 0.01,
                'lambda_extreme': 0.5,
                'extreme_threshold': 10.0,
                'orographic': {
                    'enabled': False,
                    'u_channel': None,
                    'v_channel': None,
                    'dh_dx_channel': 9,
                    'dh_dy_channel': 10,
                },
            },
            'learning_rate': 1e-4,
            'weight_decay': 1e-4,
            'lr_patience': 10,
            'early_stopping_patience': 20,
            'grad_clip_norm': 1.0,
            'use_amp': True,
            'checkpoint_dir': os.path.join(output_dir, 'models'),
            'log_dir': os.path.join(output_dir, 'logs'),
        }

        # Train
        trainer = Trainer(model, train_loader, val_loader, config)
        trainer.train(epochs=epochs)
        results[model_name] = {'best_val_loss': trainer.best_val_loss}

    # Print summary
    print(f"\n{'='*55}")
    print("Benchmark Summary")
    print(f"{'='*55}")
    for name, res in results.items():
        print(f"  {name:25s}: Best Val Loss = {res['best_val_loss']:.6f}")

    return results


def load_yaml_config(path: str) -> Dict[str, Any]:
    """Load a YAML experiment config."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required for --config. Install requirements.txt first."
        ) from exc

    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def train_from_config(config_path: str) -> nn.Module:
    """Create dataloaders, model, and trainer from a YAML config file."""
    cfg = load_yaml_config(config_path)
    data_cfg = cfg.get('data', {})
    model_cfg = cfg.get('model', {})
    train_cfg = cfg.get('training', {})
    phy_cfg = cfg.get('physics_loss', {})
    output_cfg = cfg.get('output', {})

    set_seed(train_cfg.get('seed', 42))

    train_loader, val_loader = create_dataloaders(
        h5_path=data_cfg.get('h5_path', 'ConvLSTM_Dataset_128.h5'),
        seq_len=data_cfg.get('seq_len', 11),
        precip_channel_idx=data_cfg.get('precip_channel_idx', 0),
        train_years=tuple(data_cfg.get('train_years', [2014, 2022])),
        val_years=tuple(data_cfg.get('val_years', [2023, 2023])),
        batch_size=train_cfg.get('batch_size', 8),
        num_workers=data_cfg.get('num_workers', 4),
        normalize=data_cfg.get('normalize', True),
        precip_max=data_cfg.get('precip_vmax', 100.0),
    )

    model_name = model_cfg.get('name', 'PI-ResConvLSTM')
    model_kwargs = {
        'input_channels': model_cfg.get('input_channels', 4),
        'precip_channel_idx': model_cfg.get('precip_channel_idx', 0),
        'hidden_dims': model_cfg.get('hidden_dims', [64, 128]),
        'kernel_size': model_cfg.get('kernel_size', 3),
        'use_attention': model_cfg.get('use_attention', False),
        'attention_reduction': model_cfg.get('attention_reduction', 16),
        'dropout': model_cfg.get('dropout', 0.0),
        'use_layer_norm': model_cfg.get('use_layer_norm', False),
    }
    model = create_model(model_name, **model_kwargs)

    trainer_config = {
        'model_name': model_name,
        'use_physics_loss': phy_cfg.get('use_physics_loss', True),
        'normalize_precip': data_cfg.get('normalize', True),
        'precip_vmax': data_cfg.get('precip_vmax', 100.0),
        'physics_loss': phy_cfg,
        'learning_rate': train_cfg.get('learning_rate', 1e-4),
        'weight_decay': train_cfg.get('weight_decay', 1e-4),
        'lr_patience': train_cfg.get('lr_patience', 10),
        'early_stopping_patience': train_cfg.get('early_stopping_patience', 20),
        'grad_clip_norm': train_cfg.get('grad_clip_norm', 1.0),
        'use_amp': train_cfg.get('use_amp', True),
        'checkpoint_dir': output_cfg.get('checkpoint_dir', 'outputs/models'),
        'log_dir': output_cfg.get('log_dir', 'outputs/logs'),
    }

    trainer = Trainer(model, train_loader, val_loader, trainer_config)
    return trainer.train(epochs=train_cfg.get('epochs', 100))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train PI-ResConvLSTM")
    parser.add_argument('--h5', type=str, default='ConvLSTM_Dataset_128.h5',
                        help='Path to HDF5 dataset')
    parser.add_argument('--model', type=str, default='PI-ResConvLSTM',
                        choices=['PI-ResConvLSTM', 'ResConvLSTM', 'PlainConvLSTM'],
                        help='Model to train')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML experiment config')
    parser.add_argument('--no_physics', action='store_true',
                        help='Disable physics-informed loss')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run all baselines')
    args = parser.parse_args()

    if args.config:
        train_from_config(args.config)
    elif args.benchmark:
        run_benchmark(
            h5_path=args.h5,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
        )
    else:
        set_seed(args.seed)

        train_loader, val_loader = create_dataloaders(
            h5_path=args.h5, batch_size=args.batch_size,
        )

        model = create_model(args.model)

        config = {
            'model_name': args.model,
            'use_physics_loss': not args.no_physics,
            'normalize_precip': True,
            'precip_vmax': 100.0,
            'learning_rate': args.lr,
            'weight_decay': 1e-4,
            'lr_patience': 10,
            'early_stopping_patience': 20,
            'grad_clip_norm': 1.0,
            'use_amp': True,
            'checkpoint_dir': 'outputs/models',
            'log_dir': 'outputs/logs',
        }

        trainer = Trainer(model, train_loader, val_loader, config)
        trainer.train(epochs=args.epochs)
