"""Autoregressive inference engine for PI-ResConvLSTM.

Key differences from competition version (step2_3):
    1. No hard-coded *1.5 scaling factor
    2. Model output treated as ΔP (consistent with training)
    3. P_hat = ReLU(P_t + ΔP) — pure model prediction
    4. No hand-tuned physics baseline (physics is in the loss, not inference)
    5. Autoregressive feedback with error propagation tracking
"""

import torch
import numpy as np
from typing import Optional, Dict, List, Tuple

from ..models.pi_res_convlstm import PIResConvLSTM


class InferenceEngine:
    """Autoregressive inference for typhoon precipitation prediction.

    At each timestep:
        1. Feed K input frames → model predicts ΔP
        2. Compute P_hat = ReLU(P_t + ΔP)
        3. Slide window: insert P_hat as new precipitation frame
        4. Update non-precipitation channels from future features

    Args:
        model: Trained PI-ResConvLSTM model.
        device: Torch device.
        seq_len: Number of input frames.
        precip_channel_idx: Index of precipitation channel.
    """

    def __init__(
        self,
        model: PIResConvLSTM,
        device: torch.device = None,
        seq_len: int = 11,
        precip_channel_idx: int = 0,
    ):
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.model = model.to(self.device)
        self.model.eval()
        self.seq_len = seq_len
        self.precip_channel_idx = precip_channel_idx

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        model_kwargs: Dict,
        device: torch.device = None,
        seq_len: int = 11,
    ) -> "InferenceEngine":
        """Load model from saved checkpoint.

        Args:
            checkpoint_path: Path to .pth checkpoint file.
            model_kwargs: Dict of kwargs for PIResConvLSTM constructor.
            device: Torch device.
            seq_len: Number of input frames.

        Returns:
            InferenceEngine with loaded model weights.
        """
        if device is None:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'
            )

        model = PIResConvLSTM(**model_kwargs)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        print(f"[Inference] Loaded checkpoint from {checkpoint_path}")
        print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"  Best Val Loss: {checkpoint.get('best_val_loss', 'unknown'):.6f}"
              if isinstance(checkpoint.get('best_val_loss'), float) else "")

        return cls(model, device, seq_len)

    def run_autoregressive(
        self,
        initial_sequence: np.ndarray,
        future_channels: np.ndarray,
        progress: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run autoregressive prediction for multiple future timesteps.

        Args:
            initial_sequence: [K, C, H, W] initial input frames.
                Must match the model's input channel count.
            future_channels: [T, C-1, H, W] future non-precipitation
                channel values. Precipitation channel is omitted since
                it will be predicted autoregressively.
            progress: Show progress bar.

        Returns:
            predictions: [T, H, W] predicted precipitation fields.
            deltas: [T, H, W] predicted ΔP at each step.
        """
        K = self.seq_len
        T = future_channels.shape[0]
        C = initial_sequence.shape[1]

        # Rolling buffer: maintain last K frames
        buffer = torch.tensor(initial_sequence, dtype=torch.float32)
        predictions = []
        deltas = []

        iterator = range(T)
        if progress:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="Inference")

        for t in iterator:
            # Last K frames as model input
            x = buffer[-K:].unsqueeze(0).to(self.device)

            with torch.no_grad():
                delta_p = self.model(x)  # [1, 1, H, W]

            # Compute absolute prediction: P_hat = ReLU(P_t + ΔP)
            P_last = buffer[-1, self.precip_channel_idx]  # [H, W]
            P_hat = torch.relu(
                P_last + delta_p[0, 0].cpu()
            )

            predictions.append(P_hat.numpy())
            deltas.append(delta_p[0, 0].cpu().numpy())

            # Update buffer: build new frame
            new_frame = torch.tensor(
                future_channels[t], dtype=torch.float32
            )
            # Insert predicted precipitation at channel index
            # (shift channels to make room)
            new_frame_full = torch.zeros(C, *new_frame.shape[-2:])
            new_frame_full[self.precip_channel_idx] = P_hat
            # Fill non-precip channels from future_channels
            fc_idx = 0
            for c in range(C):
                if c != self.precip_channel_idx and fc_idx < future_channels.shape[1]:
                    new_frame_full[c] = new_frame[fc_idx]
                    fc_idx += 1

            buffer = torch.cat([buffer, new_frame_full.unsqueeze(0)], dim=0)

        return np.array(predictions), np.array(deltas)

    def run_single_step(
        self,
        input_sequence: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict a single future precipitation field.

        Args:
            input_sequence: [K, C, H, W] input frames.

        Returns:
            P_hat: [H, W] predicted absolute precipitation.
            delta_p: [H, W] predicted change.
        """
        x = torch.tensor(
            input_sequence, dtype=torch.float32
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            delta_p = self.model(x)

        P_last = torch.tensor(
            input_sequence[-1, self.precip_channel_idx]
        )
        P_hat = torch.relu(P_last + delta_p[0, 0].cpu())

        return P_hat.numpy(), delta_p[0, 0].cpu().numpy()

    def compute_error_trajectory(
        self,
        initial_sequence: np.ndarray,
        future_channels: np.ndarray,
        ground_truth: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Compute error metrics along the autoregressive trajectory.

        Args:
            initial_sequence: [K, C, H, W] initial frames.
            future_channels: [T, C-1, H, W] future non-precip channels.
            ground_truth: [T, H, W] true precipitation for comparison.

        Returns:
            Dict of metric_name -> [T] arrays showing error growth.
        """
        predictions, _ = self.run_autoregressive(
            initial_sequence, future_channels, progress=False
        )

        T = predictions.shape[0]
        rmse = np.zeros(T)
        mae = np.zeros(T)

        for t in range(T):
            diff = predictions[t] - ground_truth[t]
            rmse[t] = np.sqrt(np.mean(diff ** 2))
            mae[t] = np.mean(np.abs(diff))

        return {
            'rmse': rmse,
            'mae': mae,
        }
