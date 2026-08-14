# PI-ResConvLSTM: Physics-informed Residual ConvLSTM for Typhoon Precipitation Simulation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)

**PI-ResConvLSTM** is a deep learning framework for spatiotemporal typhoon precipitation prediction that bridges data-driven and physics-based approaches.

Built upon **First Prize** work at the **Beijing Intercollegiate Mathematical Modeling Competition (2026)**, now refactored for GeoAI paper publication.

## Overview

This project predicts typhoon precipitation fields by integrating:

- **Residual ConvLSTM** for spatiotemporal modeling
- **Physics-informed loss** with terrain uplift, non-negativity, smoothness, and extreme precipitation constraints
- **Multi-source data**: typhoon track, intensity, satellite precipitation, DEM terrain, land-sea mask
- **Climate scenario simulation**: path shift, intensification, slowdown

## Key Features

- ✅ **TRUE temporal residual learning**: Model predicts ΔP (precipitation change), not absolute values
- ✅ **Physics-informed loss**: L_rain + λ1·L_nonneg + λ2·L_oro + λ3·L_smooth + λ4·L_extreme
- ✅ **Channel Attention** (SE Block) for adaptive multi-modal feature recalibration
- ✅ **Temporal data split**: Train 2014-2022 / Val 2023 / Test 2024 — no data leakage
- ✅ **Comprehensive meteorological metrics**: SSIM, CSI, POD, FAR, HSS, peak error, area error
- ✅ **Baseline comparisons**: Persistence, ConvLSTM, ResConvLSTM, ResConvLSTM+DEM, PI-ResConvLSTM
- ✅ **Climate scenario perturbation analysis** (path shift, intensification, slowdown, compound)
- ✅ **Reproducible**: Fixed seeds, YAML configs, checkpoint management

## Project Structure

```
GeoAI/
├── src/                    # Source code (PI-ResConvLSTM framework)
│   ├── models/             # Model definitions: ConvLSTM cell, PI-ResConvLSTM, baselines
│   ├── data/               # Dataset, transforms, data pipeline
│   ├── training/           # Training loop, physics loss, YAML configs
│   ├── inference/          # Autoregressive inference engine
│   ├── evaluation/         # Meteorological metrics
│   └── visualization/      # Cartopy-based precipitation maps
├── configs/                # Experiment YAML configurations
├── docs/                   # Documentation
│   ├── PROJECT_DOC.md      # Full project documentation
│   ├── CODE_CHANGE_PLAN.md # Code change blueprint
│   └── SUMMARY.md          # PI-ResConvLSTM paper blueprint
├── archive/                # Archived competition files
├── outputs/                # Checkpoints, logs, figures, predictions
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Quick Start

### Installation

```bash
# Clone and install dependencies
cd GeoAI
pip install -r requirements.txt
```

### Training

```bash
# Train PI-ResConvLSTM (with physics-informed loss)
python -m src.training.trainer --model PI-ResConvLSTM --epochs 100

# Train baseline (ResConvLSTM, no physics loss)
python -m src.training.trainer --model ResConvLSTM --no_physics --epochs 100

# Run full benchmark (all baselines)
python -m src.training.trainer --benchmark --epochs 100

# Custom config file
python -m src.training.trainer --config configs/default.yaml
```

### Inference

```python
from src.models.pi_res_convlstm import PIResConvLSTM
from src.inference.infer import InferenceEngine

# Load trained model
engine = InferenceEngine.from_checkpoint(
    "outputs/models/PI-ResConvLSTM_best.pth",
    model_kwargs={'input_channels': 4, 'hidden_dims': [64, 128]},
)

# Run autoregressive prediction
predictions, deltas = engine.run_autoregressive(
    initial_sequence,   # [K, C, H, W]
    future_channels,    # [T, C-1, H, W]
)
```

### Evaluation

```python
from src.evaluation.metrics import compute_all_metrics

metrics = compute_all_metrics(P_pred, P_true)
print(f"MAE: {metrics['MAE']:.4f}, RMSE: {metrics['RMSE']:.4f}")
print(f"SSIM: {metrics['SSIM']:.4f}, CSI_10mm: {metrics['CSI_10.0mmh']:.4f}")
```

## Model Architecture

```
Input:  [B, K=11, C=4, H=128, W=128] for the existing HDF5 dataset
          │
          ▼
  ┌──────────────────────────┐
  │  ConvLSTM Encoder #1     │  hidden=64
  │  ConvLSTM Encoder #2     │  hidden=128
  │  [Channel Attention]     │  optional SE Block
  └────────────┬─────────────┘
               │  h_deep [B, 128, H, W]
               ▼
  ┌──────────────────────────┐
  │  Decoder → Pred Head     │  → p_base
  │  RefineNet(h_deep, p_base)│  → ΔP
  └────────────┬─────────────┘
               │  ΔP [B, 1, H, W]
               ▼
     P_hat = ReLU(P_last + ΔP)
```

## Loss Function

```
L_total = L_rain + λ1·L_nonneg + λ2·L_oro + λ3·L_smooth + λ4·L_extreme

L_rain:    Weighted MSE (heavy rain pixels get α× more weight)
L_nonneg:  Penalty for negative precipitation
L_oro:     Terrain uplift consistency (wind-aligned elevation gradient)
L_smooth:  Spatial + temporal smoothness (small weight to preserve peaks)
L_extreme: Extra MSE on extreme precipitation regions (P > 10 mm/h)
```

## Data Sources

| Data | Source | Resolution | Description |
|------|--------|------------|-------------|
| CMA Best Track | China Meteorological Admin. | 6h → 0.5h (interp.) | Typhoon track & intensity |
| GPM IMERG | NASA | 0.1° (~10km), 30min | Satellite precipitation |
| ETOPO1 DEM | NOAA | 1 arc-min (~1.8km) | Terrain elevation |

## Key Results (Competition Version)

| Finding | Method | Key Insight |
|---------|--------|-------------|
| Wind ↔ Extreme Precip | Spearman ρ > 0.6 | Wind is dominant factor |
| Terrain Ablation | NoTopo experiment | P_max drops 26.4% without DEM |
| Climate Scenarios | V-INTENSE (+15%) | S_ext increases 19.2% |
| Path Curvature | SHAP dependence | Threshold effect at turning points |

## Reference

1. Shi et al. (2015). Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting. *NeurIPS*.
2. Hu et al. (2018). Squeeze-and-Excitation Networks. *CVPR*.
3. Willoughby et al. (2006). Parametric representation of the primary hurricane vortex. *Monthly Weather Review*.
4. Lundberg & Lee (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS* (SHAP).

## License

This project is for academic research purposes. See `docs/PROJECT_DOC.md` for full documentation.

---

*Project completed: April 2026 | Refactored: July 2026*
