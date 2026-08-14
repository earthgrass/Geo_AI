# PI-ResConvLSTM: Physics-informed Residual ConvLSTM for Typhoon Precipitation Nowcasting

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)

**PI-ResConvLSTM** is a deep-learning framework for spatiotemporal typhoon precipitation nowcasting. It integrates a residual ConvLSTM encoder with an (opt-in) physics-informed loss.

Built on **First Prize** work at the **Beijing Intercollegiate Mathematical Modeling Competition (2026)**, and refactored for reproducibility and scientific rigor.

---

## Project Status

This repository distinguishes **what is implemented** from **what has been experimentally validated**. Be explicit about this split in any paper or report you derive from this code.

### ✅ Implemented (code exists, runs structurally)

- Model code: `PIResConvLSTM`, `ResConvLSTM`, `PlainConvLSTM`, `PersistenceBaseline` (`src/models/`)
- Meteorological metric suite: MAE, RMSE, SSIM, CSI, POD, FAR, HSS, peak/area/center error (`src/evaluation/metrics.py`)
- Leakage-safe dataset loader: year-based and event-based splitting with fail-fast metadata validation (`src/data/dataset.py`, `src/data/splits.py`)
- 12-channel paper dataset builder (`scripts/build_paper_dataset.py`) + validator (`scripts/validate_paper_dataset.py`)
- Experiment framework: trainer, config-driven physics loss, YAML configs, seed control (`src/training/`)

### ⏳ Pending validation (NOT yet established)

- Benchmark results (baseline comparisons have **not** been run against the 12-channel dataset)
- Whether terrain-aware physics guidance improves nowcasting
- Whether physics guidance helps heavy rainfall or terrain-forced precipitation
- Final ablation results

> **Do not** cite any numbers below as validated paper results — they are legacy competition artifacts.

---

## Research Question

> Can terrain-aware physics guidance improve ConvLSTM-based typhoon precipitation nowcasting, and are the benefits particularly pronounced for heavy rainfall and terrain-forced precipitation?

This is an **open hypothesis** — the experiments have not yet been run. The hypothesis is kept separate from any conclusion.

---

## Model Architecture

```
Input: [B, K=11, C=12, H=128, W=128]
        │
        ▼
  ┌──────────────────────────┐
  │  ConvLSTM Encoder #1     │  hidden=64
  │  ConvLSTM Encoder #2     │  hidden=128
  │  [Channel Attention]     │  optional SE Block
  └────────────┬─────────────┘
               │  h_deep [B, 128, H, W]
               ▼
  Decoder → pred_head  →  p_base
  RefineNet(h_deep, p_base) → ΔP
               │
               ▼
     P_hat = ReLU(P_last + ΔP)
```

Temporal residual learning: the model predicts the precipitation **change** ΔP, and the caller computes `P_hat = ReLU(P_last + ΔP)`.

---

## Loss Function

```
L_total = L_rain + λ_smooth·L_smooth + λ_extreme·L_extreme  (+ λ_oro·L_oro, opt-in)

L_rain:    standard (unweighted) MSE over the full precipitation field
L_smooth:  weak spatial + temporal smoothness
L_extreme: MSE restricted to extreme pixels (P_true > threshold)
L_oro:     orographic uplift consistency — OPT-IN, requires explicitly
           configured environmental wind channels
```

- `L_nonneg` was **removed**: the output is already `ReLU(...) ≥ 0`, so a non-negativity penalty is identically zero.
- The orographic term is **disabled by default**. `u_move`/`v_move` are storm *translation* velocities, not atmospheric wind, and must never be used as the wind terms of an orographic constraint.

---

## Data Sources

| Data | Source | Resolution | Description |
|------|--------|------------|-------------|
| CMA Best Track | China Meteorological Admin. | 6h → 0.5h (interp.) | Typhoon track & intensity |
| GPM IMERG | NASA | 0.1° (~10km), 30min | Satellite precipitation |
| ETOPO1 DEM | NOAA | 1 arc-min (~1.8km) | Terrain elevation |

The 12-channel paper schema is defined canonically in `src/config.py` (`CHANNEL_NAMES`).

---

## Legacy Competition Results (NOT paper evidence)

The following numbers come from the **legacy competition pipeline** (hard-coded physics post-processing, random 80/20 split, no ground truth for the target typhoons). They are recorded here for provenance only and **must not be used as validated paper results**.

| Finding | Caveat |
|---------|--------|
| "Terrain ablation: P_max drops 26.4%" | Legacy hard-coded `×0.35` orographic factor; also inconsistent with the -20.91% figure in the essay |
| "V-INTENSE S_ext +19.2%" | Legacy scenario post-processing, no ground truth |
| "Spearman ρ > 0.6 (wind ↔ extreme)" | Competition Q1 analysis, not part of the nowcasting pipeline |

---

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Build the dataset (12 channels, leakage-safe)

```bash
python scripts/build_paper_dataset.py \
    --cma-dir CMABSTdata --tif-dir TIFdata --dem Global_DEM.tif \
    --out ConvLSTM_Dataset_128.h5
python scripts/validate_paper_dataset.py --h5 ConvLSTM_Dataset_128.h5
```

### Training (not yet run for the paper)

```bash
python -m src.training.trainer --config configs/default.yaml
```

### Evaluation

```python
from src.evaluation.metrics import compute_all_metrics

metrics = compute_all_metrics(P_pred, P_true)
print(metrics["MAE"], metrics["CSI_10mmh"])
```

---

## Reference

1. Shi et al. (2015). Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting. *NeurIPS*.
2. Hu et al. (2018). Squeeze-and-Excitation Networks. *CVPR*.
3. Raissi et al. (2019). Physics-informed neural networks. *Journal of Computational Physics*.

## License

This project is for academic research purposes. See `docs/PROJECT_DOC.md` for full documentation.
