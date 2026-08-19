# GeoAI — Tropical-Cyclone Precipitation Nowcasting via Two-Axis Controlled Ablation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)
[![Validation Test Status](https://img.shields.io/badge/pytest-145%2F145%20passed-brightgreen.svg)](tests/)
[![Test Status](https://img.shields.io/badge/TEST--STATUS-SEALED-red.svg)](docs/PRE_FINAL_TEST_FREEZE.md)

A research codebase for **30-minute tropical-cyclone precipitation nowcasting**
built on leakage-safe streaming data, an evaluator-v2 protocol, and a
**two-axis controlled ablation** design. Physics-informed inductive biases
are an **optional extension**, not the paper's spine.

---

## What this repository is

A reproducibility-first implementation of `Research Design C` — a frozen,
pre-registered validation matrix that isolates the marginal value of storm-
state information, static terrain, terrain geometry, smoothness regularization,
and extreme-rain emphasis. The contribution is **controlled empirical
evidence**, not a novel architecture. Architectural choices are *the fixed
backbone* (Residual ConvLSTM with `[64, 128]` hidden dims); scientific
content lives in the ablations.

> The current paper framing is **not** "Physics-Informed Residual ConvLSTM
> outperforms baselines." It is "Does terrain information help 30-min TC
> nowcasting? Does heavy-rain emphasis help retention of high CSI? — measured
> under a sealed test, with paired event-level bootstrap 95% CIs."

## What the two axes are

**Axis I — Input information ablation** (frozen single backbone):

```
I0  Persistence                 [precip only]   ─── lower bound
I1  PlainConvLSTM                [precip only]   ─── plain recurrent baseline
I2  ResConvLSTM                  [precip only]   ─── residual-backbone control  (reuses E2 checkpoint)
I3  ResConvLSTM + storm-state    [+ 7 channels]  ─── adds CMA storm-state / motion
I4  ResConvLSTM + static terrain [+ 2 channels]  ─── adds DEM + land mask
I5  ResConvLSTM + terrain grad   [+ 2 channels]  ─── adds dh/dx + dh/dy  (= P0)
```

**Axis II — Loss / inductive-bias ablation** (same backbone, all 12 channels):

```
P0  MSE only                     (= I5, exact artifact identity)
P1  MSE + Smooth
P2  MSE + Extreme (alias of legacy E6)
P3  MSE + Smooth + Extreme
```

> Conjectural `P4` (orographic prior) and `P5` (full q(V·∇h) stack) are
> **BLOCKED_BY_ENVIRONMENTAL_WIND_DATA** and intentionally have no runnable
> configuration in the freeze. They are not part of the current paper.

See [`docs/RESEARCH_DESIGN_C_FREEZE.md`](docs/RESEARCH_DESIGN_C_FREEZE.md) for
the canonical alias map, frozen controls, and disallowed claims.

## What is sealed

| Path                | Status   | Why                                                   |
|---------------------|----------|-------------------------------------------------------|
| `outputs/`          | n/a      | Local-only throwaway; never tracked.                  |
| `saved_models/*.pth`| never    | Large blob; SHA256 stored inside each `manifest.json`.|
| Test split          | **SEALED** | Single, controlled, post-mandate `final-test` auth — see [`docs/FINAL_TEST_AUTHORIZATION.md`](docs/FINAL_TEST_AUTHORIZATION.md). |
| Editor's quick-pick | **SEALED** | `evaluate_models.py` fail-fast; `--split test` refused. |

Test-set pre-registration, leakage-safe splits, train-only normalization, and
validation-only checkpoint selection are documented in:

- [`docs/PRE_FINAL_TEST_FREEZE.md`](docs/PRE_FINAL_TEST_FREEZE.md)
- [`docs/RESEARCH_DESIGN_C_FREEZE.md`](docs/RESEARCH_DESIGN_C_FREEZE.md)
- [`docs/EVALUATION_PROTOCOL_V2.md`](docs/EVALUATION_PROTOCOL_V2.md)
- [`docs/IMPLEMENTATION_STATUS.md`](docs/IMPLEMENTATION_STATUS.md) (RESOLVED risk table)
- [`docs/LEAKAGE_AND_TEST_LIMIT_AUDIT.md`](docs/LEAKAGE_AND_TEST_LIMIT_AUDIT.md)
- [`docs/FINAL_TEST_AUTHORIZATION.md`](docs/FINAL_TEST_AUTHORIZATION.md)

## Reproducing this study (minimum steps)

```bash
# 1. Environment
pip install -r requirements.txt

# 2. Validation-stage test suite (≥145/145 expected)
pytest -q tests/

# 3. Run a single formal experiment  (e.g. Axis II P3)
python scripts/run_experiment.py --mode train \
  --config configs/experiments/P3_resconvlstm_smooth_extreme.yaml

# 4. Cross-experiment event-level analysis  (per protocol §17)
python scripts/analyze_ablation_results.py \
  --results-dir results --output-dir tables/ablation_analysis

# 5. Promote GPU outputs to the paper-grade results/ tree
python scripts/archive_validation_results.py \
  --results-root results --source-dir outputs/backbone_gate \
  --source-dir outputs/axis_i --source-dir outputs/axis_ii_c1

# 6. Re-evaluate the legacy E2/I2 checkpoint with evaluator v2 (validation only)
python scripts/evaluate_checkpoint.py \
  --config  configs/experiments/E2_resconvlstm.yaml \
  --checkpoint saved_models/E2_resconvlstm_seed42/E2_resconvlstm_seed42_best.pth \
  --split val --out results/I2_resconvlstm_seed42_v2
```

GPU is required for steps 3 and 5 (validation set has 7 events / 1,266 windows).

## What is *not* claimed here

- This is **not** a "new model" paper. ConvLSTM / Residual ConvLSTM are the
  baseline backbone; SOTA claims at `n_event = 4` have no statistical
  standing.
- The auxiliary losses in Axis II are explicitly **not** a "physics prior":
  smoothness is spatial/temporal regularization; extreme-MSE is task-driven
  rare-event emphasis. A real orographic constraint requires time- and
  grid-aligned environmental wind/moisture, which is out of scope.
- All auxiliary numbers in the legacy competition essay remain legacy artifacts.

## Repository layout (paper assets)

| Path | Purpose |
|---|---|
| `src/data/`        | Streaming dataset, channel-subset semantics, transforms |
| `src/models/`      | PlainConvLSTM / ResConvLSTM / PI-ResConvLSTM / TrajGRU |
| `src/training/`    | Trainer with base-rain-MSE selection |
| `src/evaluation/`  | Evaluator v2 (`aggregator_v2` + `paired_event_differences`) |
| `scripts/`         | `run_experiment.py`, `evaluate_checkpoint.py`, `analyze_ablation_results.py`, `archive_validation_results.py`, `verify_experiment_artifact.py` |
| `configs/experiments/` | Frozen canonical configs (E0–E6, B1, P1, P3) |
| `configs/experiment_aliases_v2.yaml` | Single-source-of-truth alias registry |
| `results/`         | Paper-grade experiment assets (manifest, result_v2.json, validation.md, history.json, config.yaml; **never** *.pth) |
| `tables/ablation_analysis/` | Auto-generated ablation analysis (long CSV + per-event-diffs + Holm + Markdown) |
| `deliverables/PI_ResConvLSTM_Paper_Package/` | Submission-ready code/docs/tables/paper for the GPU host rerun |

## Data

| Data | Source | Resolution | Role |
|------|--------|-----------|------|
| GPM IMERG | NASA | 0.1° (~10 km), 30 min | Precipitation target |
| CMA Best Track | China Meteorological Admin. | 6 h → 0.5 h interp. | Storm track + intensity |
| ETOPO1 DEM | NOAA | 1 arc-min (~1.8 km) → reprojected to GPM grid | Terrain channels |

Splits: train 25 events / 4,894 windows (2014–2021) · val 7 events / 1,266
windows (2022) · **test 4 events / 707 windows (2023–2024)** — sealed.

## Status snapshot

| Item | Status |
|---|---|
| Design freeze | `776d2c7` |
| Evaluator v2 / runner / trainer | `ea50b08` |
| TrajGRU device-portability test | `1391b2d` |
| Research-analysis infra (this branch) | `research-analysis-infra` (Draft PR; not merged) |
| **VALIDATION MATRIX** | **COMPLETE (single seed = 42)** — 10 canonical experiments, 11 source artifacts (I2 deduped), see `deliverables/REAL_GPU_RESULT_AUDIT.md` |
| **MULTI-SEED** | **NOT YET CONFIRMED** — 5-seed plan in audit §N; gate for any inferential paper claim |
| GPU training gate | **OPEN** (validation closed; multi-seed pending) |
| Test status | **SEALED** — `FINAL_TEST_STATUS = NOT_AUTHORIZED`; no test split evaluation has occurred |
| Validation test suite | 170 / 170 passing (145 legacy + 25 real-GPU integration) |
| Real-GPU artifact integration | `PASS` — 10 canonical from 11 sources; I2 dedup verified; I5 ≡ P0 verified |
| 5090 paper-experiments rerun | completed (`scripts/run_paper_experiments.py`); results archived to `results/` |

## License

Academic research. See `LICENSE` once added at camera-ready; the current
research-use terms are documented in [`docs/PROJECT_DOC.md`](docs/PROJECT_DOC.md).

## Citation

A pre-print draft will accompany the camera-ready submission. Until then,
cite the underlying methodology references:

1. Shi et al. (2015). *Convolutional LSTM Network: A Machine Learning
   Approach for Precipitation Nowcasting.* NeurIPS.
2. Hu et al. (2018). *Squeeze-and-Excitation Networks.* CVPR.
3. Shi et al. (2017). *Deep Learning for Precipitation Nowcasting: A
   Benchmark and a Model for the World.* ICLR.
4. Raissi et al. (2019). *Physics-informed neural networks.* J. Comput. Phys.
