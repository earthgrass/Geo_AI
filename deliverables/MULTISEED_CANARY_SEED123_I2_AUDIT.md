# Multi-Seed Canary Audit — seed123 / I2 (Phase 2A)

> **Status: TEMPLATE — pending the official canary run on the Linux GPU host.**
> Generated 2026-08-19. This file will be completed from the real
> `outputs/multiseed/seed_123/I2/` artifacts after the human executes
> `scripts/gpu_host_phase2a_preflight.sh` (gate) then
> `python scripts/run_multiseed_core.py --execute --seeds 123 --experiments I2`.
>
> All "expected" values below were pre-verified on this working tree on
> 2026-08-19 (except the training-time values, which can only come from
> the GPU host).

---

## 1. Run identity

| Item | Value |
|---|---|
| CANARY_RUN | `seed123 / I2` (`ms_seed123_I2` → `E2_resconvlstm`) |
| Execution host | Linux GPU host (AutoDL, seed-42-type env) |
| Mode | `train` (per MULTISEED_PROTOCOL §4.1, per-seed I2 is trained fresh) |
| Frozen config | `configs/experiments/E2_resconvlstm.yaml` (seed overridden to 123 only) |
| Test seal | `TEST_STATUS = SEALED`, `FINAL_TEST_STATUS = NOT_AUTHORIZED` |

## 2. Provenance gate (pre-training, on the GPU host)

| Check | Expected | Observed | Status |
|---|---|---|---|
| normalization sha256 (LF) | `92a553f6bd6fd3770e346afb590759fc7ddf5a5c245fef800c5005df6b47cd8e` | | |
| dataset sha256 | `bb83be4616f1f3a9399f98107bbc7d7c6cd4fc5bdaf33f27fb847703241c02ea` | | |
| split sha256 | `e46cb948ecaf303910882b26a770e3ee15765e62fcfb995a003d48696d7f4a9e` | | |
| pytest | PASS | | |
| CUDA available | YES | | |
| test seal markers | SEALED / NOT_AUTHORIZED | | |
| **PHASE2A_PREFLIGHT** | **PASS** | | |

## 3. Artifact audit (post-training)

### 3.1 Manifest contract

| Field | Required | Observed | Status |
|---|---|---|---|
| experiment | `E2_resconvlstm` | | |
| seed | `123` | | |
| protocol_id | `evaluation_v2` | | |
| test_status | `SEALED` | | |
| smoke | `false` | | |
| batch_size | `4` | | |
| epochs | `20` | | |
| selection_metric | `rain_mse` | | |
| input_channel_indices | `[0]` | | |
| loss_components | `["rain"]` | | |
| dataset_sha256 | `bb83be…` | | |
| split_sha256 | `e46cb948…` | | |
| normalization_sha256 | `92a553…` | | |
| config_sha256 | `cbf9a515feffe3995da30c6748bd5b441d2293828ec8ab4cb1a97329102c7968` (seed=123) | | |
| checkpoint_sha256 | recorded (differs from seed-42 — expected) | | |

### 3.2 result_v2 contract

| Field | Required | Observed | Status |
|---|---|---|---|
| protocol_id | `evaluation_v2` | | |
| split | `val` | | |
| test_status | `SEALED` | | |
| n_events | `7` | | |
| n_windows | `1266` | | |

### 3.3 Files produced

| Artifact | Required | Status |
|---|---|---|
| `manifest.json` | YES | |
| `result_v2.json` | YES | |
| `metrics_v2.csv` | YES | |
| `validation.md` | YES | |
| `models/ResConvLSTM_best.pth` | YES | |
| `logs/` (training history) | YES | |

## 4. Config equivalence — seed42 I2 vs seed123 I2

Equal fields (must all match): `experiment, model, batch_size, epochs,
learning_rate, weight_decay, checkpoint_selection_metric, loss_components,
protocol_id, input_channel_indices, dataset_sha256, split_sha256,
normalization_sha256`.

Allowed to differ: `seed` (42 → 123), `checkpoint_sha256` (re-trained),
`runtime_seconds` (fresh train), `mode` (validate-only → train).

| Field | seed42 | seed123 | Equal |
|---|---|---|---|
| seed | 42 | 123 | N/A (the one allowed change) |
| experiment | E2_resconvlstm | | |
| model | ResConvLSTM | | |
| batch_size | 4 | | |
| epochs | 20 | | |
| selection_metric | rain_mse | | |
| loss_components | [rain] | | |
| protocol_id | evaluation_v2 | | |
| dataset_sha256 | bb83be… | | |
| split_sha256 | e46cb948… | | |
| normalization_sha256 | 92a553… | | |

## 5. Decision

```text
CANARY_RUN = seed123/I2
TRAINING_STATUS =
ARTIFACT_STATUS =

SEED42_VS_SEED123_CONFIG_EQUAL_EXCEPT_SEED = YES/NO

DATASET_SHA_MATCH =
SPLIT_SHA_MATCH =
NORMALIZATION_SHA_MATCH =

SEED123_NORMALIZATION_SHA =

PROTOCOL_ID =
VALIDATION_SPLIT =
N_EVENTS =
N_WINDOWS =

TEST_STATUS =
FINAL_TEST_STATUS =

CHECKPOINT_CREATED =
RESULT_V2_CREATED =

SCIENTIFIC_SEMANTICS_CHANGED = NO

CANARY_ACCEPTANCE = PASS/FAIL
```

> **No scientific-result filtering applies.** The canary gate evaluates
> engineering/protocol/artifact/fingerprint correctness only — never the
> MAE/RMSE/SSIM quality. Any metric value is a legal result.
