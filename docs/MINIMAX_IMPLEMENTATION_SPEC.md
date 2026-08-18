# Minimax Implementation Specification — Research Design C

**Purpose:** give the next engineering agent the smallest unambiguous change set required before any further GPU training.  
**Baseline reviewed:** `514d2ec`  
**Authorization:** implementation and validation-only re-evaluation; no dataset/model redesign, no test evaluation, no ERA5, and no training until all pre-GPU gates pass.

## 1. Required order of work

1. Preserve the current dirty worktree and inventory unrelated user files; never stage or alter them.
2. Implement evaluator v2 and its unit tests.
3. Fix runner configuration fidelity, checkpoint selection, report output, and test sealing.
4. Add the frozen aliases/configs and validate the entire matrix without training.
5. Fix E2 artifact resolution so E2 can only be reused or cause a hard stop, never retrained.
6. Run the complete CPU test suite and synthetic smoke evaluation.
7. Re-evaluate the existing E2 checkpoint on **validation only** with evaluator v2.
8. Only after all gates pass, generate/commit the GPU scripts. Execution remains a separate manual GPU step.

## 2. Files to modify

| File | Exact required change |
|---|---|
| `src/evaluation/metrics.py` | Separate continuous sufficient statistics from contingency counts; return integer `a,b,c,d`; implement denominator-safe ratios returning `NaN`; implement fixed-range SSIM; remove legacy per-window-range NRMSE and peak-relative ratio from standard v2 output. |
| `src/evaluation/evaluator.py` | Aggregate squared/absolute errors and categorical counts before computing global/event metrics; retain per-window diagnostics; emit level-qualified structures and `protocol_id`; remove the second canonical-channel slice; assert loader channel count/order instead. |
| `scripts/run_experiment.py` | Honor `data.h5_path`, `seq_len`, `model.hidden_dims`, `model.kernel_size`, every loss weight, output metadata, and AMP `auto`; add train versus validation-only mode; remove the test-evaluation flag/path; select checkpoints by base validation MSE; write v2 JSON/CSV/Markdown without hand-typed metric-key variants. |
| `src/training/trainer.py` | Add a frozen checkpoint-selection field set to base `rain` MSE, while continuing to backpropagate and log `total`; store selected epoch and both selection MSE/composite loss in checkpoints/history. Make TrajGRU use its absolute-output contract in both training and validation instead of adding `P_prev`. Do not change optimizer, model architecture, or loss formulas. |
| `src/evaluation/evaluate_models.py` | Eliminate the year-2024 default/test-seal bypass. Make this a thin validation-only wrapper around the canonical evaluator/runner, or fail with a clear deprecation message directing users to `run_experiment.py --mode validate-only`. It must reject test IDs while sealed. |
| `scripts/run_backbone_gate.sh` | Never fall through to E2 training. Resolve and verify an existing E2 checkpoint, re-evaluate it on validation, or stop. Keep E0 → E1 → E2 reuse → B1 order and no test option. |
| `scripts/gen_experiment_configs.py` | Generate the frozen 20-epoch matrix and P1/P3 configs without overwriting provenance fields; never generate P4/P5. |
| `configs/experiments/E0_persistence.yaml` through `E6_terrain_extreme.yaml`, and `B1_trajgru.yaml` | Change maximum epochs to 20; add explicit experiment identity/aliases, AMP `auto`, selection metric, and explicit loss weights/components. Do not change channel lists or model families. |
| `tests/test_experiment.py` | Replace the legacy mean-event aggregation expectation; add config fidelity, channel-subset, validation-selection, nonnegativity, test-seal, and E2-no-retrain tests. |

## 3. Files to add

| File | Purpose |
|---|---|
| `configs/experiment_aliases_v2.yaml` | Single authoritative E→I/P alias registry. |
| `configs/experiments/P1_resconvlstm_smooth.yaml` | All-12 ResConvLSTM with MSE + smoothness. |
| `configs/experiments/P3_resconvlstm_smooth_extreme.yaml` | All-12 ResConvLSTM with MSE + smoothness + extreme-MSE. |
| `tests/test_evaluation_protocol_v2.py` | Exact toy-array tests for all v2 metric and aggregation rules. |
| `scripts/verify_experiment_artifact.py` | Read-only manifest/checkpoint compatibility verifier used by resume/skip logic. It must never train or copy an artifact. |
| `scripts/run_axis_i.sh` | Run only I3/E3, I4/E4, and I5/E5 after the backbone gate; I2 is validation-only reuse. |
| `scripts/run_axis_ii_c1.sh` | Require the exact I5/P0 artifact, then run P1, P2/E6, and P3; never train a separate P0. |

Do not generate a P0 config duplicate. P0 resolves to E5. Do not generate P4 or P5 configs, scripts, stubs that calculate a proxy, or placeholder result rows.

## 4. Files forbidden to modify

The implementation agent must not modify:

- `ConvLSTM_Dataset_128.h5` or any H5/data product;
- `configs/splits_v1.yaml`;
- `configs/normalization_v1.json`;
- `configs/evaluation_thresholds_v1.json`;
- `src/data/dataset.py`, `src/data/splits.py`, or `src/data/transforms.py`;
- `src/models/baselines.py`, `src/models/convlstm_cell.py`, `src/models/pi_res_convlstm.py`, or `src/models/trajgru.py`;
- `src/training/physics_loss.py` (its Stage-C1 formulas are frozen below);
- test data, test IDs, model architecture, channel order, split, or normalization;
- existing E2 checkpoint/history/manifest content except adding a separately named v2 validation report;
- `docs/literature_review_stage1.md`.

If a required test appears to demand a forbidden-file change, stop and request architecture review rather than expanding scope.

## 5. Frozen alias registry

`configs/experiment_aliases_v2.yaml` must encode exactly:

```yaml
version: "2.0"
aliases:
  I0: E0_persistence
  I1: E1_plain_convlstm
  I2: E2_resconvlstm
  I3: E3_resconvlstm_cma
  I4: E4_static_terrain
  I5: E5_terrain_geometry
  P0: E5_terrain_geometry
  P1: P1_resconvlstm_smooth
  P2: E6_terrain_extreme
  P3: P3_resconvlstm_smooth_extreme
references:
  B1: B1_trajgru
blocked:
  P4: BLOCKED_BY_ENVIRONMENTAL_WIND_DATA
  P5: BLOCKED_BY_ENVIRONMENTAL_WIND_DATA
```

The runner must record both requested alias and canonical legacy ID in the manifest. I5 and P0 must resolve to the same canonical config path and artifact ID.

## 6. Exact experiment configuration

Every YAML must be self-contained and validated against the following common block:

```yaml
data:
  h5_path: "ConvLSTM_Dataset_128.h5"
  seq_len: 11
  split_path: "configs/splits_v1.yaml"
  normalization_path: "configs/normalization_v1.json"
  precip_vmax: 100.0

model:
  hidden_dims: [64, 128]
  kernel_size: 3

training:
  batch_size: 4
  epochs: 20
  learning_rate: 0.0001
  weight_decay: 0.0001
  lr_patience: 10
  early_stopping_patience: 10
  grad_clip_norm: 1.0
  seed: 42
  use_amp: auto
  checkpoint_selection_metric: rain_mse

physics_loss:
  enabled: true
  lambda_smooth: 0.01
  lambda_extreme: 0.5
  extreme_threshold: 10.0
  orographic:
    enabled: false
```

Per-experiment fields are exactly:

| Canonical config | Alias | `model.name` | `input_channel_indices` | `components` | Train? |
|---|---|---|---|---|---|
| E0_persistence | I0 | Persistence | `[0]` | `[rain]` (unused) | No |
| E1_plain_convlstm | I1 | PlainConvLSTM | `[0]` | `[rain]` | Yes |
| E2_resconvlstm | I2 | ResConvLSTM | `[0]` | `[rain]` | **No; reuse only** |
| E3_resconvlstm_cma | I3 | ResConvLSTM | `[0,1,2,3,4,5,6,7]` | `[rain]` | Yes |
| E4_static_terrain | I4 | ResConvLSTM | `[0,1,2,3,4,5,6,7,8,11]` | `[rain]` | Yes |
| E5_terrain_geometry | I5, P0 | ResConvLSTM | `[0,1,2,3,4,5,6,7,8,9,10,11]` | `[rain]` | Yes, once |
| P1_resconvlstm_smooth | P1 | ResConvLSTM | all 12 | `[rain, smooth]` | Yes |
| E6_terrain_extreme | P2 | ResConvLSTM | all 12 | `[rain, extreme]` | Yes |
| P3_resconvlstm_smooth_extreme | P3 | ResConvLSTM | all 12 | `[rain, smooth, extreme]` | Yes |
| B1_trajgru | B1 | TrajGRU | `[0]` | `[rain]` | Yes |

Config validation must reject unknown components, missing fields, duplicate aliases, any P4/P5 executable row, and any deviation from the frozen common controls. CLI overrides of seed, batch, epoch, channel list, model dimensions, loss weights, or thresholds are prohibited for formal mode. A separate `--smoke` mode may cap samples but its artifacts must be marked non-formal and cannot satisfy completion checks.

## 7. Frozen Stage-C1 loss formulas

All tensors are in normalized precipitation units during training, with `P_prev`, `P_hat`, and `P_true` divided by 100 mm/h.

```text
L_rain = mean((P_hat - P_true)^2)

L_smooth = mean(|P_hat[:,:,1:,:] - P_hat[:,:,:-1,:]|)
           + mean(|P_hat[:,:,:,1:] - P_hat[:,:,:,:-1]|)
           + mean(|P_hat - P_prev|)

M = 1[P_true > 0.1]
L_extreme = sum(M * (P_hat - P_true)^2) / sum(M), if sum(M) > 0
            0, otherwise
```

The threshold is strict `>` and `0.1` is 10 mm/h in normalized space. Backpropagation uses the row-specific total objective, but checkpoint selection always uses validation `L_rain`. No nonnegativity or orographic term is allowed.

## 8. Evaluator implementation behavior

Use sufficient statistics rather than averaging precomputed ratios:

- continuous accumulator: absolute-error sum, squared-error sum, pixel count;
- categorical accumulator per threshold: integer `a,b,c,d`;
- structural accumulator: finite per-window SSIM sum/count;
- metadata accumulator: event ID and window count.

`TyphoonDataset(channel_indices=...)` already returns channels in requested subset order. The evaluator must not apply canonical indices to that tensor again. It should assert `X.shape[2] == len(input_channel_indices)` and treat precipitation as subset position zero, which is guaranteed by every frozen list.

The Markdown writer must iterate actual v2 result fields. It must not construct threshold keys using floating-point string formatting. Machine-readable JSON is authoritative; CSV/Markdown are derived views.

The evaluator must fail on non-finite predictions/targets, mismatched shapes, duplicate/missing event IDs, an unexpected split, or negative predictions below tolerance `-1e-7`. Values in `[-1e-7,0)` may be clamped to zero only for roundoff and the clamp count must be logged; the current ReLU architecture should produce none.

## 9. Checkpoint selection and artifact manifest

For P0–P3 and all Axis-I trainable models, each epoch logs:

- training `rain`, auxiliary components, and `total`;
- validation `rain`, auxiliary components, and `total`;
- learning rate and epoch.

“Best” means minimum validation `rain` MSE, with the earliest epoch winning an exact tie. The checkpoint filename remains deterministic within its experiment directory. Every formal artifact manifest records:

- canonical ID and aliases;
- git commit and dirty-state flag;
- config path and SHA-256;
- dataset, split, and normalization SHA-256;
- canonical input channel list;
- architecture fields and parameter count;
- optimizer/scheduler/seed/batch/epoch/AMP/software details;
- enabled loss components/weights/threshold;
- selection metric, best epoch, best validation rain MSE;
- checkpoint SHA-256;
- evaluator protocol ID and explicit `test_status: SEALED`.

Completion is never inferred from a filename alone.

## 10. E2 reuse and backbone-gate behavior

The audited local checkout currently contains both:

- `saved_models/E2_resconvlstm_seed42/E2_resconvlstm_seed42_best.pth`;
- `results/E2_resconvlstm_seed42/manifest.json`.

Therefore the current shell skips E2 locally. It is unsafe on a GPU host where the same checkpoint exists only at `outputs/E2_resconvlstm_20ep/models/ResConvLSTM_best.pth` or in an experiment backup: the current `else` branch retrains E2.

The fixed gate must enforce this algorithm:

1. Build an ordered candidate list from task-specific environment variable `E2_CHECKPOINT_PATH`, manifest `checkpoint_local_path`, `saved_models/E2_resconvlstm_seed42/E2_resconvlstm_seed42_best.pth`, `outputs/E2_resconvlstm_20ep/models/ResConvLSTM_best.pth`, and `outputs/experiments/E2_resconvlstm/models/ResConvLSTM_best.pth`.
2. For each existing candidate, call `scripts/verify_experiment_artifact.py` and require model `ResConvLSTM`, channels `[0]`, hidden dims `[64,128]`, kernel 3, seed 42, batch 4, maximum epochs 20, rain-only loss, compatible state-dict shapes, and a matching checkpoint hash when the manifest supplies one.
3. On the first valid candidate, print its resolved absolute path and run validation-only evaluator v2. Never invoke the training mode for E2.
4. If no candidate is valid, exit nonzero with `BLOCKED_MISSING_OR_INCOMPATIBLE_E2`; tell the operator to restore/stage the official checkpoint. Do not train a replacement.

The verifier must not accept a smoke artifact. A manifest path mismatch is recoverable if content/metadata verification succeeds and the new resolved path is recorded in a separate run manifest; do not rewrite historical provenance.

## 11. Exact scripts to generate

### `scripts/run_axis_i.sh`

After a common preflight, run in order:

```text
I2/E2: validate-only official checkpoint (never train)
I3/E3: train, then validation v2
I4/E4: train, then validation v2
I5/E5=P0: train once, then validation v2
```

The script uses `set -euo pipefail`, separate output directories, timestamped logs, formal-mode configs only, and no test option. It stops rather than skipping an incomplete/corrupt artifact. It may resume a completed I3–I5 only after artifact verification.

### `scripts/run_axis_ii_c1.sh`

First verify the completed I5/E5 artifact and register it as P0 without copying or training. Then run:

```text
P1: train, then validation v2
P2/E6: train, then validation v2
P3: train, then validation v2
```

It must refuse to start if P0 does not resolve to the exact I5 checkpoint hash. It must reject P4/P5 and never mention ERA5 as an available runtime input.

Both scripts must run a config-matrix preflight and require CUDA for trainable formal rows. `AMP=auto` means enabled when CUDA supports it and disabled otherwise; a formal GPU run records the resolved value.

## 12. Tests to add or update

`tests/test_evaluation_protocol_v2.py` must cover at least:

1. exact `a,b,c,d` counts on a hand-computed 2 × 2 fixture;
2. exact CSI/POD/FAR/HSS/BIAS/ACC values from those counts;
3. proof that pooled CSI differs from mean window CSI on a constructed example and that v2 returns the pooled value;
4. per-event pooling before ratio calculation;
5. `d` inclusion in HSS and ACC;
6. zero-positive truth/forecast denominators returning `NaN`, while all-dry ACC is 1;
7. dry windows contributing counts to global pooling;
8. global RMSE as square-root of pooled MSE, not mean window RMSE;
9. fixed-range SSIM invariance to unrelated per-window maxima and all-zero equality;
10. absence of legacy range-NRMSE and `peak_rel_error` from standard v2 output;
11. stable optional `NRMSE_fixed100` if implemented;
12. failure on NaN/Inf input.

`tests/test_experiment.py` must cover at least:

1. I4's 10-channel subset evaluates without canonical re-indexing or out-of-bounds access;
2. non-prefix channel subsets preserve canonical order and precip at subset position zero;
3. YAML hidden dims/kernel/seq_len/loss weights are honored and manifest-recorded;
4. P1–P3 checkpoint selection uses validation rain MSE even when composite loss ranks epochs differently;
5. validation report contains finite per-event CSI when pooled denominators exist and uses no manually guessed key;
6. the formal runner rejects test split/IDs and has no `--allow-test-eval` option;
7. the legacy evaluator entry point cannot default into 2024/test evaluation;
8. E2 missing/incompatible artifact stops the gate and no command contains an E2 training invocation;
9. I5 and P0 resolve to one config/artifact identity;
10. P4/P5 are blocked aliases with no runnable config;
11. output nonnegativity sanity check;
12. formal config validation rejects epoch 100, seed override, and unknown loss components.
13. TrajGRU training and evaluator use the same absolute prediction tensor (no extra persistence addition in Trainer).

Update the old `test_event_aggregation`, which currently asserts arithmetic means of window metrics; that assertion encodes evaluator-v1 behavior and must not survive.

## 13. Expected verification result

Before GPU use, all commands below must exit zero:

```text
python -m pytest -q
python scripts/verify_experiment_artifact.py --experiment I2 --checkpoint <resolved-E2-path> --manifest results/E2_resconvlstm_seed42/manifest.json
python scripts/run_experiment.py --config configs/experiments/E2_resconvlstm.yaml --mode validate-only --checkpoint <resolved-E2-path> --out outputs/revalidation_v2/E2_resconvlstm
```

Expected pytest result is **zero failures, zero errors, zero unexpected skips**. Test count is not frozen because the implementation adds parameterized cases. The E2 validation command must report 7 validation events / 1,266 windows, `protocol_id=evaluation_v2`, `test_status=SEALED`, finite pooled categorical scores where denominators are nonzero, and no test loader creation. It must not update checkpoint weights.

The old E2 NRMSE `1623.65774` and peak-relative error must not appear as v2 primary metrics. The v2 values are not required to reproduce legacy overall CSI because aggregation deliberately changes.

## 14. Pre-GPU stop conditions

Stop without training if any condition holds:

- repository baseline/provenance is unknown or unrelated dirty files would be overwritten;
- dataset, split, normalization, config, or checkpoint hash differs from the accepted manifest without review;
- any test fails or is unexpectedly skipped;
- any ordinary path can load/evaluate test IDs while sealed;
- E2 official checkpoint is missing/incompatible;
- a command would train E2 or train P0 separately from I5;
- config/runtime fields differ from the frozen matrix;
- checkpoint selection is not validation base MSE;
- evaluator returns mean-window categorical scores as primary, forces undefined denominators to zero, or omits `d`;
- evaluator double-slices channel subsets;
- TrajGRU training and evaluation apply different absolute/residual prediction semantics;
- P4/P5 becomes runnable or `u_move/v_move` is used as environmental wind;
- a formal trainable run has no CUDA device, batch size 4 cannot be used, or AMP resolution is not recorded;
- non-finite data/predictions, missing event IDs, split overlap, or unexpected window/event counts are detected.

OOM or infrastructure failure does not authorize a batch-size/model change. Stop, preserve logs, and request a new design decision.

## 15. GPU next-run plan

After implementation commit and E2 v2 validation acceptance:

1. Run `scripts/run_backbone_gate.sh`: I0 validation, I1 training, I2 validation-only reuse, B1 training.
2. Review manifests and v2 validation outputs; do not use test.
3. Run `scripts/run_axis_i.sh`: I3, I4, I5/P0.
4. Confirm the Axis-I adjacent comparisons and exact I5/P0 artifact identity.
5. Run `scripts/run_axis_ii_c1.sh`: P1, P2, P3.
6. Freeze the final model/comparison/report set using validation only.
7. Request a separate explicit authorization before the one-time final test phase.

## 16. Commit strategy for the implementation agent

Do not commit checkpoints, H5 files, outputs, logs, caches, or unrelated untracked files.

Use three reviewable commits after tests pass:

1. `freeze evaluation protocol v2 implementation` — metrics, evaluator, evaluator tests, and validation-only legacy wrapper.
2. `freeze design C configs and selection semantics` — trainer selection logic, runner fidelity, aliases/configs, config tests.
3. `add safe GPU gates for design C` — artifact verifier and three shell scripts, with no execution results.

Each commit must state that test remains sealed. Do not amend the historical E2 archival commit and do not commit any code implementation as part of the present architecture-review task.

## 17. Definition of implementation-ready

Implementation is ready for GPU only when the full test suite passes, E2 is validation-only re-evaluated under v2, every artifact/config preflight is reproducible, and no authorized script can train E2, duplicate P0, access test, or enable P4/P5. Until then, `GPU_TRAINING_GATE = CLOSED`.
