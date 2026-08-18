# Geo_AI Research Design C — Frozen Specification

**Status:** FROZEN FOR IMPLEMENTATION (validation stage)  
**Baseline:** `514d2ec` — `archive E2 GPU experiment and prepare backbone gate`  
**Forecast task:** 30-minute tropical-cyclone precipitation nowcasting  
**Data:** `ConvLSTM_Dataset_128.h5`, schema v2, 6,867 windows, 12 canonical channels  
**Split:** train 25 events / 4,894 windows; validation 7 events / 1,266 windows; test 4 events / 707 windows  
**Test status:** SEALED

This document supersedes the experiment framing in the Stage-1 literature exploration. It does not delete or rewrite `docs/literature_review_stage1.md`; that file remains provenance for the exploratory research phase.

## 1. Final paper positioning

The paper is a **two-axis controlled ablation study** for tropical-cyclone precipitation nowcasting:

1. **Input Information Ablation:** what storm-state and terrain information improves a fixed nowcasting model?
2. **Loss / Inductive-Bias Ablation:** after the model receives the full storm-state and terrain information, do spatial regularization and rare-event emphasis improve skill?

Physics is one possible component of the second question, not the label for every auxiliary loss. Smoothness is regularization; extreme-MSE is task-driven rare-event emphasis; nonnegativity is an architectural property; only a future, meteorologically valid orographic term may be called a physical prior.

The contribution is controlled empirical evidence, not a claim of a novel neural architecture and not a claim that every auxiliary loss is physics-informed.

## 2. Research questions

**RQ1 — Input information.** What information improves 30-minute tropical-cyclone precipitation nowcasting: storm state, static terrain, or terrain geometry?

The causal input-information contrasts are `I3 − I2`, `I4 − I3`, and `I5 − I4`. `I0`, `I1`, and `I2` establish persistence, plain-ConvLSTM, and residual-backbone reference levels; `I1 − I0` and `I2 − I1` are not input-information effects because the model family changes.

**RQ2 — Loss / inductive bias.** Given identical all-12-channel inputs and an identical ResConvLSTM backbone, do smoothness regularization, extreme-MSE emphasis, or their combination improve validation skill relative to MSE alone?

The controlled contrasts are `P1 − P0`, `P2 − P0`, and `P3 − P0`. The interaction may be reported as `P3 − P1 − P2 + P0` but is exploratory with one training seed.

## 3. Frozen common controls

Unless a row below explicitly says “not trainable,” every run uses:

- frozen event split `configs/splits_v1.yaml`;
- frozen train-only normalization `configs/normalization_v1.json` and precipitation scale 100 mm/h;
- sequence length 11 to one 30-minute target;
- hidden dimensions `[64, 128]`, kernel size `3`;
- AdamW, learning rate `1e-4`, weight decay `1e-4`;
- ReduceLROnPlateau factor `0.5`, patience `10`;
- gradient clip norm `1.0`, early-stopping patience `10`;
- seed `42`, batch size `4`, maximum `20` epochs, AMP `auto`;
- no augmentation;
- validation-only checkpoint selection using **unweighted base validation MSE**, even when the training objective contains auxiliary terms;
- evaluation protocol `docs/EVALUATION_PROTOCOL_V2.md`;
- no test loader construction, inference, metrics, or inspection.

Using a different validation selection loss across P0–P3 is prohibited because it would confound the loss ablation with checkpoint selection.

## 4. Axis I — Input Information Ablation

| ID | Model | Canonical input channels | Training loss | Role / valid contrast |
|---|---|---|---|---|
| I0 | Persistence | `[0]` | none | Non-trainable lower bound |
| I1 | PlainConvLSTM | `[0]` | MSE | Plain recurrent baseline |
| I2 | ResConvLSTM | `[0]` | MSE | Residual-backbone control; compare with I1 only as a backbone contrast |
| I3 | ResConvLSTM | `[0,1,2,3,4,5,6,7]` | MSE | Adds storm state / CMA geometry and translation; compare with I2 |
| I4 | ResConvLSTM | `[0,1,2,3,4,5,6,7,8,11]` | MSE | Adds DEM and land mask; compare with I3 |
| I5 | ResConvLSTM | `[0,1,2,3,4,5,6,7,8,9,10,11]` | MSE | Adds terrain gradients; compare with I4; identical to P0 |

Canonical channel meanings are fixed: `0 precipitation`, `1 center_wind_speed`, `2 center_pressure`, `3 r_norm`, `4 dx_norm`, `5 dy_norm`, `6 u_move`, `7 v_move`, `8 dem`, `9 dh_dx`, `10 dh_dy`, `11 land_mask`.

## 5. Axis II — Loss / Inductive-Bias Ablation, Stage C1

All rows use all 12 canonical channels and the same ResConvLSTM. The base normalized-space loss is

`L_MSE = mean((P_hat − P_true)^2)`.

The frozen auxiliary definitions are those stated in `docs/MINIMAX_IMPLEMENTATION_SPEC.md`: `lambda_smooth = 0.01`; `lambda_extreme = 0.5`; extreme threshold `P_true > 10 mm/h` (strictly greater than); and no orographic or nonnegativity term.

| ID | Enabled components | Total training objective | Reuse rule |
|---|---|---|---|
| P0 | rain | `L_MSE` | Exactly I5; one checkpoint and one run only |
| P1 | rain + smooth | `L_MSE + 0.01 L_smooth` | New training required |
| P2 | rain + extreme | `L_MSE + 0.5 L_extreme` | Legacy alias E6; new training required unless a future run passes the frozen manifest checks |
| P3 | rain + smooth + extreme | `L_MSE + 0.01 L_smooth + 0.5 L_extreme` | New training required |

There is no nonnegativity training run. For ResConvLSTM, `P_hat = ReLU(P_prev + delta_P)`, so `P_hat >= 0` is a hard architectural guarantee. The required architectural sanity control asserts zero negative outputs on validation inference and fails if any output is negative beyond numerical tolerance.

## 6. Stage C2 — Conditional ERA5 extension

`P4` (orographic prior) and `P5` (full stack) are **BLOCKED_BY_ENVIRONMENTAL_WIND_DATA**. They have no runnable configuration in this freeze.

Enabling Stage C2 requires a separately reviewed data design providing time- and grid-aligned environmental atmospheric fields, initially at 850 hPa:

- zonal wind `u`;
- meridional wind `v`;
- specific humidity `q`.

The 700-hPa level may be evaluated only after the 850-hPa design. The distinction must remain explicit:

- `V · grad(h)` is a terrain-aligned environmental uplift proxy;
- `q(V · grad(h))` is a moisture-weighted orographic forcing proxy.

Storm translation `u_move/v_move` is not environmental wind and may not substitute for `u/v`. No ERA5 download, H5 rebuild, proxy-physics implementation, P4/P5 config, or P4/P5 run is authorized by this specification.

## 7. Legacy provenance and alias map

Legacy files remain in place. New IDs are aliases, not renamed or duplicated historical experiments.

| Legacy ID | Frozen alias | Meaning | Current reuse status |
|---|---|---|---|
| E0 | I0 | Persistence `[0]` | Re-run validation after evaluator v2; no training |
| E1 | I1 | PlainConvLSTM `[0]`, MSE | Formal run not yet complete |
| E2 | I2 | ResConvLSTM `[0]`, MSE | **Reuse existing checkpoint; never retrain** |
| E3 | I3 | ResConvLSTM `[0..7]`, MSE | Formal run not yet complete |
| E4 | I4 | ResConvLSTM `[0..8,11]`, MSE | Formal run not yet complete |
| E5 | I5 and P0 | all-12 ResConvLSTM, MSE | One new formal run, reused across both axes |
| E6 | P2 | all-12 ResConvLSTM, MSE + extreme-MSE | Formal run not yet complete |
| B1 | backbone reference only | TrajGRU `[0]`, MSE | Formal run not yet complete; not an Axis-I input contrast |

The alias registry is to be materialized as `configs/experiment_aliases_v2.yaml`. It must not contain P4 or P5 runnable configs.

## 8. Existing-run reuse and new training

The completed E2/I2 artifact is the checkpoint identified by `results/E2_resconvlstm_seed42/manifest.json`; the audited local path exists at `saved_models/E2_resconvlstm_seed42/E2_resconvlstm_seed42_best.pth`. It records seed 42, batch 4, 20 epochs, precipitation-only input, and best epoch 18. The model weights are reusable.

The published E2 validation metrics are **legacy evaluator v1 outputs**. They must be recomputed on validation only with evaluator v2; re-evaluation is not retraining. Until then, the recorded MAE `0.23722`, RMSE `0.70091`, SSIM `0.90271`, and CSI values must be labeled legacy because RMSE/SSIM aggregation and categorical aggregation change in v2. NRMSE and peak-relative error are invalid as primary results.

Required new work after the implementation gate:

- validation-only I0/E0;
- training I1/E1 and B1 for the backbone gate;
- training I3/E3, I4/E4, and I5/E5=P0;
- training P1, P2/E6, and P3;
- validation-only v2 re-evaluation of every accepted checkpoint, including E2/I2.

No run may train I2/E2 or train P0 separately from I5.

## 9. Test-sealing rule

The test event IDs `2306`, `2310`, `2402`, and `2418` and their 707 windows remain sealed until all of the following are frozen and committed: experiment matrix, loss weights, checkpoint-selection rule, evaluator v2, statistical analysis, reporting tables, and the final set of comparisons.

Before unsealing, code must not construct a test `Dataset` or `DataLoader`, load test targets for inference, compute test predictions, print test metrics, or use test behavior to choose any design. The existing general evaluation entry point that defaults to 2024 must be guarded before any GPU work. Final test evaluation is a single, separately authorized phase. After it, no hyperparameter, threshold, model, or reporting-rule changes are permitted; any unavoidable correction must be disclosed and all affected models re-evaluated without selection on test.

## 10. Seed policy

The frozen primary design uses seed `42` once per trainable experiment. The same seed controls Python, NumPy, PyTorch, CUDA, and training-loader shuffling; deterministic CuDNN mode remains enabled. The manifest must record software/CUDA versions and whether exact deterministic execution was achieved.

Event-level uncertainty is not initialization uncertainty. The paper must explicitly disclose that a single training seed does not establish robustness to random initialization. A later multi-seed sensitivity study is outside this frozen matrix; if separately approved, it must use the identical seed set for every trainable row and may not be motivated by test results.

## 11. Experiment-comparison rules

1. Only adjacent Axis-I information contrasts (`I3−I2`, `I4−I3`, `I5−I4`) support claims about added information.
2. Only `P1−P0`, `P2−P0`, and `P3−P0` support loss/inductive-bias claims.
3. I5 and P0 must resolve to the same artifact identity, not merely equal-looking configs.
4. All comparisons use the same evaluator version and validation predictions. Legacy-v1 metrics may not be mixed with v2 metrics.
5. Model selection is by validation base MSE for every trainable row. Auxiliary-loss totals may be logged but may not select the epoch.
6. Per-event paired analysis treats a typhoon, not a window, as the independent unit. Windows are temporally correlated and may not be used as independent replicates.
7. Primary categorical results pool contingency counts first. Primary continuous and SSIM aggregation follow `docs/EVALUATION_PROTOCOL_V2.md`.
8. Thresholds `[5, 10, 20, 30]` mm/h are frozen before test access.
9. No row may be rerun selectively because its validation result is disappointing. Failed infrastructure runs may restart only with the reason and artifact disposition recorded.
10. Parameter counts and effective architecture fields must be recorded so that a config parsing error cannot silently change the comparison.

## 12. Forbidden claims

- Do not describe the whole study or all auxiliary losses as “physics constraints” or “physics-informed.”
- Do not call smoothness a physical law; call it spatial/temporal regularization.
- Do not call extreme-MSE physical; call it task-driven rare-event emphasis.
- Do not claim a nonnegativity loss improvement; nonnegativity is hard-coded by ReLU.
- Do not call `u_move/v_move` environmental wind or use them in an orographic forcing claim.
- Do not write `q(V · grad(h))` unless moisture `q` and environmental wind `V` are actually available and aligned.
- Do not infer an information effect from I1 versus I2, where the backbone changes.
- Do not claim independent-sample significance from window-level tests.
- Do not claim generalization from validation results or initialization robustness from seed 42 alone.
- Do not call B1 a state-of-the-art benchmark without a scoped literature- and implementation-equivalence justification.

## 13. Current scientific and engineering risks

1. `src/evaluation/evaluator.py` currently averages window-level categorical ratios. This is not the frozen pooled-count definition.
2. `scripts/run_experiment.py` writes per-event CSI using keys `CSI_10.0mmh` and `CSI_20.0mmh`, while metrics emit `CSI_10mmh` and `CSI_20mmh`; the displayed `nan` values are a report-key bug.
3. Current NRMSE divides each window RMSE by that window's observed range with a `1e-6` floor, producing pathological values for nearly dry windows. Current `peak_rel_error` has the same denominator pathology.
4. Current SSIM uses a prediction/observation-dependent scale per window, so legacy SSIM is not the frozen v2 definition.
5. `evaluate_model` applies canonical channel indices a second time after `TyphoonDataset` has already produced a subset. I4/E4 can index beyond its 10-channel tensor; other subsets can silently rely on positional coincidence.
6. `run_experiment.py` does not honor configured `hidden_dims` or `kernel_size`, hard-codes physics-loss enablement, and omits explicit `lambda_smooth` propagation. Defaults currently mask some errors but do not constitute a frozen implementation.
7. Existing experiment YAML files say 100 epochs; the backbone shell overrides them to 20, but direct execution would violate the freeze.
8. Current Trainer selects the best epoch by each experiment's composite validation objective; P0–P3 require a shared base-MSE selection rule.
9. `src/evaluation/evaluate_models.py` defaults to year 2024 and uses an older year-based loader, creating a test-seal bypass and a divergent evaluation path.
10. The E2 gate recognizes only one checkpoint layout. It skips correctly in the audited local workspace, where both expected files exist, but it can retrain E2 on AutoDL when the same artifact is under `outputs/E2_resconvlstm_20ep/...` or an external backup.
11. TrajGRU has inconsistent prediction semantics: the Trainer treats its already-ReLU absolute output as a residual and adds `P_prev`, while the evaluator treats the same output as absolute. B1 is invalid until train/evaluation semantics are identical.
12. Seven validation events and four future test events provide limited inferential power, especially at high thresholds; effect sizes and event-level uncertainty must be emphasized.
13. Single-seed training leaves initialization variance unmeasured.

## 14. Freeze decision

The conceptual design is frozen. GPU execution remains blocked until every pre-GPU item and test in `docs/MINIMAX_IMPLEMENTATION_SPEC.md` passes. P4/P5 remain independently blocked by environmental wind/moisture data design.
