# IMPLEMENTATION_STATUS — Risk resolution against the frozen design

> Aligned with `docs/RESEARCH_DESIGN_C_FREEZE.md` §13.
> The frozen research design itself is **not modified** by this document; this file records the resolution status of each historical implementation risk identified before the validation matrix was launched.

Status labels:

| Symbol | Meaning |
|---|---|
| ✅ **RESOLVED** | Implemented and verified by a frozen test in `tests/`. |
| 🟡 **BLOCKED** | Cannot be closed without an external action (GPU host, data, or external authorization). Documented; no code change required to *unblock*; an unblock event will be recorded in §"Unblock log". |
| 🟠 **OPEN**     | Implementation is not yet complete; not currently exercised by the matrix; planned for a future branch. |

## 1. Status table

| §13 risk | Status | Evidence (commit / script / test) |
|---|---|---|
| 1. `evaluator.py` averaged window-level categorical ratios | ✅ RESOLVED | `src/evaluation/evaluator.py::aggregate_v2` (commit `ea50b08`); `tests/test_evaluation_protocol_v2.py::test_pooled_contingency_count_first` |
| 2. `run_experiment.py` reported wrong CSI keys | ✅ RESOLVED | `src/evaluation/reporting.py` iterates `threshold_key`; `tests/test_evaluation_protocol_v2.py::test_threshold_key_unified` |
| 3. NRMSE / `peak_rel_error` denominator pathology | ✅ RESOLVED | `docs/EVALUATION_PROTOCOL_V2.md` §14–§15; `tests/test_evaluation_protocol_v2.py::test_legacy_nrmse_dropped` |
| 4. Per-window SSIM data_range | ✅ RESOLVED | `src/evaluation/metrics.py::structural_similarity_window(data_range=100)`; `tests/test_evaluation_protocol_v2.py::test_ssim_fixed_range_100` |
| 5. Evaluator re-sliced canonical indices | ✅ RESOLVED | `src/evaluation/evaluator.py::evaluate_model_v2` asserts `X.shape[2] == len(channel_indices)`; `tests/test_evaluation_protocol_v2.py::test_no_double_slice` |
| 6. `run_experiment.py` ignored hidden_dims / loss weights | ✅ RESOLVED | `scripts/run_experiment.py` formal-config-only; `tests/test_experiment.py::test_run_experiment_formal_config_only` |
| 7. YAML epoch mismatch | ✅ RESOLVED | All `configs/experiments/*.yaml` declare `epochs: 20`; `tests/test_experiment.py::test_config_epoch_matches_freeze` |
| 8. Trainer selected best epoch by composite loss | ✅ RESOLVED | `src/training/trainer.py` selects by validation base `rain_mse`; `tests/test_experiment.py::test_selection_metric_is_rain_mse_only` |
| 9. `evaluate_models.py` defaulted to 2024 (test bypass) | ✅ RESOLVED | Test-seal; `scripts/evaluate_checkpoint.py --split test` refuses; `tests/test_evaluation_protocol_v2.py::test_test_sealed_in_all_normal_runners`; `tests/test_experiment.py::test_checkpoint_eval_test_refused` |
| 10. E2/I2 retrain risk (AutoDL checkpoint layout divergence) | 🟡 BLOCKED on GPU host | `scripts/run_backbone_gate.sh` refuses retrain; `scripts/verify_experiment_artifact.py` COMPATIBLE on local artifact; **v2 re-evaluation pending on the GPU host** — see `EXPERIMENT_PHASE_REPORT.md` §0 |
| 11. TrajGRU semantics mismatch | ✅ RESOLVED | Trainer + evaluator consume absolute ReLU; `tests/test_experiment.py::test_trajgru_train_eval_semantics_match`; `tests/test_evaluation_protocol_v2.py::test_trajgru_absolute_semantics` (device-portability `1391b2d`) |
| 12. Limited inferential power (7 val + 4 test events) | 🟡 BLOCKED on data | Inherent to the design — disclosed in `docs/LEAKAGE_AND_TEST_LIMIT_AUDIT.md`; the analysis surface `scripts/analyze_ablation_results.py` enforces `n_pairs ≥ 4` for inferential reporting (raw + Holm) and refuses to fabricate CI/p at lower n |
| 13. Single-seed training | 🟠 OPEN | Acknowledged limitation of the current matrix; multi-seed extension is OUT of scope for this PR and requires a separate freeze amendment |

> Note: `137–13` are aligned 1:1 with `RESEARCH_DESIGN_C_FREEZE.md` §13 numbered items. If the design freeze ever numbers its risks differently, this table is updated — not the freeze itself.

## 2. How to re-audit

```bash
pytest -q tests/test_evaluation_protocol_v2.py tests/test_experiment.py
# expected: all tests pass (target 69/69 at gate commit ea50b08)

python scripts/verify_experiment_artifact.py \
  --config configs/experiments/E2_resconvlstm.yaml \
  --checkpoint saved_models/E2_resconvlstm_seed42/E2_resconvlstm_seed42_best.pth
# expected last line: STATUS: COMPATIBLE

git diff -- scripts/run_experiment.py scripts/evaluate_checkpoint.py src/evaluation/evaluate_models.py | head
# expected: empty diff against the implementation gate (ea50b08)
```

## 3. Unblock log

| Date | Risk | Event | Author |
|---|---|---|---|
| _none recorded yet_ |  |  |  |

## 4. Notes for branches

A working branch that adds *paper artifact* or *analysis* infrastructure MUST NOT close any 🟡 BLOCKED or 🟠 OPEN row above; that is the responsibility of a later branch whose scope explicitly includes the unblock action.

## 5. Closing

This file is frozen at the implementation gate (commit `ea50b08` plus device-portability `1391b2d`). It is only appended to in §3 (unblock events) and §1 (status-class reassignments) when a new commit explicitly closes an item.
