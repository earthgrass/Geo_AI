# EXPERIMENT_PHASE_REPORT — 正式实验阶段报告

> 生成日期：2026-08-14
> 2026-08-18 更新：首次 GPU 实验 E2 完成（legacy evaluator v1）
> 2026-08-18 更新：evaluator v2 与双轴实验基础设施落地（implementation gate）

---

## 0. 当前状态（2026-08-18）

| 字段 | 值 |
|---|---|
| RESEARCH_DESIGN | TWO_AXIS_CONTROLLED_ABLATION |
| EVALUATOR_VERSION | V2_IMPLEMENTED |
| GPU_TRAINING_GATE | CLOSED（全部 tests 通过、E2 v2 重评估完成后方可 OPEN_FOR_VALIDATION_TRAINING） |
| TEST_STATUS | SEALED |
| E2_TRAINING | COMPLETED（seed 42, batch 4, 20 epochs, best epoch 18, best val MSE 0.0001598686） |
| E2_EVALUATOR_V1_RESULT | ARCHIVED（results/E2_resconvlstm_seed42/，禁止覆盖） |
| E2_EVALUATOR_V2_REEVALUATION | PENDING（artifact verifier 已确认 COMPATIBLE；v2 重评估在 CPU 上约 80 min，本环境长时后台进程会被终止，需在 GPU 主机执行——命令见 §18.2） |
| BACKBONE_GATE | PARTIAL（E2 完成；E0/E1/B1 待 GPU） |

## 1. commit SHA
`62e9ba0`（科研决策报告）+ 本轮工程改造 commits（见 §15）

## 2. pytest 数量
**69 passed**（2026-08-18 implementation gate 后；含 `test_evaluation_protocol_v2.py` 与更新后的 `test_experiment.py`）

## 3. channel subset implementation
- ✅ `src/data/dataset.py`：`TyphoonDataset(channel_indices=[...])`，`_reconstruct_input` 构建 12 通道后按 canonical 索引选取子集。
- ✅ `src/data/transforms.py`：`ChannelNormalize`（precip min-max + track/terrain z-score，按 canonical 通道索引）。
- ✅ `src/evaluation/evaluator.py` v2：**不再二次切片** —— 对已子集的 tensor 只断言通道数（`X.shape[2] == len(channel_indices)`），precip 恒在子集位置 0。
- ✅ 测试 `test_i4_channel_subset_no_double_slice` / `test_canonical_channel_semantics_preserved`。

## 4. test sealing status
- ✅ `scripts/run_experiment.py`：**移除 `--allow-test-eval`**；只构建 train/val loader；test 事件仅被丢弃。
- ✅ `src/evaluation/evaluate_models.py`：已弃用，fail-fast，无法进入 2024/test。
- ✅ `scripts/evaluate_checkpoint.py`：`--split test` 直接拒绝（`SystemExit`）。
- ✅ 回归测试 `test_test_sealed_in_all_normal_runners` / `test_checkpoint_eval_test_refused` / `test_e2_reuse_never_retrains`。

## 5. Phase 1 results（Backbone sanity）
**PARTIAL**：E2 ResConvLSTM 已在 GPU 完成（seed 42，20 epoch，best val MSE 0.0001598686）；E0/E1/B1 待 GPU。

CPU smoke test（E2 ResConvLSTM，8 样本 × 1 epoch）：
- params = 1,155,714
- val loss = 0.000686，1 epoch = 25.5s（含模型初始化 + 数据加载）

## 6. Backbone Gate
**PENDING（需要 GPU 环境执行）** — `scripts/run_backbone_gate.sh` 已按 spec 修复：
- E2/I2 **永不重训**；通过 artifact verifier 解析官方 checkpoint，找不到即 `BLOCKED_MISSING_OR_INCOMPATIBLE_E2`。
- 顺序：I0 validation → I1 train → I2 validate-only reuse → B1 train。

## 7. Phase 2 results（Terrain ablation）
**未运行**（依赖 GPU，`scripts/run_axis_i.sh`）。

## 8. E3 vs E4 conclusion
**未运行**（pending）

## 9. E4 vs E5 conclusion
**未运行**（pending）

## 10. E5 vs E6 conclusion
**未运行**（pending；Axis II 由 `scripts/run_axis_ii_c1.sh` 执行 P1/P2/P3）

## 11. TrajGRU comparison
- ✅ `src/models/trajgru.py` 已实现（precipitation-only、含 warp）。
- ✅ **训练/评估语义已统一**：Trainer 与 evaluator 均将 TrajGRU 视为绝对输出模型（`forward` 已含 ReLU head），不再二次加 `P_prev`（`test_trajgru_train_eval_semantics_match`）。
- ⚠️ 未训练（GPU）。

## 12. event-level validation results
- ✅ `src/evaluation/evaluator.py` v2：per_window / per_event / overall_global / overall_window_mean 四级；**先聚合 contingency counts 再算 skill**；零分母返回 NaN。
- ✅ 阈值 key 统一 `threshold_key`（`CSI_10mmh` 而非 `CSI_10.0mmh`），修复 legacy per-event CSI 显示 nan 的报告 bug。
- ✅ SSIM 固定 `data_range=100 mm/h`；NRMSE/peak_rel_error 移出 primary。
- ✅ 测试 `test_evaluation_protocol_v2.py`（含 pooled CSI ≠ mean window CSI、全局 RMSE = sqrt(pooled MSE)、dry window 贡献 counts 等）。

## 13. terrain regime analysis
- ✅ `scripts/fit_eval_thresholds.py`（仅 train 事件）：
  - `HIGH_DEM = 585.00 m`（train-land DEM P75）
  - `HIGH_GRAD = 16.26 m/km`（train-land |∇h| P75）
- ✅ 写入 `configs/evaluation_thresholds_v1.json`（冻结）。

## 14. ERA5_EXTENSION_RECOMMENDED
**YES**（仅在 orographic 物理扩展需要时；核心论文不需要）——依据 `ACADEMIC_RESEARCH_DECISION_REPORT.md` §8/§14。P4/P5 = BLOCKED_BY_ENVIRONMENTAL_WIND_DATA，无 runnable 配置。

## 15. Git commits（建议/实际）
1. `62e9ba0` add academic research decision report ✅
2. `776d2c7` freeze dual-axis research design and evaluation protocol v2 ✅
3. `implement evaluator v2 and dual-axis experiment infrastructure`（本轮，见 FINAL_REPORT）

## 16. unresolved risks
- **无 GPU**：完整训练需 GPU 环境（RTX 4090 级别）。
- test 仅 4 事件：统计检验弱（已在决策报告明确事件级方案）。
- 归一化 z-score 依赖 `normalization_v1.json`（train-only，已冻结）。
- checkpoint selection 已统一为 **validation base rain MSE**（`checkpoint_selection_metric: rain_mse`），P0–P3 不再被 composite loss 干扰。

## 17. TEST_STATUS
**SEALED**

## 18. exact next recommended action
1. 全部 tests 通过（69 passed）+ **E2 v2 重评估** 完成后：`GPU_TRAINING_GATE = OPEN_FOR_VALIDATION_TRAINING`。
2. E2/I2 v2 validation-only 重评估（不训练、不碰 test）：
   ```
   python scripts/evaluate_checkpoint.py \
     --config configs/experiments/E2_resconvlstm.yaml \
     --checkpoint saved_models/E2_resconvlstm_seed42/E2_resconvlstm_seed42_best.pth \
     --split val --out results/I2_resconvlstm_seed42_eval_v2
   ```
3. 在 **GPU 环境** 依次运行：
   - `bash scripts/run_backbone_gate.sh`（I0/I1/I2/B1）
   - `bash scripts/run_axis_i.sh`（I2 reuse, I3/I4/I5 train）
   - `bash scripts/run_axis_ii_c1.sh`（P0=I5 reuse, P1/P2/P3 train）
4. 全程 **只用 train+val**，不碰 test。
5. 汇总 v2 validation 结果后，单独申请 final test 授权。

---

## 最终判定

**PARTIAL（BACKBONE_GATE_STATUS / GPU_TRAINING_GATE=CLOSED）**：evaluator v2 与双轴实验基础设施已实现并全部 tests 通过；E2 已完成（legacy v1 归档）；evaluator-v2 validation-only 重评估仍 PENDING，待在 GPU 主机完成后方可打开 GPU_TRAINING_GATE。剩余 E0/E1/B1 与 I3/I4/I5、P1/P2/P3 由三个 GPU 脚本在下次 AutoDL 开机时执行。

**不宣称任何论文结论。当前仅 E2 一项 validation 证据；test 仍 SEALED。**
