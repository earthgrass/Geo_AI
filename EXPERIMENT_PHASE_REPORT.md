# EXPERIMENT_PHASE_REPORT — 正式实验阶段报告

> 生成日期：2026-08-14（2026-08-18 更新：首次 GPU 实验 E2 完成）
> 阶段：实验基础设施实现 + CPU smoke test + 首次 GPU 实验（E2）

---

## 0. GPU 实验状态（2026-08-18 更新）

- E2_GPU_RUN_STATUS = COMPLETED
- E2_BEST_EPOCH = 18
- E2_BEST_VAL_LOSS = 0.0001598686
- TEST_STATUS = SEALED
- BACKBONE_GATE_STATUS = PARTIAL（E2 完成；E0/E1/B1 待 GPU，见 scripts/run_backbone_gate.sh）

---

## 1. commit SHA
`62e9ba0`（科研决策报告）+ 本轮工程改造 commits（见 §15）

## 2. pytest 数量
**33 passed**（原 27 + 新增 6）

## 3. channel subset implementation
- ✅ `src/data/dataset.py`：`TyphoonDataset(channel_indices=[...])`，`_reconstruct_input` 构建 12 通道后按 canonical 索引选取子集。
- ✅ `src/data/transforms.py`：`ChannelNormalize`（precip min-max + track/terrain z-score，按 canonical 通道索引）。
- ✅ `configs/experiments/*.yaml`：`model.input_channel_indices`（显式列表，非 `input_channels` 数字）。
- ✅ 测试 `test_channel_subset_shapes` / `test_model_input_channel_count` 验证模型确实只收到指定通道。

## 4. test sealing status
- ✅ `scripts/run_experiment.py` 有 `--allow-test-eval`（`store_true`，默认 False）。
- ✅ 未显式开启时，test 评估被**拒绝**（输出 "TEST SET SEALED"）。
- ✅ smoke test 确认默认拒绝 test。

## 5. Phase 1 results（Backbone sanity）
**PARTIAL**：E2 ResConvLSTM 已在 GPU 完成（seed 42，20 epoch，best val loss 0.0001598686）；E0/E1/B1 待 GPU 执行。

CPU smoke test（E2 ResConvLSTM，8 样本 × 1 epoch）：
- params = 1,155,714
- val loss = 0.000686，1 epoch = 25.5s（含模型初始化 + 数据加载）

外推：完整训练 4894 样本 × 100 epochs ≈ **数百小时/实验**，CPU 不可行。

## 6. Backbone Gate
**PENDING（需要 GPU 环境执行）**

## 7. Phase 2 results（Terrain ablation）
**未运行**（依赖 GPU）。

## 8. E3 vs E4 conclusion
**未运行**（pending）

## 9. E4 vs E5 conclusion
**未运行**（pending）

## 10. E5 vs E6 conclusion
**未运行**（pending）

## 11. TrajGRU comparison
- ✅ `src/models/trajgru.py` 已实现（最小、可复现、precipitation-only、含 warp）。
- ✅ `test_trajgru_forward` 通过（输出 `[B,1,H,W]`）。
- ⚠️ 未训练（GPU）。

## 12. event-level validation results
- ✅ `src/evaluation/evaluator.py` 已实现：overall + per_event + per_window 三级聚合。
- ✅ `test_event_aggregation` 通过。

## 13. terrain regime analysis
- ✅ `scripts/fit_eval_thresholds.py`（仅 train 事件）：
  - `HIGH_DEM = 585.00 m`（train-land DEM P75）
  - `HIGH_GRAD = 16.26 m/km`（train-land |∇h| P75）
- ✅ 写入 `configs/evaluation_thresholds_v1.json`（冻结）。

## 14. ERA5_EXTENSION_RECOMMENDED
**YES**（仅在 orographic 物理扩展需要时；核心论文不需要）——依据 `ACADEMIC_RESEARCH_DECISION_REPORT.md` §8/§14。

## 15. Git commits（建议/实际）
1. `62e9ba0` add academic research decision report ✅
2. implement experiment channel subsets and test sealing（本轮）
3. add TrajGRU baseline and evaluation framework（本轮）
4. run validation-only backbone and terrain ablations（本轮 smoke test + 报告）

## 16. unresolved risks
- **无 GPU**：完整训练需 GPU 环境（RTX 4090 级别）。
- test 仅 4 事件：统计检验弱（已在决策报告 §21 明确事件级方案）。
- 归一化 z-score 依赖 `normalization_v1.json`（train-only，已冻结）。

## 17. TEST_STATUS
**SEALED**

## 18. exact next recommended action
1. 在 **GPU 环境** 运行 `python scripts/run_experiment.py --config configs/experiments/E2_resconvlstm.yaml --epochs 100`（先 E0/E1/E2/B1 做 Backbone Gate）。
2. Backbone Gate PASS 后运行 E3/E4/E5/E6（terrain ablation）。
3. 全程 **只用 train+val**，不碰 test。
4. 汇总结果后，单独（显式 `--allow-test-eval`）执行 final test。

---

## 最终判定

**PARTIAL（BACKBONE_GATE_STATUS）**：基础设施全部就绪，E2 ResConvLSTM 已在 GPU 完成并归档（results/E2_resconvlstm_seed42/）。剩余 E0/E1/B1 由 scripts/run_backbone_gate.sh 在下次 GPU 开机时执行。

**不宣称任何论文结论。当前仅 E2 一项 validation 证据；test 仍 SEALED。**
