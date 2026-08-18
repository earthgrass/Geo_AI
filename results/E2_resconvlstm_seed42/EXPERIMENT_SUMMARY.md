# E2 ResConvLSTM — Experiment Summary (seed 42)

> 归档日期：2026-08-18
> 状态：**COMPLETED**（train + val only；test SEALED）
> 权重：`saved_models/E2_resconvlstm_seed42/E2_resconvlstm_seed42_best.pth`（不入 git）

---

## 1. 实验配置

| 项 | 值 |
|----|----|
| 实验 / 模型 | E2 / ResConvLSTM（precipitation-only，canonical channel `[0]`） |
| hidden dims / kernel | [64, 128] / 3 |
| seed | 42 |
| batch_size | 4 |
| max_epochs | 20（config 写 100，实际以 `--epochs 20` 跑） |
| AMP | True |
| device | cuda |
| 损失 | MSE（`components: [rain]`；λ_extreme 已配置但未启用） |
| 归一化 | `configs/normalization_v1.json`（train-only，冻结） |
| 划分 | train 25 / val 7 / test 4 事件（4894 / 1266 / 707 窗口） |
| 训练 commit | `cbcd46869d82dfc9fe7b9f811272c2404786e0d9`（本地 HEAD，归档时假定=云端训练 commit） |
| n_params | 1,155,714 |
| runtime | 1365.6 s |

---

## 2. 关键结果（validation）

| 指标 | 值 |
|------|----|
| best_epoch | 18 |
| best_val_loss (MSE, 归一化) | 0.0001598686 |
| MAE | 0.23722 mm/h |
| RMSE | 0.70091 mm/h |
| SSIM | 0.90271 |
| CSI@5 / 10 / 20 / 30 | 0.23397 / 0.14652 / 0.04907 / 0.01469 |
| POD@10 / FAR@10 / HSS@10 | 0.18786 / 0.18120 / 0.20441 |
| center_displacement | 45.61 km |
| peak_error | 7.57 mm/h |

---

## 3. history.json 分析（收敛 / 过拟合）

Loss 均为归一化 MSE（单 rain 分量，train==total）。

| Epoch | Train (×1e-5) | Val (×1e-5) |
|-------|---------------|-------------|
| 1 | 11.659 | 18.280 |
| 2 | 10.372 | 17.043 |
| 3 | 10.079 | 16.944 |
| 4 | 9.927 | 17.511 |
| 5 | 9.788 | 16.777 |
| 6 | 9.756 | 16.955 |
| 7 | 9.692 | 16.198 |
| 8 | 9.676 | 16.395 |
| 9 | 9.609 | 16.869 |
| 10 | 9.574 | 16.691 |
| 11 | 9.562 | 16.662 |
| 12 | 9.545 | 19.657 |
| 13 | 9.516 | 16.611 |
| 14 | 9.524 | 16.128 |
| 15 | 9.489 | 16.293 |
| 16 | 9.459 | 16.166 |
| 17 | 9.444 | 17.333 |
| 18 | 9.393 | **15.987** |
| 19 | 9.406 | 16.759 |
| 20 | 9.377 | 16.309 |

### 结论

- **基本收敛：是。** val loss 在 epoch 7 就降到 1.62e-4 并进入平台期，之后 13 个 epoch 在 [1.60e-4, 1.97e-4] 内波动，无下降趋势。train loss 前几 epoch 快速下降，之后近乎平坦（最后 10 个 epoch 仅再降 ~2%）。
- **明显过拟合：否。** train–val 差距约 1.7×（9.4e-5 vs 1.6e-4），但该差距**稳定**——train 继续缓慢下降时 val 并未恶化，无经典过拟合的"val 上翘"特征。
- **20 epoch 作为统一默认是否合理：是。** 平台期在 ~7 epoch 已到达，20 epoch 已充分覆盖且能捕获 best checkpoint（early-stopping patience=10 本也会在 ~epoch 17-18 停）。唯一注意点：val 在平台期是噪声，`best_epoch=18` 有随机性，换 seed 可能落在 14–19。这正是最终 test 用 3-seed（42/123/2026，见 PRE_FINAL_TEST_FREEZE.md）取均值的原因。

---

## 4. validation.md 审计（只审计，未改任何指标定义/代码）

### 4.1 NRMSE = 1623.65774 —— 定义/实现异常（确认）

**根因**：`src/evaluation/metrics.py:62-63`

```python
p_range = max(P_true.max() - P_true.min(), 1e-6)
metrics['NRMSE'] = metrics['RMSE'] / p_range
```

NRMSE 定义为 `RMSE / (单窗口真值极差)`，然后 `evaluator._aggregate` 对**逐窗口比值求平均**。降水场高度稀疏，绝大多数窗口真值极差趋近 0（`max-min` 极小），导致分母极小、比值爆到 1e2–1e4 量级，均值即 1623.66。**该值无物理意义**。

同类问题：`peak_rel_error = 52384.10`（`metrics.py:78-80`，`peak_error / P_true.max()`，同样逐窗口小分母 + 比值平均）。

**建议（待确认后再改）**：NRMSE 应使用数据集级归一化（`RMSE / (全局 max-min)` 或 `RMSE / 全局均值`），由汇总后的总 RMSE 与总量纲一次计算，而非逐窗口比值平均。peak_rel_error 同理。

### 4.2 per-event CSI_10 / CSI_20 全 nan —— key 命名 bug（确认，非"无命中"）

**根因**：`scripts/run_experiment.py:219` `_write_results`

```python
CSI_10={m.get('CSI_10.0mmh', float('nan')):.4f} CSI_20={m.get('CSI_20.0mmh', float('nan')):.4f}
```

但 `metrics.py:254` 生成的实际 key 是 `f'{k}_{thresh:.0f}mmh'` → **`CSI_10mmh` / `CSI_20mmh`**（`.0f` 把 `10.0` 格式成 `10`）。

因此 `m.get('CSI_10.0mmh')` 永远查不到，返回默认 `nan`。

- **overall CSI 有值**：因为 `_write_results` 对 overall 是 `sorted(overall.items())` 直接遍历真实 key，所以 `CSI_10mmh: 0.14652` 正常显示。
- **per-event 底层数据是正确的**：`res["per_event"]` 里每个事件的 CSI 是有限值（干旱事件为 0.0），只有 validation.md 的字符串提取写错了 key。

**建议（待确认后再改）**：`_write_results` 的查询 key 改为 `CSI_10mmh` / `CSI_20mmh`。这不是指标定义问题，是展示层 key 拼写 bug；修复不触及科学定义，但按指令仍先审计不动手。

---

## 5. 可复现性

- 配置：`configs/experiments/E2_resconvlstm.yaml`（冻结）
- 归一化 / 划分：`configs/normalization_v1.json` / `configs/splits_v1.yaml`（冻结）
- 权重：`saved_models/E2_resconvlstm_seed42/E2_resconvlstm_seed42_best.pth`（13.9 MB，`*.pth` 已被 .gitignore 忽略）
- 复现命令：`python scripts/run_experiment.py --config configs/experiments/E2_resconvlstm.yaml --epochs 20 --batch-size 4 --amp on`

---

## 6. 下一步

Backbone Gate 剩余：E0（Persistence，仅 validation）、E1（PlainConvLSTM）、B1（TrajGRU）。见 `scripts/run_backbone_gate.sh`。
