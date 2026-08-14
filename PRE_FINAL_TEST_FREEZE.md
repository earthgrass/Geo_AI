# PRE_FINAL_TEST_FREEZE — 最终测试前冻结清单

> 生成日期：2026-08-14
> 状态：**TEST_STATUS = SEALED**（正式 test 评估待用户明确批准后单独执行）

## 冻结项

| 项 | 值 |
|----|----|
| 数据集 | `ConvLSTM_Dataset_128.h5`（schema v2，FROZEN，6867 样本） |
| 事件划分 | `configs/splits_v1.yaml`（train 25 / val 7 / test 4，交集为空） |
| 归一化 | `configs/normalization_v1.json`（train-only，冻结） |
| 地形阈值 | `configs/evaluation_thresholds_v1.json`（HIGH_DEM=585m, HIGH_GRAD=16.26 m/km，train-only） |
| 实验矩阵 | E0-E6 + B1（`configs/experiments/`） |
| 主模型 | Terrain-Aware Residual ConvLSTM（ΔP 残差，[64,128]） |
| 强 baseline | TrajGRU（precipitation-only） |
| 损失 | MSE（默认）；E6 加 `λ_extreme·ExtremeMSE`（threshold 10 mm/h） |
| 降雨阈值 | 5 / 10 / 20 / 30 mm/h |
| 评价指标 | MAE / RMSE / SSIM + CSI / POD / FAR（overall + per-event） |
| 种子 | 42（Phase 1/2 第一遍） |

## 最终复现种子（FINAL_REPLICATION_PLAN）
- 42 / 123 / 2026（三个 seed，正式 final test 前冻结核心配置与超参）

## 科研规则（IRON RULE）
- 开发/调参/模型选择阶段：**只准 train + validation**。
- 禁止根据 test 结果调参、选阈值、选模型。
- final test 评估必须显式 `--allow-test-eval`。

## 禁止项（本轮）
- 不重建 HDF5、不改划分、不下载 ERA5、不实现 PINN / q(V·∇h)、不加 Transformer/Diffusion/Attention、不查 test 性能。

---

**TEST_STATUS = SEALED**
