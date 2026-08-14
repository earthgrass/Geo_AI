# NORMALIZATION_AUDIT — 归一化审计

> 生成日期：2026-08-14
> 输出：`configs/normalization_v1.json`

## 1. 拟合方式

- **仅用 train 事件**（25 个台风，4894 样本），val/test 不参与。
- 降水统计同时覆盖 **input + target** 降水。

## 2. 统计结果

| 字段 | 统计 |
|------|------|
| 降水 max / p99.9 | 100.97 / 27.12 mm/h |
| center_wind_speed | mean 24.13, std 11.87 |
| center_pressure | mean 986.52, std 20.27 |
| u_move | mean -6.76, std 18.15 |
| v_move | mean 9.56, std 11.64 |
| dem | mean 50.58, std 205.39 |
| dh_dx | mean -0.07, std 4.57 |
| dh_dy | mean -0.08, std 3.59 |
| land_mask | **不归一化**（0/1） |
| log1p 降水 | **默认关闭** |

## 3. 关键校验

- 训练样本数 4894 == smoke test 的 train split 样本数 ✓
- u_move std=18.15（修复日界线速度 bug 前为 1079，现合理）
- 归一化统计只用 train，val/test 复用（冻结）

## 4. 结论

归一化配置已就绪，可用于所有实验（train/val/test 统一用 train 统计量）。
