# PRE_EXPERIMENT_FREEZE_AUDIT — 最终预实验数据集加固

> 生成日期：2026-08-14
> 本轮范围：Causal Track Features + Physical Coordinate Consistency + Time-Match Audit + Freeze
> 本轮**未**训练模型、**未**下载 ERA5、**未**重新设计 HDF5 schema（仅作下述必要修正）。

---

## 1. 因果轨迹策略（P0）

- `interpolate_track`（全事件三次样条）**已删除**，改为 `causal_track_at()`。
- 对 anchor_time=t 的样本，**只有 `Time <= t` 的 CMA 定位点可用**：
  - 历史输入时刻被两个都 ≤ t 的定位点夹住 → **线性插值**。
  - 时刻在"最新可用定位点之后、≤ t" → 位置用**常数速度外推**（最近两个定位点）；风速/气压用**持久化**（最新定位点）。
- **绝不用未来 CMA 定位点估计 anchor 位置**；绝不做全事件样条。
- 新增 `/meta/latest_cma_fix_time`、`/meta/cma_fix_age_sec`。
- 关键测试 `test_future_cma_cannot_alter_sample`：篡改 anchor 之后的 CMA 记录 → 样本不变；篡改 anchor 之前的记录 → 样本改变。

---

## 2. GPM 时间匹配策略（P0）

- `lookup_gpm()` 返回 `(path, actual_timestamp, signed_offset_seconds)`，其中 `signed_offset = actual - requested`。
- 存储 `/meta/input_gpm_match_offset [N,11]`、`/meta/target_gpm_match_offset [N]`、`/meta/actual_anchor_gpm_time`、`/meta/actual_target_gpm_time`。
- Validator 报告**真实预报时效** `actual_target - actual_anchor`（应 ≈ 1800s），不再把 ±15min 匹配静默当作精确 +30min。

---

## 3. 网格策略

- **固定锚点网格**：anchor = 最后一个输入帧；所有 11 输入 + 1 target 帧共用一个 128×128 网格；target 不用未来中心。
- **DEM 重采样到 GPM 网格**（rasterio reproject），梯度在重采样后计算。
- `scripts/audit_gpm_grid.py` 预检 GPM 栅格 CRS/transform/分辨率/尺寸/nodata 兼容性，不兼容即 FAIL。
- `--seq-len` **已移除**（option A）：SEQ_LEN=12、INPUT_SEQ_LEN=11 固定，不可与 schema 静默不一致。

---

## 4. 地形梯度单位（P0）

- `dh_dx` / `dh_dy` 单位改为 **m/km**。
- 像素物理尺寸从 anchor 网格 geotransform + anchor 纬度计算（经向/纬向 km/px），不再假设 10km 像素。
- 单位作为 HDF5 根属性 `terrain_gradient_units = "m/km"` 存储。

---

## 5. 最终模型通道 schema（12 通道）

| idx | 名称 | 说明 |
|:---:|------|------|
| 0 | precipitation | GPM 真实降水 (mm/h) |
| 1 | center_wind_speed | CMA 观测最大风速，广播 (m/s) |
| 2 | center_pressure | CMA 观测中心气压，广播 (hPa) |
| 3 | **r_norm** | 归一化到网格中心距离（无量纲） |
| 4 | **dx_norm** | 归一化纬向坐标 ∈ [-1,1] |
| 5 | **dy_norm** | 归一化经向坐标 ∈ [-1,1] |
| 6 | u_move | 台风平移纬向速度 (km/h) |
| 7 | v_move | 台风平移经向速度 (km/h) |
| 8 | dem | 地形高程 (m) |
| 9 | dh_dx | 地形梯度 (m/km) |
| 10 | dh_dy | 地形梯度 (m/km) |
| 11 | land_mask | 海陆掩膜 (0/1) |

> 几何通道已从像素单位改为归一化（`r_norm`/`dx_norm`/`dy_norm`），不再把像素距离称作"km"。

---

## 6. 默认损失（P1）

- **base 默认 = 仅 MSE**（`components: ["rain"]`）。
- 显式配置：`base`(MSE) / `extreme`(MSE+L_extreme) / `smooth`(MSE+L_smooth) / `extreme_smooth`(MSE+L_extreme+L_smooth)。
- Orographic 项保持关闭（需真实环境风场）。
- **不再**把 "MSE + extreme + smooth" 称作 "physics-informed"。

---

## 7. 剩余 SKIP 测试（需服务器依赖）

| 测试 | 覆盖 | 依赖 |
|------|------|------|
| `tests/test_causal_freezing.py` | 因果冻结/梯度单位/归一化几何/签名偏移/MSE默认 | numpy/pandas/h5py/rasterio/affine/torch |
| `tests/test_dataset_v2.py` | DEM 对齐/流式/字段级 strict/零样本 | numpy/h5py/rasterio |
| `tests/test_dataset_safety.py` | 物理损失清理 | torch |

本机已通过：语法检查（13 文件）、纯逻辑泄漏测试（4/4）。

---

## 8. 第一次实验前的精确阻塞项

1. **`python scripts/audit_gpm_grid.py`** → GPM 网格预检 PASS。
2. **`python scripts/audit_data_coverage.py --tolerance 0 --label EXACT`** + **`--tolerance 900 --label T15`** → 两份覆盖报告，据此**设计 event-level split**（不冻结年份假设）。
3. **`python scripts/build_paper_dataset.py`** → 重建因果 v2 HDF5。
4. **`python scripts/validate_paper_dataset.py`** → 必须 PASS（含 forecast-lead 校验）。
5. **`python scripts/fit_normalization.py --train-typhoons <ids>`** → 归一化配置。
6. 服务器 **`pytest tests/`** 全绿。

---

## 9. 冻结声明

本 commit 后**停止数据集重构**，除非某个测试或覆盖报告暴露具体缺陷。
