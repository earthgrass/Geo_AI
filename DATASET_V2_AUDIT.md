# DATASET_V2_AUDIT — 地理对齐 + 流式数据集 v2

> 生成日期：2026-08-14
> 本轮范围：Dataset Schema V2 + Geospatial Alignment + Streaming Builder + Data Coverage Audit
> 本轮**未**训练模型、**未**下载 ERA5、**未**构建完整生产 HDF5、**未**宣称任何结果。

---

## 1. 修复的既往问题（P0）

| # | 问题 | 修复 |
|---|------|------|
| 1 | 每帧围绕该帧**真实台风中心**裁剪 → target 用未来位置，引入 oracle 信息 | 改为**固定锚点网格**（见 §2） |
| 2 | DEM 与 GPM 独立各取 128 像素，原生分辨率不同，网格不对齐 | DEM 用 rasterio **重采样到 GPM 网格**（见 §3） |
| 3 | 缺失时对整 12 通道帧插值 | 只对 INPUT 降水插值；TARGET 缺失 → 丢弃样本 |
| 4 | `np.ogrid` 产生 `[1,W]`/`[H,1]` 形状 | 显式 `meshgrid` 广播，dx/dy/distance 均为 `[H,W]` |
| 5 | 标量 track 特征广播成 128×128 存储 | v2 schema 存为 `/track [N,11,F]` 紧凑张量 |
| 6 | `write_hdf5` 一次性堆全部样本进内存 | **流式写入**，增量 flush |

---

## 2. 坐标系统定义（固定锚点网格）

- **anchor_time** = 最后一个输入帧（第 11 帧，索引 10）的时刻。
- **anchor_lat / anchor_lon** = 该时刻台风中心。
- 以 anchor 为中心建立**一个** 128×128 地理网格（GPM 原生 0.1° 像素，即 12.8° 视窗）。
- **全部 11 个输入帧 + 1 个 target 帧**都用这**同一个**网格裁剪。
- **target 绝不用它自己的未来中心定义裁剪** → 无未来轨迹 oracle。

存 `anchor_lat`、`anchor_lon`、`grid_transform`（6 个仿射参数）到 `/meta`。

---

## 3. 地理对齐方法（DEM → GPM）

- 用 GPM 锚点网格作为空间基准（CRS + bounds + transform + 尺寸）。
- `rasterio.warp.reproject`（双线性）把 DEM 重采样到**完全相同**的 CRS/bounds/transform/128×128。
- **不假设** 1 个 DEM 像素 = 10 km。
- 地形梯度在**重采样之后**计算（`np.gradient`）。

---

## 4. HDF5 schema v2

```
/precip/input   [N, 11, H, W]   float32   GPM 真实降水
/precip/target  [N, 1,  H, W]   float32   目标降水（永不被插值）
/terrain        [N, 4,  H, W]   float32   dem, dh_dx, dh_dy, land_mask
/track          [N, 11, 6]      float32   lat, lon, center_wind_speed, center_pressure, u_move, v_move

/meta/typhoon_id [N]   /meta/year [N]
/meta/start_time [N]   /meta/anchor_time [N]   /meta/target_time [N]   (int64 unix sec)
/meta/anchor_lat [N]   /meta/anchor_lon [N]    (float32)
/meta/grid_transform [N,6] (float64)
/meta/input_imputed_mask [N,11] (uint8)
/meta/gpm_match_offset [N,11] (float32, 秒)

根属性: schema_version, channel_names, track_feature_names,
        terrain_channel_names, seq_len, input_seq_len, grid_size,
        precipitation_units, created_by
```

`TyphoonDataset` **动态重建** 12 通道模型输入：
`precip` ← `/precip/input`；`center_wind_speed/center_pressure/u_move/v_move` ← `/track` 广播；`distance_center/dx/dy` ← 静态网格；`dem/dh_dx/dh_dy/land_mask` ← `/terrain`。

---

## 5. 缺失值策略

- **降水与 track/terrain 分离**：绝不插值整帧。
- **仅 INPUT 降水可时序插值**（时间加权线性，`max_missing` 上限）。
- **TARGET 永不被插值**：目标 GPM 缺失 → **丢弃样本**。
- 记录 `/meta/input_imputed_mask [N,11]` + `/meta/gpm_match_offset [N,11]`（秒）。

---

## 6. 数据来源与存储

- CMA 最佳路径（`CMABSTdata/`）→ 三次样条 0.5h 插值 + 平移速度。
- GPM IMERG（`TIFdata/`）→ 预索引 `typhoon_id → {时间戳 → 路径}`，精确优先、近邻回退（记录偏移、容忍重复使用）。
- ETOPO1 DEM（`Global_DEM.tif`）→ 重采样到 GPM 网格。
- 存储：可扩展 HDF5 数据集 + gzip，`--buffer-size` 增量 flush，**永不整库进 RAM**。

---

## 7. 归一化策略

- `scripts/fit_normalization.py` 用 **train events only** 拟合统计量 → `configs/normalization_v1.json`。
- 降水：min-max（vmin=0，vmax=训练集最大）。
- track 特征 + 地形：z-score（逐特征 mean/std）。
- `land_mask`：**不归一化**（0/1）。
- `log1p` 降水：**默认关闭**（未来实验选项）。
- train/val/test 复用 train 统计量（冻结）。

---

## 8. 增广安全

- `RandomRotation` 对矢量通道（u/v、dh_dx/dh_dy、dx/dy）不安全。
- 默认**禁用**：实例化时发 `RuntimeWarning`，未实现矢量感知旋转。

---

## 9. 测试

| 测试 | 覆盖 | 本机状态 |
|------|------|:---:|
| `tests/test_splits.py` | 事件级泄漏纯逻辑 | ✅ 4/4 PASS |
| `tests/test_dataset_v2.py` | DEM 对齐/锚点网格/插值/流式/字段级 strict/零样本 | ⏳ 需服务器（numpy/h5py/rasterio） |
| `tests/test_dataset_safety.py` | 物理损失清理 + 元数据过滤 | ⏳ 需服务器（torch） |
| 语法检查 `py_compile`（10 文件） | — | ✅ PASS |

`test_dataset_v2.py` 用两个**不同原生分辨率**的合成栅格证明：DEM 重采样到 GPM 网格、锚点网格一致、target 缺失丢弃、input 可插值并标记、land_mask 保持二值、dx/dy 为 `[H,W]`、流式写入不累积、字段级 strict 独立失效、零样本校验干净 FAIL。

---

## 10. 第一次训练前必须完成

1. `python scripts/audit_data_coverage.py` → 出 `DATA_COVERAGE.csv/.md`，据此设计 event-level split。
2. `python scripts/build_paper_dataset.py` → 重建 v2 HDF5。
3. `python scripts/validate_paper_dataset.py` → 必须 PASS。
4. `python scripts/fit_normalization.py --train-typhoons <ids>` → 归一化配置。
5. 服务器跑 `pytest tests/`。

---

*本轮目标（Schema V2 + Geospatial Alignment + Streaming + Coverage Audit）已达成。*
