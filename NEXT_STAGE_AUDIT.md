# NEXT_STAGE_AUDIT — 可复现数据集管线 + 泄漏安全 + 损失清理

> 生成日期：2026-08-14
> 本轮范围：Reproducible Dataset Pipeline + Leakage Safety + Loss Cleanup
> 本轮**未**训练模型、**未**写论文正文、**未**实现 ERA5 摄取。

---

## 1. 变更文件清单

| 文件 | 操作 | 内容 |
|------|:---:|------|
| `src/config.py` | 修改 | 16 通道 → 12 通道 `CHANNEL_NAMES`（唯一权威来源） |
| `src/data/splits.py` | **新增** | 事件级泄漏安全纯逻辑（无第三方依赖） |
| `src/data/dataset.py` | 修改 | `strict_metadata=True` + 元数据校验，缺失元数据不再静默回退 |
| `src/training/physics_loss.py` | 修改 | 删 L_nonneg、L_rain 改标准 MSE、L_oro 配置驱动 |
| `src/training/trainer.py` | 修改 | 移除硬编码 ORO 通道索引、`_build_physics_aux` 配置驱动 |
| `configs/default.yaml` | 修改 | 12 通道 + 新物理损失结构 |
| `src/training/configs/default.yaml` | 修改 | 与根 config 同步 |
| `scripts/build_paper_dataset.py` | **新增** | 12 通道数据集构建器（重构旧 step1.1/2.1/2.3 的有效逻辑） |
| `scripts/validate_paper_dataset.py` | **新增** | 数据集 PASS/FAIL 校验脚本 |
| `README.md` | 修改 | 科学诚实性：Implemented / Pending 分离，竞赛数值标记 legacy |
| `COMPETITION_CODE_AUDIT.md` | 修改 | step2_2 分类修正为 LEGACY/REFERENCE ONLY |
| `tests/test_splits.py` | **新增** | 纯逻辑泄漏安全测试（4/4 通过） |
| `tests/test_dataset_safety.py` | **新增** | 合成 HDF5 fixture 冒烟测试（需 torch/numpy/h5py） |

---

## 2. 最终 12 通道 schema（`src/config.py` 权威定义）

| idx | 名称 | 来源 | 说明 |
|:---:|------|------|------|
| 0 | precipitation | GPM IMERG | 真实降水 (mm/h) |
| 1 | center_wind_speed | CMA 观测 | 最大风速，空间广播 (m/s) |
| 2 | center_pressure | CMA 观测 | 中心气压，空间广播 (hPa) |
| 3 | distance_center | 计算 | 距台风中心距离 (km) |
| 4 | dx | 计算 | 纬向相对坐标 (km) |
| 5 | dy | 计算 | 经向相对坐标 (km) |
| 6 | u_move | CMA 轨迹差分 | **台风平移**纬向速度 (km/h) |
| 7 | v_move | CMA 轨迹差分 | **台风平移**经向速度 (km/h) |
| 8 | dem | ETOPO1 | 地形高程 (m) |
| 9 | dh_dx | DEM 梯度 | 地形梯度 d(elev)/dx (m/km) |
| 10 | dh_dy | DEM 梯度 | 地形梯度 d(elev)/dy (m/km) |
| 11 | land_mask | DEM | 海陆掩膜 (0=海, 1=陆) |

> 旧版的**参数化合成风场/气压场已移除**：`center_wind_speed`/`center_pressure` 来自 CMA 观测，不再用 `render_wind_field`/`render_pressure_field` 合成。

---

## 3. 泄漏安全措施

1. **`strict_metadata=True`（默认）**：请求 `split_years`/`typhoon_ids` 但元数据缺失 → 抛 `RuntimeError`（不再静默回退全部样本）。
2. **元数据校验**：长度 == 样本数、`sample_idx` 唯一、年份为合法整数、`typhoon_id` 无缺失。
3. **`assert_disjoint_event_split()`**：train/val/test 台风 ID 两两交集必须为空，否则抛 `RuntimeError`（不打印警告）。
4. **构建器保证每个滑动窗口只属于一个台风**：按 `Typhoon_ID` 分组后再滑窗。
5. **HDF5 写入 `/meta/typhoon_id` + `/meta/year`**，年份划分不再依赖 sidecar CSV 猜测。

---

## 4. 物理损失变更

**新损失**：
```
L_total = L_rain + λ_smooth·L_smooth + λ_extreme·L_extreme  (+ λ_oro·L_oro，opt-in)
```

| 变更 | 说明 |
|------|------|
| **删 L_nonneg** | 输出已 `P_hat=ReLU(P_t+ΔP)≥0`，`ReLU(-P_hat)²` 恒为 0，无信号 |
| **L_rain = 标准 MSE** | 去掉 `heavy_rain_alpha` 加权，极端权重只归 `L_extreme`，消融更干净 |
| **L_extreme 独立** | 只对 `P_true > threshold` 的像素算 MSE |
| **L_oro 默认禁用** | 需显式配置真实环境风场通道；`enabled=true` 但 `u_channel/v_channel=null` → 抛 `RuntimeError` |

**通道索引来源**：由 config 的 `physics_loss.orographic.{u_channel,v_channel,dh_dx_channel,dh_dy_channel}` 提供，**不再硬编码** 6/7/9/10。

---

## 5. 为什么 u_move / v_move 不能用于 L_oro

- `u_move`/`v_move` 是**台风中心的平移速度**（轨迹差分），描述的是"整个风暴系统往哪走"，量级 ~ 几十 km/h，是**标量广播场**。
- 地形抬升约束 `O = u·∇h` 需要的是**环境大气风场**（低层水平风，随空间变化的矢量场），它驱动气块沿坡抬升。
- 两者物理含义完全不同：用平移速度当环境风，等于把"台风走路"错当成"空气爬坡"，是**物理错误**。

**结论**：当前 12 通道 schema 里**没有**环境风场数据。要启用 L_oro，必须先引入真实大气风场（例如 ERA5 的 u/v 分量），并显式配置 `u_channel`/`v_channel`。在此之前 L_oro 保持禁用。

---

## 6. 第一次训练前必须完成

1. **重建数据集**（服务器）：`python scripts/build_paper_dataset.py --cma-dir CMABSTdata --tif-dir TIFdata --dem Global_DEM.tif --out ConvLSTM_Dataset_128.h5`
2. **校验数据集**：`python scripts/validate_paper_dataset.py --h5 ConvLSTM_Dataset_128.h5`（必须 PASS）
3. **跑冒烟测试**（装依赖后）：`python -m pytest tests/ -v`
4. **决定是否引入环境风场**：若要在论文里主张"地形感知物理引导"，需要 ERA5 u/v 风场 + 显式配置 L_oro；否则论文的 physics 主张只能停在"极端加权 + 平滑"层面。

---

## 7. 测试结果

| 检查 | 结果 |
|------|------|
| 语法检查 `py_compile`（9 文件） | ✅ PASS |
| 纯逻辑泄漏安全测试 `tests/test_splits.py` | ✅ 4/4 PASS |
| 完整冒烟测试 `tests/test_dataset_safety.py` | ⏳ SKIP（本机无 torch/numpy/h5py；需在服务器跑） |
| 静态检查：`nonneg`/`heavy_rain_alpha`/硬编码通道索引 | ✅ 已清除（仅 physics_loss.py 文档字符串保留"为何移除"的历史注释） |

---

*本轮目标（Legacy Audit → Provenance → Reproducible Pipeline → Push）已达成。禁止在本轮宣称任何论文结果。*
