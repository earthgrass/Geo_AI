# FINAL_REPORT — 数据管线冻结与实验就绪最终报告

> 生成日期：2026-08-14
> 提交：`376b14e`（push 成功）
> 范围：Final Data Validation → Coverage → Event Split → Production Build → Normalization → Experiment-Ready Freeze

---

## 0. 一句话结论

**数据集管线已冻结、生产数据集已构建并验证通过、归一化与实验配置已就绪 —— 项目进入 RESEARCH_READY 状态，可进入文献调研与模型训练阶段。**

---

## 1. 最终结果汇总

| # | 项目 | 结果 |
|---|------|------|
| 1 | commit SHA | `376b14e` |
| 2 | push 状态 | ✅ 成功 |
| 3 | pytest | ✅ **27/27 passed** |
| 4 | GPM 网格预检 | ✅ PASS（每文件是台风中心瓦片，逐文件 reproject 处理） |
| 5 | EXACT 覆盖 | 302 台风，精确匹配 7318，36 个事件有可用数据 |
| 6 | T15 覆盖 | 与 EXACT 完全一致（nearest=0） |
| 7 | GPM 匹配策略 | **EXACT**（tolerance=0） |
| 8 | train/val/test 事件数 | 25 / 7 / 4 |
| 9 | train/val/test 样本数 | 4894 / 1266 / 707（合计 6867） |
| 10 | 生产 HDF5 样本数 | **6867**（1.2GB，schema v2，12 通道） |
| 11 | 生产 HDF5 验证 | ✅ **PASS（67 项）** |
| 12 | 归一化 | ✅ `configs/normalization_v1.json`（train-only） |
| 13 | 模型前向冒烟 | ✅ **PASS** |
| 14 | 最终分类 | ✅ **RESEARCH_READY** |
| 15 | 下一步 | 文献调研（ERA5、实验矩阵、research gap） |

---

## 2. 生产数据集规格

- **文件**：`ConvLSTM_Dataset_128.h5`（schema v2）
- **通道 schema（12）**：precipitation / center_wind_speed / center_pressure / r_norm / dx_norm / dy_norm / u_move / v_move / dem / dh_dx / dh_dy / land_mask
- **因果性**：轨迹特征只用 anchor 时刻之前的 CMA 定位点（无未来泄漏）
- **几何**：固定锚点网格，DEM 精确重采样到 GPM 网格，梯度单位 m/km
- **统计**：降水 max=100.97 / p99.9=26.5；气压 915–1008 hPa；风速 10–62 m/s；forecast lead 恒 1800s；插值率 0.03%

---

## 3. 本轮修复的 bug（6 个，均自主诊断）

1. **pandas 3.0 `datetime64[us]`**：`.astype("int64") // 10**9` 得到错误秒数（差 1000 倍），轨迹纬度一度达 1e7。
2. **anchor_time off-by-one**：误用 target 帧作锚点，导致轨迹与降水错位一帧。
3. **CMA 经度 0–360° 未归一化**：需转 -180..180（EPSG:4326）。
4. **经度外推跨 180° 日界线未回绕**：产生 507° 经度、u_move 达 72000 km/h。
5. **验证器 terrain 索引 off-by-one × 2 + forecast-lead 语义**。
6. **GPM TIF 逐文件瓦片**：非单一全局网格，需逐文件 `rasterio.warp.reproject` + DEM 单次打开（否则构建需数小时）。

---

## 4. 关键决策记录

| 决策 | 结论 |
|------|------|
| GPM 时间戳匹配 | EXACT（无系统偏移，近邻匹配无收益） |
| 事件划分 | 时间外推：train 2014-2021 / val 2022 / test 2023-2024 |
| 归一化 | train-only 拟合，降水 min-max，track/terrain z-score，land_mask 不归一化 |
| 默认损失 | 仅 MSE（extreme/smooth 显式 opt-in，orographic 关闭） |
| 增广 | RandomRotation 禁用（矢量通道不安全） |

---

## 5. 生成的审计/配置文档

- `RESEARCH_READY_AUDIT.md`、`PRODUCTION_DATASET_VALIDATION.md`、`NORMALIZATION_AUDIT.md`
- `SPLIT_AUDIT.md`、`COVERAGE_DECISION.md`、`GPM_GRID_AUDIT_RESULT.md`
- `DATA_COVERAGE_EXACT.csv/.md`、`DATA_COVERAGE_T15.csv/.md`
- `configs/splits_v1.yaml`、`configs/normalization_v1.json`、`configs/experiments/01-06.yaml`

---

## 6. 重要边界（诚实声明）

- **未训练任何神经网络模型**、**未下载 ERA5**、**未宣称任何研究结论**。
- 覆盖率低（36/302 台风）的根因是 **TIFdata 只预处理了 36 个台风**（数据可得性），非代码缺陷。
- 1 个已知边缘样本（tid=2306 跨日界线，全零降水）的 `actual_target_gpm_time` 有偏移，但 forecast lead（请求时间）恒 1800s，无实际影响。

---

## 7. 下一步（尚未执行）

进入**文献调研**阶段，需决定：
1. 是否引入 ERA5 环境风场（若需启用 orographic 物理约束）；
2. 哪个气压层最可辩护；
3. `q·(V·∇h)` 地形抬升项是否有依据；
4. 最终实验矩阵（E0-E5 是否调整）；
5. 最终 research gap 陈述。

在此之前，**停止数据集重构**，除非某个测试或覆盖报告暴露具体缺陷。
