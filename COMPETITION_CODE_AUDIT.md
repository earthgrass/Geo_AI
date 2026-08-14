# COMPETITION_CODE_AUDIT — 比赛代码完整审查

> 生成日期：2026-08-14
> 审查对象：`比赛代码/`（20 个 .py + 3 个 .pyc，2026 数学建模竞赛原始代码）
> 目标：Legacy Code Audit → Provenance Mapping → Reproducible Pipeline Preparation

---

## 1. 文件清单与分类

| 文件 | 作用 | 输入 → 输出 | Q1/Q2/Q3 | 处置 |
|------|------|------|:---:|:---:|
| `step1.1 data process.py` | CMA 轨迹解析 + 三次样条 0.5h + 运动学特征 | CMABSTdata/*.txt → All_Years_Typhoon_Features.csv | Q1→Q2 | **MUST MIGRATE** |
| `step1.2 tif process.py` | GPM TIF **标量**特征(面积/峰值/质心) | TIFdata → TIF_Features_Base.csv | Q1 | DELETE |
| `step1.3 data integration.py` | inner merge + D_offset/I_asy + 留白列 | 两个CSV → Typhoon_Full_Dataset_Q1.csv | Q1 | DELETE(对齐逻辑可参考) |
| `step1.4 data analysis.py` | Spearman + RF + SHAP | Q1 CSV → 矩阵/重要性 | Q1 | DELETE |
| `step1.5 visualization.py` | Q1 可视化 | 分析结果 → Figures1 | Q1 | DELETE |
| `data analysis.py` | step1.4 的重复副本 | 同 | Q1 | DELETE |
| `convLSTM_model.py` | 旧模型架构 SpatialResidualConvLSTM | — | Q2 | LEGACY |
| `step2.1_spatial_dataloader.py` | HDF5 张量构建(裁剪/插值/滑窗/通道) | Q1 CSV + TIFdata → ConvLSTM_Dataset_128.h5 | Q2 | **MUST MIGRATE**(需修) |
| `step2_2_train_cloud.py` | 模型训练 | HDF5 → typhoon_convlstm_best.pth | Q2 | LEGACY / REFERENCE ONLY |
| `step2_3_generate_metrix.py` | 物理推演引擎 + DEM 读取 | 轨迹+DEM+权重 → DataPackage.npz | Q2/Q3 | DELETE(仅 RealGeographyEngine 可复用) |
| `step2_3_rainfall_heatmap.py` | 旧热力图 | npz → png | Q2 | DELETE |
| `step2_4_rainfall_heatmap.py` | cartopy 降水热力图 | npz → png | Q2 | LEGACY |
| `step_3_3_visualision.py` | 可视化(文档曾引用缺失) | — | Q3 | DELETE |
| `step3_1_virtural_generator.py` | 虚拟台风生成 | CH2024BST → Virtual_Typhoons_2026.txt | Q3 | DELETE |
| `step3_2_metrics_table.py` | 情景量化指标 | npz → Sensitivity_Analysis_Results.csv | Q3 | DELETE |
| `step3_4_china_rainfall.py` | 中国沿海降水 | npz → png | Q3 | DELETE |
| `step4_1_virtual_spatiotemporal.py` | 虚拟台风时空演变 | npz → png | Q3 | DELETE |
| `step4_2_china_landfall_map.py` | 登陆风险图 | npz → png | Q3 | DELETE |
| `step4_2_china_rainfall_map.py` | 沿海降水图 | npz → png | Q3 | DELETE |
| `step4_3_sensitivity_impact.py` | 单因素敏感性遍历 | 轨迹+权重 → Fig4_6 | Q3 | DELETE |

---

## 2. 原始 Pipeline

```
step1.1 (CMA 轨迹→0.5h 特征)  ┐
step1.2 (GPM TIF→标量特征)    ├→ step1.3 (inner merge + D_offset/I_asy) → step1.4 (Spearman/RF/SHAP) → step1.5 (Q1 图)
                              │
                              └→ step2.1 (HDF5 4通道张量) → step2.2 (训练 SpatialResidualConvLSTM)
                                                                    ↓
                                    step2.3 (物理推演引擎, 读 DEM) → step2.4 (热力图)
                                                                    ↓
step3.1 (虚拟台风) → step2.3 复用 → step3.2 (指标) → step3.4/step4.x (情景图 + 敏感性)
```

---

## 3. 数据构建流程（重点 · step2.1）

| 项 | 事实 | 证据 |
|----|------|------|
| 输入 | `Typhoon_Full_Dataset_Q1.csv`（step1.3 产物） | step2.1:215 |
| 输出 | `ConvLSTM_Dataset_128.h5` | step2.1:217 |
| Tensor shape | `[N, 12, 4, 128, 128]`（12=11输入+1目标） | step2.1:111 |
| **Channel 顺序** | **`[0]=降水(GPM真实场), [1]=风场, [2]=气压场, [3]=距中心距离`** | step2.1:155-156 |
| 风/气压场性质 | **参数化合成场**（非真实观测） | step2.1:135-136 |
| **DEM 是否进数据** | **否。4 通道无 DEM/地形梯度/海陆掩膜** | step2.1:111 |
| 时间分辨率 | 0.5h（CMA 6h 三次样条插值而来） | step1.1:88 |
| 空间分辨率 | 128×128 ≈ 10km/px | step2.1:27 |
| 滑动窗口 | stride=1，台风内重叠窗口 | step2.1:144 |
| missing value | 时间加权线性插值，max_missing=2 | step2.1:76-96 |
| label/target | `sample[-1, 0]`（最后一帧降水场） | step2.1:206 |
| **台风 ID/年份进样本** | **否。HDF5 无 `/meta` 组，只存 `data`** | step2.1:109-115 |
| GPM 与 CMA 匹配 | step1.3 按 `[Typhoon_ID, Time]` inner merge | step1.3:83-89 |

---

## 4. 模型训练流程（step2.2）

| 项 | 事实 |
|----|------|
| 模型 | `SpatialResidualConvLSTM`（内部残差 `relu(p_pred+Δp)`） |
| loss | **纯 `nn.MSELoss()`**（无任何物理项） |
| split | **`random_split(dataset, [0.8, 0.2])` — 随机划分** |
| optimizer / lr | AdamW / 1e-3 |
| batch / epochs | 8 / 20 |
| normalization | **无** |
| seed | **未设置（不可复现）** |
| AMP | ✅ |
| 保存 | `typhoon_convlstm_best.pth`（裸 state_dict） |

---

## 5. 数据泄漏检查（最高优先级）⚠️

**结论：DATA LEAKAGE — NOT PAPER USABLE**

证据链：
1. `step2.1` 以 `stride=1` 生成**重叠窗口**，且同一台风按 `Typhoon_ID` 分组（step2.1:117）。
2. `step2.2:139-141` 用 `random_split(dataset, [0.8, 0.2])` 做划分，**不区分台风、不区分年份**。
3. 因此同一台风的**相邻时间窗**（甚至同一时刻的前后重叠帧）会同时落入 train 和 val。

**后果**：旧版 `best_val_loss = 1.44` 等指标**虚高**，不能作为论文证据。

**新论文必须**：leave-one-typhoon-out 或 event-level 划分，保证同一台风不在 train/test 同时出现。

---

## 6. 物理机制真实性检查 ⚠️

**结论：旧"物理"是推理阶段的后处理，不是训练约束；且几乎全是硬编码经验公式。**

1. **训练阶段完全没有物理**：`step2_2` 只用 `nn.MSELoss()`。
2. **DEM 只在推理进入**：`step2_3` 的 `RealGeographyEngine` 读取 DEM，但只用于生成 `phys_rain` 基准场，**从未作为模型输入通道参与训练**。
3. **训推不一致**：模型训练时通道 0 是**真实 GPM 降水**；推理时通道 0 被替换为 `phys_rain/10.0`（step2.3:134）。
4. 硬编码公式清单（均无文献）：
   - `base_rain = (wind/5.0)^1.2`（step2.3:74）
   - `orographic = 1 + (dem/1000)*0.35`（step2.3:66）
   - `friction = 1.2`(陆)/`1.0`(海)（step2.3:67）
   - `final = base_rain + nn_residual*1.5`（step2.3:146）
5. **两个 Rmax 公式冲突**：step1.1 用 Willoughby `46.4*exp(-0.0155V+0.0169|lat|)`；step2.3 用 `(66.785-0.09102V_knots+0.0105|lat|)*1.852`。

**唯一可保留的物理信息**：地形抬升**概念**、科氏参数定义 `f=2Ω sinφ`。

---

## 7. 可复用模块（MUST MIGRATE / LEGACY）

### A. MUST MIGRATE（论文复现必须重构）
1. **`step1.1 data process.py`** — CMA 解析 + 三次样条 + 运动学特征（Lat/Lon/Pressure/Wind/Distance/Speed/Direction/Curvature/Rmax）。逻辑正确，仅路径硬编码需改。
2. **`step2.1_spatial_dataloader.py`** — TIF 裁剪、时间加权插值、滑动窗口、通道拼接。**需三处修正**：① 加入 DEM/地形梯度通道；② 写 `/meta` 组存台风 ID+年份；③ 移除参数化风/气压场的合成假设（改用真实数据或明确标注）。
3. **`step2_3` 的 `RealGeographyEngine.get_real_dem_and_mask`** — DEM 的 rasterio 裁剪/双线性重采样，可直接复用于构建 DEM 通道。

### B. LEGACY ONLY（历史追溯，禁作主实现）
- `convLSTM_model.py` — 旧架构参照。
- `step2_2_train_cloud.py` — **旧训练脚本，仅作参考**。含 `random_split` 泄漏、旧 `SpatialResidualConvLSTM` 架构、纯 MSE、无 seed、旧训练语义；其有用的训练思想已被 `src/training/trainer.py` 取代，禁止作为论文主实现。
- `step2_4_rainfall_heatmap.py` — cartopy 配色/底图参照。

### C. DELETE FROM PAPER PIPELINE（彻底弃用）
- Q1 全部：`step1.2`、`step1.3`(对齐逻辑除外)、`step1.4`、`step1.5`、`data analysis.py`。
- Q3 全部：`step3.1`、`step3.2`、`step3.4`、`step4.1`、`step4.2 ×2`、`step4.3`、`step_3_3`。
- `step2_3` 的 `PhysicsInformedEngine` 与 `PINN_InferenceEngine`（硬编码物理）。
- `step2_3_rainfall_heatmap.py`。

> 注：本轮**不删除**任何原始文件，仅做分类标记。

---

## 8. 数据重建方案（HDF5 能否完整重建）

**结论：可以完整重建，且这是论文实验的前置第一步。**

- 需要的输入**全部在仓库**：`CMABSTdata/`(12 年轨迹)、`TIFdata/`(33,420 TIF)、`Global_DEM.tif`(465MB)。
- 重建脚本 = `step2.1` 逻辑 + 三处修正（见 §7-A-2）。
- **关键新增**：重建时必须写入 `/meta/year` 和 `/meta/typhoon_id`，否则新版 `src/data/dataset.py` 的年份划分无法工作（当前只能靠 sidecar CSV）。

---

## 9. 对当前论文计划的影响

1. **数据泄漏结论坐实**：`RESEARCH_AUDIT.md` 的"旧版随机 80/20 泄漏"已从源码证实。
2. **DEM 从未进模型**：坐实。论文要谈"terrain-aware physics guidance"，必须先**把 DEM/地形梯度真正加入输入通道**（当前 4 通道没有）。
3. **旧"物理"不可迁移**：坐实。论文的"物理约束"必须重新设计为**训练损失项**，而非推理后处理。
4. **旧 `.pth` 不可用**：坐实。它对应 `SpatialResidualConvLSTM`（内部残差），无法加载进新 `PIResConvLSTM`（时序残差）。
5. **无任何可复用 evaluation / baseline**：旧版只有 MSE + step3.2 粗指标；新论文的 baseline 与气象指标全部要新跑。
6. **数据工程可以救**：step1.1 + step2.1 的 TIF 裁剪/插值/滑窗逻辑是可复用的真实资产，重建 HDF5 的工作量可控。

---

*本文档与 `CODE_PROVENANCE_MAP.md`、`RESEARCH_AUDIT.md`、`PAPER_PLAN.md` 配套。所有结论均可回溯到 `比赛代码/` 与 `src/` 的具体行号。*
