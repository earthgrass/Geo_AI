# CODE_PROVENANCE_MAP — 比赛代码 → 论文新代码 来源关系对照

> 生成日期：2026-08-14
> 对照对象：`比赛代码/`（20 个 .py，2026 竞赛原始代码） vs `src/`（重构论文代码）

---

## 0. 一句话结论

新 `src/` **不是**旧竞赛代码的等价重写：**模型架构语义、损失函数、数据划分、推理机制、评估指标全部发生了实质变化**。旧代码唯一能"原样迁移"的只有 CMA 轨迹解析、TIF 裁剪、滑动窗口这三段**数据工程逻辑**；其余要么是全新编写，要么在旧代码中根本不存在（baseline / attention / 物理损失 / 气象指标）。

---

## 1. 逐模块对照

| 论文新模块 | 来源旧代码 | 等价性 | 发生了什么变化 | 风险 |
|-----------|-----------|:---:|--------------|------|
| `src/data/dataset.py` | `step2.1_spatial_dataloader.py` | **部分等价** | 新增：按年份/台风 split、`/meta` 读取、返回 `P_prev`、sidecar CSV 兜底；旧版只读 `data` 张量、无 meta | 旧 HDF5 无 `/meta` 组，年份/台风 ID 只能靠 sidecar `ConvLSTM_Dataset_128_metadata.csv` 重建 |
| `src/data/transforms.py` | 无 | **全新** | MinMaxNormalize / LogTransform / RandomRotation；旧版训练时**无任何归一化** | 新归一化会改变 loss 量纲，与旧 `.pth` 不可比 |
| `src/models/convlstm_cell.py` | `convLSTM_model.py:ConvLSTMCell` | **等价** | 门控卷积逻辑一致；新增 `layer_norm`(GroupNorm) 开关 | 低 |
| `src/models/pi_res_convlstm.py` | `convLSTM_model.py:SpatialResidualConvLSTM` | **不等价** | 旧：内部残差 `relu(p_pred + Δp)`（预测绝对值，再自修正）；新：时序残差 `P_hat = ReLU(P_t + ΔP)`（预测变化量）；新增加 `pred_head`+`refine_net` 双路径、可选 SE attention | **旧 `typhoon_convlstm_best.pth` 无法加载进新模型**（语义与结构都不同） |
| `src/models/baselines.py` | 无 | **全新** | Persistence / PlainConvLSTM / ResConvLSTM；旧版**无任何 baseline** | 论文必需，但需从头训练 |
| `src/models/channel_attention.py` | 无 | **全新** | SE Block；旧版无 attention | 可选消融模块 |
| `src/training/physics_loss.py` | `step2_3_generate_metrix.py` 的 `PhysicsInformedEngine` | **不等价** | 旧：推理阶段硬编码公式 `(wind/5)^1.2 × coriolis × orographic(0.35) × friction(1.2)`；新：训练 loss 项（加权 MSE + 非负 + 地形相关 + 平滑 + 极端）。**两者机制完全不同** | 旧的"物理"是**推理后处理**，不是训练约束；禁止把旧公式包装成论文创新 |
| `src/training/trainer.py` | `step2_2_train_cloud.py` | **部分等价** | 新增：seed、早停、LR 调度、梯度裁剪、物理损失、年份 split；旧版：`random_split` 80/20 + 纯 MSE + 无 seed | **旧版随机划分已确认数据泄漏**；新版年份划分修复了它，但**从未运行验证** |
| `src/inference/infer.py` | `step2_3_generate_metrix.py` 的 `PINN_InferenceEngine` | **不等价** | 旧：`final = base_rain + nn_residual × 1.5`（物理基准 + 硬编码缩放）；新：`P_hat = ReLU(P_t + ΔP)` 纯模型输出 | 训推一致性问题已在新版修复 |
| `src/evaluation/metrics.py` | 无 | **全新** | MAE/RMSE/SSIM/CSI/POD/FAR/HSS/峰值/面积/中心偏移；旧版仅 MSE + `step3_2` 的粗指标(P_max/S_ext) | 论文核心资产，但**尚未跑出任何数字** |
| `src/visualization/plot.py` | `step2_4_rainfall_heatmap.py` / `step4_x` | **部分等价** | cartopy 绘图逻辑可参考旧版配色与底图 | 低 |

---

## 2. 数据层来源（详）

| 环节 | 旧脚本 | 可迁移性 |
|------|--------|:---:|
| CMA 轨迹解析 + 三次样条 0.5h | `step1.1 data process.py` | ✅ MUST MIGRATE（逻辑正确，仅路径硬编码需改） |
| GPM TIF **标量**特征(P_max/面积/质心) | `step1.2 tif process.py` | ⚠️ 仅 Q1 用；nowcasting 需要的是**2D 场**，2D 场裁剪在 step2.1 |
| 时空对齐(inner merge) + D_offset/I_asy | `step1.3 data integration.py` | ⚠️ 对齐逻辑可复用；D_offset/I_asy 属 Q1 特征，论文可删 |
| HDF5 构建(裁剪/插值/滑窗/通道拼接) | `step2.1_spatial_dataloader.py` | ✅ MUST MIGRATE（但需加 DEM 通道 + /meta 组） |
| DEM 读取(rasterio 裁剪重采样) | `step2_3` 的 `RealGeographyEngine` | ✅ 该**类**可复用；其余硬编码物理引擎删除 |

---

## 3. 关键架构差异（旧 vs 新模型）

| 维度 | 旧 `SpatialResidualConvLSTM` | 新 `PIResConvLSTM` |
|------|------|------|
| 残差形式 | **内部残差** `relu(p_pred + Δp)` | **时序残差** `ReLU(P_t + ΔP)` |
| 输出语义 | 绝对降水场 | 降水变化量 ΔP |
| ConvLSTM 层 | 2 层 [64, 128] | 2 层 [64, 128]（同） |
| SE attention | 无 | 可选 |
| decoder | Conv→BN→ReLU→pred_head | 同 + refine_net 双路径 |
| input_channels | 4 | 可配（默认 4） |
| 参数量 | ~5M 量级 | 同量级（+attention 略增） |
| 输出 ReLU | ✅ | ✅（`compute_prediction` 内） |

---

## 4. 旧训练 vs 新训练

| 维度 | 旧 `step2_2_train_cloud.py` | 新 `trainer.py` |
|------|------|------|
| optimizer | AdamW lr=1e-3 | AdamW lr=1e-4（默认） |
| loss | **纯 MSE**（无物理） | 物理损失（可开关） |
| batch / epochs | 8 / 20 | 8 / 100 |
| normalization | **无** | MinMax (0,100) |
| split | **`random_split` 80/20（泄漏）** | 年份划分 2014-22/2023/2024 |
| seed | **无** | `set_seed(42)` |
| early stopping / LR 调度 / grad clip | 无 | 有 |
| checkpoint | `state_dict()` 裸存 | 含 config 的完整字典 |

---

## 5. 旧"物理"到底是什么（拆解）

`step2_3_generate_metrix.py` 的物理引擎，按性质分类：

| 项 | 公式 | 性质 | 能否进论文 |
|----|------|------|:---:|
| 科氏参数 | `f = 2Ω sin φ` | ✅ 正确物理定义 | 可（作为地形/纬度先验） |
| 基础降水 | `base_rain = (wind/5)^1.2` | ❌ 经验幂律，无文献 | 禁止 |
| 地形抬升 | `1 + (dem/1000)*0.35` | ⚠️ 概念真实，系数 0.35 任意 | 概念可迁移，系数需论证 |
| 陆地摩擦 | `×1.2`(陆)/`×1.0`(海) | ❌ 硬编码 | 禁止 |
| 科氏非对称 | `1 + (f*1e4)*cos(θ)` | ❌ 无物理量纲论证 | 禁止 |
| NN 残差缩放 | `nn_residual × 1.5` | ❌ 硬编码超参 | 禁止 |

**结论**：旧代码里**唯一可迁移的物理信息是"地形抬升对降水的增强概念"和"科氏参数定义"**；其余全是竞赛情景推演的硬编码，**禁止包装成论文创新**。

---

*本文档与 `COMPETITION_CODE_AUDIT.md` 配套；生成依据为 `比赛代码/` 与 `src/` 的实际源码。*
