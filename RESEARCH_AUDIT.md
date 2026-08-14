# RESEARCH_AUDIT — 仓库审查与科研可行性评估

> 评估对象：`GeoAI` 仓库（北京高校数学建模联赛 B 题一等奖 · 台风极端降水时空推演）
> 评估角色：科研导师 / 论文审稿人
> 评估日期：2026-08-14
> 本轮范围：**Repository Audit + Research Feasibility Assessment**（未修改任何代码，仅在根目录生成本报告）

---

## 0. Executive Summary（一句话结论）

这是一个**工程完成度高、数据资产真实、应用场景有价值**的数学建模项目，但**目前不具备直接投稿的论文状态**。核心模型是"ConvLSTM + 时序残差 + SE 通道注意力 + 软物理正则"的**标准组件组合**，无自创方法；**重构后的新代码从未真正运行过**（训练集 HDF5 缺失、`outputs/` 为空、无任何新实验指标）；论文里最有分量的结论（"地形使极端降水峰值 +26.4%"）来自**旧版硬编码物理引擎且无法验证**。

**综合评级：B（可改论文，但必须重新设计研究问题并重跑全部实验）。**
最推荐的最小创新路线是：**把"地形约束 + 极端事件加权的物理启发损失"做实并做严格消融**，将论文重新定位为"物理约束对台风极端降水临近预报到底有没有用"的实证/方法交叉论文。

---

## 1. 仓库结构认知（Phase 1）

### 1.1 顶层结构

```
GeoAI/
├── README.md                          # 已重写为论文项目 README（英文）
├── requirements.txt                   # 依赖清单
├── src/                               # ★ 重构后的论文级代码包（2026-07 新增）
│   ├── models/        convlstm_cell.py / pi_res_convlstm.py / baselines.py / channel_attention.py
│   ├── data/          dataset.py / transforms.py
│   ├── training/      trainer.py / physics_loss.py / configs/default.yaml
│   ├── inference/     infer.py
│   ├── evaluation/    metrics.py / evaluate_models.py
│   └── visualization/ plot.py
├── configs/default.yaml               # 训练配置
├── docs/                              # PROJECT_DOC / SUMMARY / CODE_CHANGE_PLAN / 项目评估报告 等
├── deliverables/PI_ResConvLSTM_Paper_Package/   # ★ 论文交付包
│   ├── code/           src/ 的副本
│   ├── paper/main.tex  # ★ "论文"正文（实为竞赛论文改标题，仍为中文竞赛体）
│   ├── tables/         raw/ + derived/ 各 9-10 个 CSV
│   ├── figures/        existing/ 36 张 + generated/ 6 张
│   └── models/         typhoon_convlstm_best.pth（旧权重）
├── archive/competition/               # 竞赛原始材料
│   ├── essay/essay/essay.tex          # 原始竞赛 LaTeX（835 行）
│   ├── essay/essay/*.py               # step1.1 ~ step4.3 竞赛脚本
│   └── old_docs/                      # 校赛 v4、PINN-ConvLSTM 等多版本文档
├── CMABSTdata/                        # CMA 最佳路径 2014-2025（12 个 txt + readme.pdf）
├── TIFdata/                           # GPM IMERG 降水 GeoTIFF，33,420 个 .tif
├── Global_DEM.tif                     # ETOPO1 全球地形 465 MB
├── *.csv                              # 中间特征/结果表（Q1 相关）
├── *.npz / *.pth                      # 竞赛旧预测包 + 旧模型权重
└── outputs/                           # ★ 空（仅 3 个空子目录）
```

### 1.2 版本判断

| 版本 | 位置 | 状态 |
|------|------|------|
| 竞赛原始版 | `archive/competition/essay/essay/essay.tex` + `archive/competition/essay/essay/*.py` | 旧、硬编码、数据泄漏 |
| 竞赛旧文档 | `archive/competition/old_docs/`（校赛 v4、PINN-ConvLSTM、级联残差等） | 历史版本 |
| **当前最终版** | **`src/` + `docs/` + `deliverables/` + `README.md`**（2026-07 重构） | **代码重构完成，但未运行、无结果** |
| "论文"正文 | `deliverables/.../paper/main.tex` | **竞赛论文改标题的变体，仍是中文竞赛体** |

**结论**：当前最终版是 `src/` 重构框架 + `deliverables` 交付包。但"论文正文"没有单独成文，而是竞赛 essay 的改题版本。

---

## 2. 科研视角重新描述项目（Phase 2）

### Research Question（一句话）

> **能否通过将地形抬升约束与极端降水加权注入损失函数，在台风降水临近预报（nowcasting）中同时提升物理一致性与极端事件预测能力？**

（这是项目**想要**回答的问题；但当前仓库并未真正用实验回答它——见第 5 节。）

- **输入**：过去 K=11 帧的多通道时空场 `[K, C, 128, 128]`（降水场 + 风场 + 气压场 + 距台风中心距离；DEM/地形梯度/海陆掩膜在设计文档中被提及但**未进入实际数据通道**）。
- **输出**：下一时刻（+0.5h）降水场 `[1, 128, 128]`，通过 `P_hat = ReLU(P_t + ΔP)` 得到。
- **要解决的问题**：台风降水场的短临时空外推（spatiotemporal nowcasting），重点是极端降水的空间结构与峰值。
- **为什么值得研究**：台风极端降水是真实灾害场景（防灾/内涝/风险），数据多源真实（路径 + 卫星降水 + 地形），是 AI 与大气科学的交叉问题。
- **研究方向**：Applied AI / GeoAI / 气象时空预测（Weather Nowcasting）；跨"ML + 大气科学"。可归入：Time Series + Spatiotemporal + Physics-Informed ML + Computer Vision（降水场可视为图像预测）。

**一句话定位**：这是一个**"物理启发损失 + ConvLSTM 类模型"在台风降水临近预报上的应用型方法交叉课题**，不是算法原创课题。

---

## 3. 技术路线拆解（Phase 3）

```
CMA 路径 txt + GPM 降水 tif + ETOPO1 DEM
        ↓  [step1.1/1.2：三次样条插值、TIF 特征提取、时空对齐]
多源特征 CSV（Q1 特征体系：位置/强度/运动/形态/变化）
        ↓  [step1.3：融合 + 派生特征]
Typhoon_Full_Dataset_Q1.csv
        ↓  [step1.4：Spearman + 随机森林 + SHAP 归因]   ← 竞赛 Q1，科研可删
        ↓  [step2.1：渲染物理场 + 滑窗 → HDF5 张量]
ConvLSTM_Dataset_128.h5  [N, 12, 4, 128, 128]  ← ★ 仓库中缺失
        ↓  [src/models + src/training：ConvLSTM 编码 → 残差 ΔP 预测 + 物理损失]
PI-ResConvLSTM
        ↓  [src/inference：自回归推理]
降水场预测
        ↓  [src/evaluation：MAE/RMSE/SSIM/CSI/POD/FAR/HSS/峰值/面积/中心偏移]
评估
        ↓  [竞赛 step3/step4：虚拟台风 + 敏感性分析]   ← 竞赛 Q3，科研需大幅收敛
气候情景推演（case study）
```

### 各模块逐项定性

| 模块 | 方法 | 是否标准 | 自研内容 | 能否成 contribution | 工程/竞赛性质 |
|------|------|:---:|------|:---:|------|
| 数据同化（样条插值/Haversine/曲率/Rmax） | 标准 | 是 | 特征工程组合 | 否 | 工程 |
| Q1 归因（Spearman/RF/SHAP） | 成熟标准方法 | 是 | 无 | 否（应用分析） | **竞赛 Q1，科研可删** |
| ConvLSTM 编码器 | Shi 2015 | 是 | 无 | 否 | 基线 |
| 时序残差 ΔP 学习 | 成熟思想（ResNet 式） | 是 | 无（predict ΔP 是常见 trick） | 弱 | 可作消融 |
| SE 通道注意力 | Hu 2018 | 是 | 无 | 否（作消融） | 可选 |
| 物理启发损失 | 加权 MSE + 非负 + 地形相关 + 平滑 + 极端 MSE | 部分 | 软约束组合 | **潜在核心（但当前实现有问题）** | 核心候选 |
| 自回归推理 | 标准 | 是 | 误差轨迹记录 | 否 | 工程 |
| 评估指标套件 | 气象学标准（Wilks 2011） | 是 | 无 | 否（但完善） | 基础设施 |
| 虚拟台风情景推演 | 参数扰动 + 敏感性 | 是 | 无 | 否 | **竞赛 Q3，科研需收敛/删除** |

### 当前最核心的技术模块

**表面看**：PI-ResConvLSTM 的"物理启发损失"（`src/training/physics_loss.py`）是全项目唯一可能构成方法贡献的模块。

**实际看**（见第 5.4 节证据）：该损失中**最有"物理味"的两项——地形抬升项 L_oro 与非负项 L_nonneg——在真实数据配置下都是无效的**（前者因通道索引硬编码不匹配而从不触发；后者因输出已被 ReLU 恒为 0）。因此当前真正生效的"物理启发损失"≈ 加权 MSE + 时空平滑 + 极端像素 MSE，物理含量远低于论文宣称。

---

## 4. 创新性审查（Phase 4）—— 最严苛部分

> 结论先行：**方法创新 = 弱；问题创新 = 中；实验创新 = 当前为零。**

### A. Problem Novelty：中偏低
- 台风/热带气旋降水临近预报是**已被大量研究的问题**（ConvLSTM 降水预报、TrajGRU、HKO-7、DGMR、FourCastNet、GraphCast 等）。
- 本项目**没有提出新的 formulation、新约束、新数据场景**。它只是在"台风降水 + 地形 DEM"这个具体应用域做了一次标准的 ConvLSTM 应用。
- 唯一稍显新意的角度是"**地形抬升约束 + 极端事件加权**"的软物理注入，但这属于**应用组合**，不是新问题。

### B. Method Novelty：弱
- 逐项检查，**没有**：新模型、新损失函数、新优化目标、新算法、新特征机制、新权重策略、新约束、新框架、新的组合机制、新的动态策略、新评价体系。
- 主体是 **ConvLSTM + 时序残差 + SE attention + 软物理正则**，全部 off-the-shelf。
- 物理损失项（加权 MSE / ReLU 非负 / TV 平滑 / 相关性地形约束 / 阈值 MSE）**每一个都是已有技术**，组合方式也常见于 physics-informed 文献。
- 诚实结论：**这是"算法组合"，不是算法创新。** 不能因为北京市一等奖而把它描述成新方法。

### C. Experimental Novelty：当前为零（但存在上升空间）
- 数据是**公开标准数据集**（CMA + GPM IMERG + ETOPO1），非新数据、非新 benchmark。
- 若未来补上"多台风独立测试 + 极端事件分层评估 + 地形消融 + baseline 对比"，可以支撑一篇**应用型/实证型**论文，但**支撑不了方法型论文**。
- "大规模实验"不成立：当前只有一个 6921 样本、20 epoch 的旧训练，且收敛失败。

**审稿人视角的一句话判断**：方法上是"A+B+C+D"组合，若无扎实消融与 baseline 对比，会被直接判定为 incremental application；若补足，可作为一篇质量尚可的 GeoAI 应用交叉论文，但绝非顶刊顶会。

---

## 5. 实验完整性审查（Phase 5）

### 5.1 数据

| 检查项 | 结论 | 证据 |
|--------|------|------|
| 数据来源是否明确 | ✅ 明确（CMA/NASA GPM/NOAA ETOPO1） | README + PROJECT_DOC |
| 数据是否公开 | ✅ 均公开 | 同上 |
| 原始数据是否在仓库 | ✅ CMA(12 txt) / TIFdata(33,420 tif) / Global_DEM.tif(465MB) | 实际文件 |
| **训练张量是否在仓库** | ❌ **`ConvLSTM_Dataset_128.h5` 缺失** | `ls` 无此文件，仅有 `ConvLSTM_Dataset_128_metadata.csv` |
| 数据泄漏 | ⚠️ 旧版随机 80/20 泄漏；新版代码已按年份划分，但**未运行验证** | trainer.py `create_dataloaders` |
| train/val/test | ✅ 设计为 2014-2022 / 2023 / 2024（年份划分，避免同台风跨集） | dataset.py |
| 数据量 | ⚠️ 6921 样本（12 帧/样本），对深度时空模型偏小；且仅 6 年（缺 2016/2018-2021） | main.tex Tab dataset_summary |
| 可复现 | ❌ **不可复现**：训练 HDF5 缺失、旧脚本硬编码路径、无 seed 验证产物 | 多处 |

> **严重问题**：核心训练数据 `ConvLSTM_Dataset_128.h5` 不在仓库（约 3.9GB，见 PROJECT_DOC 自述）。没有它，`src/` 的 trainer/evaluate/infer 全部无法运行。这是"论文代码"与"可复现论文"之间的鸿沟。

### 5.2 Baseline

| 层级 | 现状 | 是否补 |
|------|------|------|
| naive baseline | Persistence（已实现，`baselines.py`） | 需**运行** |
| traditional method | 无（无 NWP 类比、无 climatology、无纯物理公式基准） | **需补**（至少 climatology / 纯地形物理基准） |
| strong baseline | PlainConvLSTM、ResConvLSTM（已实现） | 需**运行** |
| SOTA / representative | **完全缺失**（无 TrajGRU / U-Net / PredRNN / DGMR / FourCastNet 等） | **必须补至少 1 个强代表性方法** |

> **关键缺陷**：README 声称"Baseline comparisons: Persistence, ConvLSTM, ResConvLSTM, ResConvLSTM+DEM, PI-ResConvLSTM"，但这些 **baseline 从未被实际训练和评估**——`run_paper_experiments.py` 存在但无任何输出 CSV，`outputs/` 为空。论文里 baseline 表格只有"设计"，没有"数字"。

### 5.3 Metrics

`src/evaluation/metrics.py` 已实现一套**完整且专业**的指标：MAE/RMSE/NRMSE/SSIM/峰值误差/暴雨面积误差/中心偏移/CSI/POD/FAR/HSS/ACC/BIAS/分级精度。这是全项目**最完善的部分**，可直接用于论文。**但没有任何一个数字被真正算出来。**

建议补充：`runtime`、`convergence`（旧日志已显示收敛失败，需要真实收敛曲线）、极端阈值分层（已支持 1/5/10/20/50 mm/h）。不需要机械加 AUC（非分类任务）。

### 5.4 关键正确性问题（优先报告）

1. **L_oro（地形抬升损失）是死代码**：
   - `src/training/trainer.py:29-32` 硬编码 `ORO_U_CHANNEL_IDX=6 / ORO_V=7 / ORO_DH_DX=9 / ORO_DH_DY=10`。
   - `configs/default.yaml:23` 与 `trainer.py:514` 实际 `input_channels=4`（降水/风场/气压/距离）。
   - `trainer.py:247` 守卫 `if X.shape[2] > ORO_DH_DY_CHANNEL_IDX`（即 `4 > 10` 为假）→ `oro_lift` 永不构建 → `physics_loss.py:97` 因 `'oro_lift' not in aux` **L_oro 永不计算**。
   - **结论**：论文的核心卖点"地形抬升约束"在真实数据配置下**从未进入训练损失**。

2. **L_nonneg（非负约束）恒为 0（空约束）**：
   - `trainer.py:236-237` 中 `P_hat = relu(P_prev + delta_p)`，故 `P_hat ≥ 0` 恒成立。
   - `physics_loss.py:94` `L_nonneg = relu(-P_hat)^2` 因此**恒等于 0**。文档自己也承认"若输出已 ReLU 该项可降为安全约束"。

3. **头部结论来自旧版硬编码物理引擎，且无法验证**：
   - "地形使 P_max +26.4% / +20.91%"、"V-INTENSE S_ext +19.2%" 等全部来自旧 `step2_3_generate_metrix.py` 的 `base_rain=(wind/5)**1.2` + `nn_residual×1.5` 硬编码公式（PROJECT_DOC 自述"缺乏文献支撑"）。
   - 新 `src/` 已删除该硬编码，但**没有新的替代结果**。

4. **数值不一致**：
   - `deliverables/.../paper/main.tex:520` 写"37.20 → 29.42，下降 **20.91%**"（数学正确：(37.20-29.42)/37.20）。
   - `README.md:155` 与 `docs/PROJECT_DOC.md:274` 却写"**P_max drops 26.4%**"。
   - 两者矛盾，属于"论文数据与文档不一致"，需修正。

5. **核心预测任务（Q2）无真值验证**：
   - 竞赛目标台风 KONG-REY / MAN-YI 在 GPM 中**无降水记录**（PROJECT_DOC 自述），因此 P_max=37.20 等**全部无法与真值对比**。
   - 正式论文**必须**改为"对 GPM 覆盖完整的历史台风做留出验证"，用 RMSE/SSIM/CSI 证明模型真的比 baseline 好。

### 5.5 Ablation / Sensitivity / Robustness

| 检查项 | 现状 |
|--------|------|
| 消融实验 | 仅"设计"（残差/DEM/attention/各损失项），**无数字**；且 DEM 通道根本不在当前输入里 |
| 敏感性分析 | 旧版有（气压/纬度/移速扰动），但基于不可验证的硬编码引擎 |
| 鲁棒性 | **完全没有**：无多 seed、无噪声扰动、无不同划分、无 leave-one-typhoon-out 实测 |

---

## 6. "数学建模论文味"审查（Phase 6）

`deliverables/.../paper/main.tex`（当前"论文"）**几乎完全是竞赛体**，逐条对照：

1. ✅ 存在"问题一/二/三 + 模型一/二/三"的分题写法（`\section{问题概述}`、`问题重述`、三阶段递进）。
2. ✅ 为覆盖全部赛题堆了 Spearman + RF + SHAP + ConvLSTM + PINN + Attention + 情景推演（"全家桶"）。
3. ✅ 存在无必要的评价体系（SHAP 阈值"物理结论"长达 6 条，属竞赛炫技）。
4. ✅ 主观权重/经验参数过多（`(wind/5)^1.2`、`×1.5`、`α=0.35` 地形系数等）。
5. ✅ 结果无 baseline 对比（baseline 只有"设计表"，无数值）。
6. ✅ 只证明"模型能跑出漂亮图"，未证明"比已有方法好"。
7. ✅ 大量依赖台风问题背景，缺乏可泛化研究问题。
8. ✅ Q3 情景推演纯属 case study（数值不可验证）。
9. ✅ 存在大量比赛式修辞："完美克服""无懈可击""降维打击""彻底揭开黑盒""研究结语与人文反思"。

### 应保留 / 应删除

**保留（论文核心）**：
- 多源数据同化 pipeline（track + satellite + DEM）
- PI-ResConvLSTM 架构（ConvLSTM + 时序残差 + 可选 attention）
- 物理启发损失（**修好后**：地形抬升 + 极端加权 + 平滑）
- 气象评估指标套件
- 按年份划分（防泄漏）的设计

**砍掉（正式论文应删除）**：
- Q1 的 SHAP/RF/Spearman 归因整章（竞赛填题，与核心方法无关；最多压缩成 3 行动机）
- Q3 气候情景推演与"中国沿海风险图"（不可验证 case study；最多保留为一个小应用段落）
- "问题重述 / 基本假设及合理性论证 / 符号说明"三件套（竞赛结构）
- 所有"人文反思"式结语与夸张修辞
- "可复现性与交付说明"（这是竞赛交付物清单，非论文内容）

---

## 7. 论文改造可行性评级（Phase 7）

### 评级：**B**

> 可改论文，但需要**明显重新设计研究问题**，并**从零重跑全部实验**。不是"A"的原因：方法无创新、核心实验全部未跑、最有分量的结论不可验证。不是"C/D"的原因：数据资产真实完整、应用价值真实、已有一半论文级代码（指标套件、防泄漏划分、baseline 骨架），具备收敛为一篇诚实应用论文的条件。

### 分维评分

| 维度 | 分数 | 说明 |
|------|:---:|------|
| Research Question | 6/10 | 任务清晰，但被竞赛三问切割，未收敛为单一可证伪 RQ |
| Problem Value | 7/10 | 台风极端降水是真问题，有实际意义 |
| Method Novelty | 3/10 | 标准组件组合，无自创方法，物理损失实现还有 bug |
| Experimental Completeness | 2/10 | 无已验证结果、baseline 未跑、Q2 无真值、HDF5 缺失 |
| Reproducibility | 3/10 | 代码整洁但缺数据、从未运行、无 seed 验证产物 |
| Paper Potential | 5/10 | 可成应用型/实证型 GeoAI 论文，非方法论文 |
| 改造成本（10=极高） | 8/10 | 需重建 HDF5 + 修损失 + 训所有 baseline/消融 + 多台风验证 |
| 快速发表潜力 | 3/10 | 当前几乎无可写内容，需数周实验 |

---

## 8. 最小可行研究创新（Phase 8）

> 前提：**不推倒重做**，基于现有 `src/` 与数据资产，用最小新增工作量换取最大论文完整度。

### 方案 A（推荐·最快）：做实"地形约束 + 极端加权"的物理启发损失，并做严格消融
- **Idea**：修复 L_oro 死代码（把 DEM 通道真正加入输入、修正通道索引、把非负项改为有意义约束或删除），围绕"物理约束对台风极端降水预测是否有增益"做完整消融 + baseline 对比。
- **Motivation**：这是项目唯一可能的"方法贡献"，且当前实现是坏的，修好它 = 论文故事自动成立。
- **Method**：输入加 DEM/地形梯度/海陆掩膜通道（从 4→~7-8 通道）；重写 `_build_physics_aux` 通道索引；L_oro 用相关性/阈值式地形约束；L_nonneg 改为对 ΔP 的约束或直接删除；保留极端加权与平滑。
- **Novelty**：一个"地形抬升约束 + 极端事件加权"的软物理损失在台风降水预报上的**可复现实证**（诚实定位为应用/方法交叉，不吹成新算法）。
- **Experiments**：重建 HDF5 → 训 Persistence/ConvLSTM/ResConvLSTM/PI-ResConvLSTM(±DEM±物理损失) → 报告 MAE/RMSE/SSIM/CSI/POD/FAR/HSS + 峰值/面积误差，**在 GPM 完整覆盖的历史台风上留出验证**。
- **Cost**：中（数据重建 + 多模型训练，RTX 4090 需数天）
- **Expected Value**：中

### 方案 B（推荐·价值最高）：聚焦"极端降水"的尾部精度，用极端事件专用损失 + 极端事件 benchmark
- **Idea**：把论文重新定位为"台风极端降水临近预报"，把极端事件（>10/20 mm/h）的 CSI/POD/FAR 作为主指标，引入一个针对尾部分布的损失（如 focal-style 加权 / 分位数损失），证明其对极端事件的增益。
- **Motivation**：极端降水是防灾最关心的量，也是 MSE 型模型系统性低估的对象，问题真实且被关注。
- **Method**：在现有 `_extreme_loss` 基础上，系统比较"加权 MSE vs focal 加权 vs 分位数损失"对极端阈值 CSI/POD 的影响。
- **Novelty**：极端事件导向的损失设计与评估在台风降水上的系统比较（实证贡献）。
- **Experiments**：同 A 的 baseline + 消融，外加极端阈值分层评估表。
- **Cost**：中
- **Expected Value**：中高

### 方案 C：深度集成 + 不确定性量化（UQ）
- **Idea**：论文正文已写"N=10 集成"但代码无实现。补齐多 seed 深度集成 / MC-dropout，输出预测不确定性，并验证集合对极端事件命中率的提升。
- **Motivation**：气象临近预报中不确定性信息与点预测同等重要，是评审认可的加分项。
- **Method**：多 seed 训练 → 集合均值/方差 → 用方差做极端事件风险图。
- **Novelty**：台风降水 nowcasting 的 UQ（应用贡献）。
- **Experiments**：多 seed（≥5）、集合 vs 单模型的 CSI/POD/CRPS。
- **Cost**：中高（多次训练）
- **Expected Value**：中

### 方案 D：补强 baseline 的实证基准研究
- **Idea**：放弃方法创新，诚实地做"物理启发 vs 纯数据驱动"的受控对比，补 TrajGRU / U-Net 等强 baseline，输出一篇基准/实证论文。
- **Motivation**：最诚实、最快，且能回答"physics 到底帮不帮忙"。
- **Method**：引入 1-2 个强 baseline（TrajGRU 或 U-Net），统一划分统一评估。
- **Novelty**：弱（纯实证），但审稿风险最低。
- **Experiments**：同 A。
- **Cost**：中
- **Expected Value**：中低（较难作为亮点，但最稳）

### 排序与推荐

| 排序 | 方案 | Cost | Value | 定位 |
|:---:|------|:---:|:---:|------|
| 1 | **A（做实物理损失 + 严格消融）** | 中 | 中 | 最快形成"质量尚可"的第一篇论文 |
| 2 | B（极端事件聚焦） | 中 | 中高 | 价值最高，可与 A 合并 |
| 3 | C（UQ） | 中高 | 中 | 加分项，可作 A/B 的扩展 |
| 4 | D（纯基准） | 中 | 中低 | 兜底最稳 |

> **最终推荐**：**A + B 合并**——把论文定位为"**地形约束 + 极端事件加权的物理启发 ConvLSTM，用于台风极端降水临近预报**"，主实验做 A 的消融，主指标用 B 的极端事件评估。这是"最小新增工作量（主要是把现有半成品代码跑通 + 修一个通道 bug）换取最大论文完整度"的路线。**不要**凭空发明复杂算法。

---

## 9. 重构论文 Story（Phase 9）

- **Problem**：台风极端降水临近预报中，纯数据驱动模型（ConvLSTM）在缺乏充分历史引导时倾向输出平滑均值，**系统性低估极端降水峰值与面积**；同时完全忽略地形强迫抬升等物理先验。
- **Existing Limitation**：标准 ConvLSTM 只用 MSE 拟合，对极端事件尾部不敏感；物理信息（DEM 地形）未被结构化注入。
- **Our Observation**：将地形抬升约束与极端事件加权**写进损失函数**，可在不改变推理结构的前提下，让模型在迎风坡与强降水核心区获得更强监督。
- **Proposed Method**：PI-ResConvLSTM——ConvLSTM 时序残差编码器 + 地形抬升约束 + 极端事件加权 + 平滑正则（注意力作为可选消融）。
- **Experiments**：在 GPM 覆盖完整的多个历史台风上做按年份/按台风留出验证；对比 Persistence / ConvLSTM / ResConvLSTM / PI-ResConvLSTM；逐项消融（残差、DEM、地形损失、极端损失、注意力）。
- **Results**：预期证明（需真实跑出）——物理损失显著提升极端阈值 CSI/POD，且峰值/面积误差下降；注意力收益弱于物理约束。
- **Conclusion**：物理约束是一种低成本的极端降水预报增强手段，地形与极端事件是两个最有价值的注入点。

**为什么理论上/直觉上有效**：极端降水分布长尾 + 地形抬升是已知的降水增强机制；MSE 天然偏向主体像素，对尾部不敏感；把这两条先验作为梯度直接注入，等于用领域知识补偿小样本下的尾部欠拟合。

**实验如何证明**：不是靠"图好看"，而是靠**极端阈值分层 CSI/POD 在多个留出台风上的平均提升 + 消融显著性**。

---

## 10. 初步论文框架（Phase 10）

（若评级 ≥B，本框架成立；当前为 B）

1. **Introduction** ← 迁移：竞赛背景段落中"极端降水重要性 + 纯数据模型冷启动/低估极端"的动机（去掉人文修辞）
2. **Related Work** ← 需新写：ConvLSTM/降水 nowcasting（Shi 2015、TrajGRU、HKO-7）、physics-informed 天气建模（Raissi 2019 PINN 及后续）、极端事件预报
3. **Problem Formulation** ← 迁移：输入/输出/残差 ΔP 定义（`符号说明` 压缩而成）
4. **Method** ← 迁移：模型架构（PI-ResConvLSTM）+ 物理损失（`physics_loss.py` 修正后）+ 数据管道（多源同化）
5. **Experiments** ← 迁移：数据划分（年份 split）+ baseline 设计表 + 指标套件；**需重跑**
6. **Results / Discussion** ← **全新**：基线对比表 + 消融表 + 极端分层表 + 可视化；**当前无数字**
7. **Conclusion** ← 需新写（砍掉"人文反思"）

**竞赛论文中可迁移的部分**：数据源说明 → Section 3/5；模型架构与残差/注意力公式 → Section 4；baseline 设计表 → Section 5；指标定义 → Section 5。**不可迁移**：Q1 归因、Q3 情景推演、问题重述/假设/符号说明三件套、所有夸张修辞。

---

## 11. 论文题目（Phase 11）

1. **Physics-Informed Residual ConvLSTM for Typhoon Extreme Precipitation Nowcasting**
2. **Terrain-Constrained Spatiotemporal Nowcasting of Tropical-Cyclone Rainfall**
3. **Where Does Physics Help? Orographic and Extreme-Event Constraints for Typhoon Rainfall Prediction**
4. **Extreme-Event-Weighted ConvLSTM with Orographic Constraints for Tropical-Cyclone Precipitation Forecasting**
5. **Residual Spatiotemporal Nowcasting with Physics-Guided Losses for Typhoon Rainfall Extremes**

（避免 "A Study of / Research on / Based on"，采用"方法名 + 任务域"或"问题导向"风格。）

---

## 12. 最终结论（Phase 12）

### Verdict：**CONDITIONAL YES**

### 最大优势（3 个）
1. **数据资产真实且完整**：CMA(12 年) + GPM(33,420 TIF) + ETOPO1 DEM 都在仓库，具备可复现的物质基础。
2. **论文级代码骨架已就位**：`src/evaluation/metrics.py` 的气象指标套件、防泄漏的年份划分、baseline/attention/物理损失的模块化实现，是难得的"半成品论文代码"。
3. **应用价值真实**：台风极端降水是明确的灾害场景，问题本身可辩护，适合 GeoAI/气象 AI 交叉定位。

### 最大问题（3 个）
1. **方法无创新**：ConvLSTM + 时序残差 + SE attention + 软物理正则，是标准组件组合，无自研方法。
2. **实验完全未落地**：重构代码从未运行（`outputs/` 空、训练 HDF5 缺失、baseline 未训练、无任何指标数字），且核心结论（地形 +26.4%）来自旧硬编码引擎、数值还自相矛盾（20.91% vs 26.4%）、且对无真值台风无法验证。
3. **物理损失实现有 bug**：最有"物理味"的地形抬升项 L_oro 因通道索引硬编码不匹配而**从不触发**，非负项 L_nonneg 因输出已 ReLU 而**恒为 0**——即论文的核心卖点在代码层面是失效的。

### 最推荐路线

```
现有项目
  ↓ 删除：Q1 归因整章、Q3 情景推演整章、问题重述/假设/符号说明、人文修辞、旧硬编码物理引擎
  ↓ 增加：DEM/地形梯度通道进输入；修 L_oro 通道索引；删/改空约束 L_nonneg；
  ↓       补 1 个强 baseline（TrajGRU 或 U-Net）；重建训练 HDF5
  ↓ 重跑：按年份/按台风留出划分 → 训 Persistence/ConvLSTM/ResConvLSTM/PI-ResConvLSTM
  ↓       → 逐项消融（残差/DEM/地形损失/极端损失/注意力）→ 极端阈值分层评估
  ↓
最终论文核心 contribution：
  "一个地形约束 + 极端事件加权的物理启发残差 ConvLSTM，用于台风极端降水临近预报；
   在多个历史台风留出验证上，证明物理约束显著提升极端阈值 CSI/POD 与峰值误差。"
```

---

## 附录：证据索引（Evidence Index）

| # | 结论 | 证据文件:行 |
|---|------|------|
| E1 | 训练 HDF5 缺失 | `ConvLSTM_Dataset_128.h5` 不存在；仅有 `ConvLSTM_Dataset_128_metadata.csv` |
| E2 | 重构代码未运行 | `outputs/` 三子目录均为空；无 `paper_experiment_metrics_summary.csv` |
| E3 | L_oro 死代码 | `src/training/trainer.py:29-32`（通道 6/7/9/10）vs `configs/default.yaml:23`（input_channels:4）；`trainer.py:247` 守卫；`physics_loss.py:97` |
| E4 | L_nonneg 恒为 0 | `trainer.py:236-237`（ReLU 输出）→ `physics_loss.py:94`（relu(-P_hat)²≡0） |
| E5 | 旧硬编码物理公式 | PROJECT_DOC 自述 `base_rain=(wind/5)**1.2`、`nn_residual×1.5` 无文献支撑 |
| E6 | 数值不一致 | `main.tex:520`（-20.91%）vs `README.md:155` / `PROJECT_DOC.md:274`（-26.4%） |
| E7 | Q2 无真值 | PROJECT_DOC "KONG-REY/MAN-YI 在 GPM 中无实测降水" |
| E8 | baseline 未训练 | `run_paper_experiments.py` 存在但无输出；main.tex 仅给"设计表" |
| E9 | "论文"=竞赛体 | `deliverables/.../paper/main.tex` 含"问题重述/基本假设/符号说明/人文反思"，ctexart 中文 |
| E10 | 训练收敛失败 | `docs/项目评估报告.md`（val loss 1.44→10.04，后期停滞 3.73） |

---

## 附录 B：补遗（2026-08-14，基于 `比赛代码/` 新证据）

> 用户补充了竞赛原始代码 `比赛代码/`（20 个 .py）。以下结论基于源码逐行核对，**强化并细化**了正文，无推翻性更正。详见 `COMPETITION_CODE_AUDIT.md` 与 `CODE_PROVENANCE_MAP.md`。

| # | 更新结论 | 证据 |
|---|---------|------|
| R1 | **数据泄漏坐实**：旧训练用 `random_split(dataset, [0.8, 0.2])`，同台风相邻窗口跨 train/val | `step2_2_train_cloud.py:139-141` |
| R2 | **DEM 从未进模型**：旧训练数据 4 通道 = `[降水, 风场, 气压, 距离]`，无 DEM/地形/海陆；DEM 只在推理引擎读取 | `step2.1_spatial_dataloader.py:111,155-156`；`step2_3` |
| R3 | **旧训练无任何物理损失**：`nn.MSELoss()` 纯数据拟合；"物理"只在推理阶段作硬编码后处理 | `step2_2_train_cloud.py:149` |
| R4 | **旧 `.pth` 架构 = SpatialResidualConvLSTM（内部残差 `relu(p_pred+Δp)`）**，与新版时序残差 `P_t+ΔP` 不同，无法直接加载 | `convLSTM_model.py:68` |
| R5 | **HDF5 可以完整重建**：`CMABSTdata/`+`TIFdata/`+`Global_DEM.tif` 均在仓库；重建脚本 = step2.1 逻辑 + 三处修正（加 DEM 通道 / 写 `/meta` / 移除参数化合成场） | `COMPETITION_CODE_AUDIT.md §8` |
| R6 | **旧代码无任何可复用 baseline 或气象指标**：仅有单模型 + MSE + step3.2 的粗指标(P_max/S_ext)；新论文的 Persistence/ConvLSTM 等 baseline 与 CSI/POD/FAR 全部需新跑 | 全库无对应实现 |
| R7 | **旧"物理"全为硬编码经验公式**（`(wind/5)^1.2`、`×0.35`、`×1.2`、`×1.5`），无文献支撑；唯一可保留的是地形抬升**概念**与科氏参数定义 | `step2_3_generate_metrix.py:66-74,146` |

**对正文评级的净影响**：评级维持 **B** 不变；但"可复现性"一项从 3/10 上调至 4/10（数据重建路径已明确），"方法创新"仍为 3/10（旧代码进一步证明无创新点）。

---

*本报告仅依据仓库内实际文件与代码生成；标注为"缺失/未运行"之处，均经文件系统与代码交叉核对。*
