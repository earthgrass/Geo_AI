# ACADEMIC_RESEARCH_DECISION_REPORT — 台风降水临近预报科研方向决策研究

> 生成日期：2026-08-14
> 项目：Geo_AI 台风降水临近预报（TC Precipitation Nowcasting）
> 本报告为**文献调研 + 科研方向决策**，不涉及代码修改、模型训练或数据下载。

---

## 1. Executive Summary

本项目的核心任务——**用过去 11 个 30 分钟帧预测 +30 分钟台风完整降水场（GPM IMERG）**——是一个**已被大量研究但仍有明确应用价值**的问题。方法层面（ConvLSTM + 残差 + 多源特征融合）**不构成新方法**：最近文献（graph-guided ConvLSTM for TC、PTC-ConvLSTM、TCP-Diffusion、TyrainNow、Nowcasting-Nets）已经做了"TC track 融合 ConvLSTM""GPM IMERG 30 分钟临近预报""极端降水损失"。

**诚实结论：方法创新有限（WEAK），但存在一条可辩护的应用/实证论文路径**——即"地形感知（terrain-aware）对台风降水临近预报的**严格消融 + 事件级未见台风泛化评估**"，而非"物理信息/PINN 新方法"。

关键决策（详见各节）：
- **ERA5 = OPTIONAL**（核心问题可用现有 DEM 数据回答，ERA5 仅用于可选的 orographic 物理扩展）
- **q(V·∇h) 目前无法计算**（缺环境风场 V；u_move/v_move 是平移速度不是环境风），若引入 ERA5 可作 INPUT FEATURE / 弱物理约束，**不是 PINN**
- **Full PINN = NO-GO**（Smith & Barstad 2004 需要不可观测的云水/水凝物潜变量，36 事件/6867 样本不可辨识）
- **定位 = Terrain-Aware**（不用 Physics-Informed / PINN 术语）

---

## 2. Literature Landscape

三个活跃子领域：
1. **TC 降水临近预报深度学习**：从 ConvLSTM(2015)→TrajGRU(2017)→TC 专用（graph-guided 2022、PTC-ConvLSTM 2024、TyrainNow 2025、TCP-Diffusion 2024）。
2. **卫星降水临近预报**：Nowcasting-Nets(2021，GPM IMERG)、Dense-Cast(2026，GPM IMERG 30min)。
3. **物理信息降水**：LUPIN(2024)、PIANO(2025)、PID-GAN(2024)、Choma(2023)——均用**平流/平流-扩散方程**作为物理约束，不是地形抬升。

---

## 3. 20 篇最相关论文（已验证存在）

> 标注 "ABSTRACT ONLY" 表示本调研仅取得摘要/元数据，未读全文。DOI 均来自检索结果，未编造。

| # | 论文 | 年份 | 出处 | 相关性 |
|---|------|------|------|--------|
| 1 | Shi et al., ConvLSTM for Precipitation Nowcasting | 2015 | NeurIPS; arXiv:1506.04214 | 基础模型 |
| 2 | Shi et al., TrajGRU: A Benchmark and A New Model | 2017 | NeurIPS; arXiv:1706.03458 | 基础/强 baseline |
| 3 | Yang et al., Spatio-temporal graph-guided ConvLSTM for TC precipitation nowcasting | 2022 | Applied Soft Computing | **最接近竞品** |
| 4 | Liu et al., PTC-ConvLSTM (TC track + ConvLSTM for inflow flood) | 2024 | EGU24 | 竞品 |
| 5 | TCRainNet (interpretable TC rainfall nowcasting + flood) | 2024 | J. Hydrology | 竞品 |
| 6 | Yao & Chen, TyrainNow (U-Net + STR loss) | 2025 | WRR; DOI:10.1029/2025WR039897 | 竞品 |
| 7 | Huang et al., TCP-Diffusion (multi-modal TC precip forecast + NWP) | 2024 | arXiv:2410.13175 | 竞品 |
| 8 | Meng et al., TCR-GAN (TC IR→rainfall) | 2022 | arXiv:2201.07000 | 竞品 |
| 9 | Ehsani et al., Nowcasting-Nets (GPM IMERG nowcasting) | 2021 | arXiv:2108.06868 | **同数据** |
| 10 | Kalita & Singh, Dense-Cast (GPM IMERG 30min) | 2026 | arXiv:2608.06082 | 同数据/同期 |
| 11 | Ko et al., Effective Training Strategies (pre-train + heavy-rain loss) | 2022 | arXiv:2202.10555 | 极端损失 |
| 12 | Cao et al., StarBriNet (multi-sigmoid loss) | 2019 | arXiv:1907.08069 | 损失 |
| 13 | Kleiber et al., Stochastic TC precipitation field generation | 2020 | arXiv:2011.09918 | TC 降水统计 |
| 14 | Pavlík et al., LUPIN (Lagrangian physics-informed nowcasting) | 2024 | arXiv:2402.10747 | 物理信息 |
| 15 | Chin et al., PIANO (physics-informed dual neural operator, advection-diffusion PINN loss) | 2025 | arXiv:2512.01062 | 物理信息 |
| 16 | Yin et al., PID-GAN (physics-informed discriminator GAN) | 2024 | arXiv:2406.10108 | 物理信息 |
| 17 | Choma et al., Improving nowcasting by using prior knowledge (PhyCell advection-diffusion) | 2023 | arXiv:2301.11707 | 物理先验 |
| 18 | van Wonderen & Mehrkanoon, MAD-SmaAt-GNet (multimodal advection-guided) | 2026 | arXiv:2603.04461 | 物理+多模态 |
| 19 | Smith & Barstad, A Linear Theory of Orographic Precipitation | 2004 | J. Atmos. Sci. 61(12):1377-1391 | **地形物理核心** |
| 20 | Raissi et al., Physics-Informed Neural Networks | 2019 | J. Comput. Phys. 378:686-707 | PINN 定义 |

---

## 4. 最接近的 5–8 个竞品

| 论文 | 任务/时效 | 数据 | 模型 | GPM | TC track | DEM | ERA5 | 环境风 | 极端损失 | 物理约束/PINN |
|------|-----------|------|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Yang 2022 graph-guided ConvLSTM | TC 降水临近预报 | 雷达+TC 多源 | ConvLSTM+GSTM | — | ✅ | — | — | — | — | — |
| Liu 2024 PTC-ConvLSTM | TC 降水→洪水 | 雷达+TC track | ConvLSTM | — | ✅ | — | — | — | — | — |
| TyrainNow 2025 | TC 降水 0-2h | 雷达(1km/6min) | U-Net | — | — | — | — | — | ✅(STR) | — |
| TCP-Diffusion 2024 | TC 降水 12h | GPM+NWP+环境 | Diffusion | ✅ | ✅ | — | ✅(NWP) | ✅ | — | 弱(残差) |
| Nowcasting-Nets 2021 | 降水 0-1.5h | GPM IMERG | RNN/CNN | ✅ | — | — | — | — | — | — |
| Dense-Cast 2026 | 降水 30min | GPM IMERG | DenseNet+Trans | ✅ | — | — | — | — | — | — |
| TCR-GAN 2022 | TC IR→降水 | 红外+微波 | GAN | ✅ | — | — | — | — | — | — |

**与我们的相同点**：TC 降水 + 卷积/循环时空模型 + TC track 融合（Yang/Liu/TCP-Diffusion）+ GPM IMERG 数据（Nowcasting-Nets/Dense-Cast）。

**真正不同点**：
1. **DEM/地形梯度/海陆掩膜作为显式通道 + 严格地形消融**——上述竞品**均未**把静态地形作为独立可消融输入（这是本项目的潜在差异化）。
2. **严格因果 + 事件级未见台风划分**——多数竞品用随机窗口划分，未见 TC 泛化评估更弱。
3. **30 分钟 + 30 分钟 GPM 单一卫星源**——与雷达产品（TyrainNow、Yang）不同，与 Nowcasting-Nets/Dense-Cast 相同。

---

## 5. Novelty Audit

| 项目 | 判定 | 依据 |
|------|------|------|
| ConvLSTM | NOT NOVEL | Shi 2015 |
| 残差 ConvLSTM | WEAK | 常见 trick（TCP-Diffusion ARP 也做残差） |
| 降水残差预测 ΔP | WEAK | TCP-Diffusion ARP、Nowcasting-Nets 残差头 |
| TC track 融合 | NOT NOVEL | Yang 2022、Liu 2024 |
| CMA 最佳路径融合 | WEAK | Liu 2024 用 TC track |
| 风暴运动 u_move/v_move | WEAK | 竞品多用 TC 位置/速度 |
| DEM | **POTENTIALLY NOVEL**（在 TC 降水临近预报中少见） | 竞品均无 DEM |
| 地形梯度 dh_dx/dh_dy | **POTENTIALLY NOVEL** | 同上 |
| 海陆掩膜 | WEAK | 常见 |
| ERA5 | NOT NOVEL | TCP-Diffusion 用 NWP |
| 极端降水损失 | WEAK | Ko 2022、TyrainNow STR、StarBriNet |
| 物理引导损失 | WEAK | LUPIN、PIANO、Choma |
| q(V·∇h) | WEAK（标准 orographic 强迫代理） | Smith & Barstad 2004 的经典概念 |
| PINN | NOT NOVEL | Raissi 2019 |
| 事件级未见 TC 泛化 | **POTENTIALLY NOVEL**（方法学更强，但非"新方法"） | 多数竞品随机划分 |
| 严格因果轨迹构建 | **POTENTIALLY NOVEL**（实现细节，非贡献） | 未见竞品明确讨论 |
| 30 分钟 TC 降水 | WEAK | Nowcasting-Nets、Dense-Cast |

**核心判断**：唯一可能构成**真实差异化**的是"**静态地形信息（DEM+梯度+海陆）在 TC 降水临近预报中的独立贡献，通过严格消融 + 事件级未见台风验证**"。这属于**应用/实证贡献**，不是方法贡献。

---

## 6. Current Project Strengths

1. **数据管线严谨**：因果轨迹重建、事件级划分（未见台风）、DEM 地理精确对齐、泄漏防护——这是多数竞品论文**没有做到**的方法学严谨性。
2. **地形消融的独特性**：DEM/地形梯度作为显式可消融输入，在 TC 降水临近预报中少见。
3. **可复现性**：固定种子、冻结划分、流式构建、完整验证器。

---

## 7. Current Project Weaknesses

1. **样本规模小**：36 事件 / 6867 样本，测试仅 4 事件。
2. **模型不新**：ConvLSTM 已 10 年，残差/多源融合是常规。
3. **无环境风场**：无法真正做 orographic 物理（q(V·∇h)），"物理"主张受限。
4. **test 集 4 事件**：统计检验难、伪复制风险高。

---

## 8. ERA5 Recommendation

### 结论：**ERA5 = OPTIONAL**

**理由**：
- **核心问题（地形是否帮助 TC 降水临近预报）可用现有 DEM 数据回答**，不需要 ERA5。
- 竞品（Yang、Liu、TyrainNow）大多**不用 ERA5**，用雷达/IR/track 就能达到 state-of-the-art。
- ERA5 引入额外成本（下载、时空对齐、**因果性问题**，见 §22）与过拟合风险（36 事件）。

**仅在以下情况 ADD**：追求 orographic 物理扩展（q(V·∇h) 输入或弱约束）时，才需要 ERA5 的 u/v/q。

---

## 9. 850 vs 925 vs Multi-Level Decision

- **若需风-湿度-地形交互**（orographic 强迫），最可辩护的表示是 **925 hPa**（近地面，最贴近地形强迫层），而非 850 hPa 惯例。
- **单层不足以**刻画地形抬升（地形高度跨越 925→850 hPa）。
- **最物理正确**：**垂直积分水汽通量 IVT（integrated vapor transport）**，因为它直接进入地形降水的强迫项（Smith & Barstad 2004 的 vertically-integrated 框架）。

### 结论
- 若选单层：**925 hPa**（u/v/q925）。
- 若选更严谨：**IVT 或 u/v/q 的 925+850 双层层叠**。
- **不要仅按惯例选 850 hPa。**

---

## 10. q(V·∇h) 物理推导

### 1. V·∇h 从哪来？
来自**地形下边界条件**：当水平气流 V 遇到地形高度 h，空气被迫沿坡上升，其垂直速度近似为
```
w_oro ≈ V · ∇h = u·∂h/∂x + v·∂h/∂y
```
这是山岳气象学（Houze, "Orographic Effects on Precipitating Clouds"）与线性地形降水理论（Smith 1979; Smith & Barstad 2004）的基础。

### 2. 物理含义
V·∇h 是**地形诱导的垂直运动**（地形强迫抬升的垂直速度），不是降水率。

### 3. q(V·∇h) 的含义
乘以比湿 q 后，`q(V·∇h)` 近似**向上的水汽通量 / 凝结潜力 / 地形强迫代理**（"上坡水汽供应率"）。它是**强迫项**，不是降水本身。

### 4. 单位
- V·∇h：[m/s]（垂直速度）
- q(V·∇h)：[(kg/kg)·(m/s)] = 水汽通量密度（近似凝结率代理）

### 5. 能否直接表示降水率？
**不能。** 降水 = 凝结 × 转化效率 × 平流/蒸发。q(V·∇h) 只代表"潜在抬升水汽供给"，忽略微物理转化时间（Smith & Barstad 的 cloud delay time τ_c、τ_f）、下坡蒸发、降水平流。

### 6. 有效性假设
线性理论假设：稳定层结、小振幅地形、饱和气流、稳态、无深对流、无地形阻挡/绕流。

### 7. 何时失效
未饱和气流、深对流（TC 眼墙/雨带）、微物理、地形阻挡/绕流、背风效应、雨带动力学、水汽辐合主导（TC 自身环流）、非地形强迫的降水。

---

## 11. q(V·∇h) Validity Assessment

- 在**层状地形降水（orographic stratiform rain）**下，q(V·∇h) 是合理的强迫代理（Smith & Barstad 2004）。
- 在**台风核心（眼墙、螺旋雨带）**下，降水主要由 TC 自身的对流/辐合驱动，**q(V·∇h) 只是次优修正**，不能代表主降水机制。
- 因此，把它当作"降水率"是**过度简化**；当作"地形强迫代理/诊断量"是**可辩护的**。

---

## 12. PINN Definition Audit

| 术语 | 定义 | 本项目是否满足 |
|------|------|:---:|
| physics-derived feature | 从物理量派生的输入特征 | ✅ DEM 梯度 |
| physics-based feature | 基于物理的输入特征 | ✅（若加 q(V·∇h)） |
| physics-guided model | 用物理指导架构/特征 | ⚠️ 部分 |
| physics-guided regularization | 物理软约束损失 | ❌ 未实现 |
| physics-informed loss | 损失含物理残差/约束 | ❌ |
| physics-informed neural network | 损失含 PDE 残差（Raissi 2019） | ❌ |
| PINN | 严格：控制 PDE + 残差 + 初/边值 + 数据损失 | ❌ |

**明确回答**：
- 把 q(V·∇h) 只作为**输入特征** → **不是 PINN**，是 physics-derived feature。
- 把 q(V·∇h) 用于**软损失** → 也不是严格 PINN，是 **physics-guided regularization**。
- 严格 PINN 要求：**已知控制 PDE、可计算 PDE 残差、初值/边界条件、数据损失项**（Raissi et al. 2019）。

---

## 13. Full PINN Feasibility

Smith & Barstad (2004) 的线性地形降水是一组**垂直积分稳态方程**（气流动量、凝结水平流、下坡蒸发），其连续形式涉及：
- 凝结水源项 S（≈ 地形强迫 × 水汽）
- 凝结水 advection + 转化延迟（cloud water → hydrometeor）

用户示例方程
```
∂qc/∂t + U·∇qc = S_oro − qc/τ_c
∂qh/∂t + U·∇qh = qc/τ_c − qh/τ_f
P = qh/τ_f
```
是 Smith & Barstad 思想的**离散化示意**，不是原文逐字方程（原文是 Fourier 域的垂直积分形式）。

**所需变量**：qc（云水）、qh（水凝物）、U（风）、S_oro（地形强迫）——其中 qc、qh、τ_c、τ_f **不可直接观测**。

**结论：Full PINN = NO-GO**（详见 §9 观察性限制）。这些潜变量在 36 事件/6867 样本下不可辨识，训练不稳定，2-4 个月不可行。

---

## 9(补充). Observability / Data Limits

| 变量 | 可观测？ |
|------|:---:|
| GPM 降水 P | ✅ |
| CMA 风暴状态（气压/风速/位置/平移） | ✅ |
| DEM / 梯度 / 海陆 | ✅ |
| ERA5 u/v/q（若加） | ✅（诊断，非严格实时） |
| 云水 qc | ❌ |
| 水凝物 qh | ❌ |
| 凝结率 | ❌ |
| 微物理转化率 τ_c/τ_f | ❌ |

**可行性**：这些潜变量**不能可靠学习**——需要额外大气观测、系统不可辨识、训练不稳定。在 36 事件/6867 样本/2-4 个月约束下，**不可行**。

---

## 14. Route A vs B vs C

| 维度 | A 物理派生特征 | B 物理引导（特征+软约束） | C Full PINN |
|------|:---:|:---:|:---:|
| 科学严谨 | 中 | 中高 | 高 |
| 新颖性 | 低-中 | 中 | 高 |
| 实现难度 | 低 | 中 | 高 |
| 数据要求 | 低 | 中（需 ERA5 风） | 高（需潜变量） |
| 训练稳定性 | 高 | 中 | 低 |
| 审稿人可辩护性 | 中 | 中 | 低（若不成立） |
| 期望贡献 | 低 | 中 | 高（但风险极高） |
| 风险 | 低 | 中 | 高 |
| 2-4 月可行性 | ✅ | ⚠️ | ❌ |

### 结论：**Route A（物理派生特征）为主路线**；Route B 作为 ERA5 加入后的可选扩展；Route C 放弃。

---

## 15. Recommended Architecture

**保持现状（Terrain-Aware Residual ConvLSTM，12 通道，ΔP 残差）**，不增加复杂模块。理由：
- 残差 ΔP 已合理（Nowcasting-Nets、TCP-Diffusion 也用残差）。
- 竞品（Yang、Liu）证明 ConvLSTM + TC 信息即可达 SOTA 附近，无需更大模型。
- 审稿人优先看重**干净消融**而非模型大小。

可选小改进（非必须）：用 **U-Net 或 TrajGRU 作为强 baseline** 而非只比 ConvLSTM。

---

## 16. Recommended Physics Integration

**只做 Route A**：
- DEM、dh_dx、dh_dy、land_mask 作为**输入特征通道**（现状）。
- **不**在损失里加物理约束（无环境风，且会与"地形是否帮助"的消融混淆）。
- **若未来加 ERA5**：可加 q(V·∇h) 作为额外**输入特征**（Route A），或作为**弱正则**（Route B，需谨慎，标 WEAK HEURISTIC）。

---

## 17. Recommended Loss

**保持 MSE 为主损失（现状），E5 加 L_extreme（极端像素 MSE）。**

- MSE 会低估强降水（回归偏向均值），加权/平衡损失有据（Ko 2022、TyrainNow STR、StarBriNet）。
- 当前 `MSE + λ·ExtremeMSE` **科学上合理且简单**，无需 focal/分位数等复杂设计。
- **不要**把 extreme 损失称作"物理"。

---

## 18. Recommended Strong Baselines

**推荐 1 个强 baseline：TrajGRU（Shi 2017）**——标准、与 ConvLSTM 同类可直接对比、实现成熟。

备选：U-Net（TyrainNow 用，简单，但非时序）。**不建议** DGMR/NowcastNet/PredRNN（工程成本高，样本少）。

---

## 19. Final Experiment Matrix

| ID | 实验 | 输入 | 损失 | 目的 |
|----|------|------|------|------|
| E0 | Persistence | P(t) | — | 下限 |
| E1 | Plain ConvLSTM | 降水 | MSE | 纯数据驱动 |
| E2 | ResConvLSTM | 降水 | MSE | 残差学习 |
| E3 | ResConvLSTM + CMA | 降水+强度+气压+运动+几何 | MSE | 风暴信息 |
| **E4** | **Terrain-Aware ResConvLSTM** | E3 + DEM+梯度+海陆 | MSE | **主模型：地形贡献** |
| E5 | Terrain-Aware + Extreme | E4 | MSE+λ·L_extreme | 极端降水 |
| B1 | TrajGRU（强 baseline） | 降水 | MSE | 强对比 |

共 **7 个实验**（现实可行）。

---

## 20. Evaluation Plan

- **连续**：MAE、RMSE、SSIM；可选 FSS（邻域技巧，适合位移敏感性）。
- **分类（阈值）**：CSI、POD、FAR、ETS；HSS 可选。
- **阈值**：GPM 是 mm/h 瞬时降水率；TC 降水文献常用 **5、10、20、30 mm/h**（Yang 2022 用 5/10/20/30）。**30 分钟**临近预报建议聚焦 **1、5、10、20 mm/h**（30min 窗口内 50mm/h 罕见、样本少）。

---

## 21. Statistical Testing Plan

**核心：不要把 707 个滑动窗口当作 707 个独立样本。**

- **主指标按事件聚合**：先对每个台风事件算指标，再对 4 个 test 事件做**事件级统计**。
- **事件级配对比较**（E4 vs E3 在每个 test 台风上的指标差）。
- **事件级 bootstrap**（对 4 个事件重采样）或 **块 bootstrap**（对时间块重采样）。
- 报告 **中位数 + 置信区间**，避免伪复制。
- 4 个 test 事件不足以做强显著检验 → 诚实报告为**探索性/描述性**，并说明小样本限制。

---

## 22. ERA5 Causality Assessment

- ERA5 是 **4D-Var 再分析**，其"分析场"在时间 t 使用了**同化窗口内 t 之后**的观测（平滑、未来信息）。
- 因此 **ERA5 analysis 用于 t→t+30min 预测，严格来说不是因果的实时临近预报**。
- 这是**离线后报研究（offline hindcast）**，不是**业务实时临近预报（operational nowcasting）**。

**缓解**：
- 明确论文定位为 **offline hindcast study**。
- 或用 **ERA5 短时预报场（forecast fields）**（严格因果，但精度低）作为替代。
- 或用 **ERA5T（近实时版）**，仍有同化延迟。

---

## 23. Reviewer 2 Attack

| 攻击 | 风险 | 缓解 |
|------|:---:|------|
| 36 事件太少 | HIGH | 诚实报告，事件级统计，不夸大 |
| test 仅 4 事件 | HIGH | 事件级评估 + 明确限制；可加 leave-one-typhoon-out |
| ConvLSTM 过时 | MEDIUM | 定位为应用/基准，非方法创新；加 TrajGRU 对比 |
| 地形收益=更多通道 | MEDIUM | 控制变量消融：E3 vs E4 仅差地形通道，同等通道数对照 |
| DEM 只编码位置 | MEDIUM | 加"DEM 消融 vs 随机地形"对照，证明不是纯位置 |
| q(V·∇h) 过度简化 | MEDIUM | 若用则标"强迫代理"，不称降水率 |
| 单层 850/925 不足 | MEDIUM | 若加 ERA5 用 925 或 IVT，并说明局限 |
| ERA5 非因果 | MEDIUM | 定位 offline hindcast；或用预报场 |
| 比湿 vs 混合比 | LOW | 用标准比湿定义并说明 |
| 缺饱和条件/微物理 | MEDIUM | 承认地形抬升≠降水，只作诊断 |
| 地形梯度精度 | LOW | 已用 m/km 物理单位 + 重采样 |
| 抬升≠降水 | MEDIUM | 措辞严谨 |
| 极端损失调参 | LOW | 固定阈值，报告敏感性 |
| PINN 术语滥用 | HIGH | **绝不用 PINN**，用 terrain-aware |
| +30min persistence 强 | HIGH | 必须比 Persistence，报告相对提升 |
| 新颖性=多源特征堆叠 | HIGH | 聚焦"地形消融 + 事件级未见泛化"这个可证伪命题 |

---

## 24. Final Research Gap

> 尽管深度学习已用于台风降水临近预报，多数模型融合 TC 轨迹与降水数据，却**鲜有工作系统隔离"静态地形信息"的独立贡献**，并在**严格因果、事件级未见台风**协议下评估——这正是本项目的切入点。

---

## 25. Final Research Question

> **加入静态地形信息（高程、地形梯度、海陆掩膜）能否在降水+风暴轨迹基线之上提升 30 分钟台风降水临近预报，且收益是否集中于强降水和地形强迫降水？**

---

## 26. Recommended Paper Title

> **Terrain-Aware Residual ConvLSTM for 30-Minute Tropical-Cyclone Precipitation Nowcasting: A Causal Event-Level Evaluation**

（备选：**Does Static Terrain Information Improve Tropical-Cyclone Precipitation Nowcasting?** — 问题式标题，更诚实）

---

## 27. Recommended Paper Positioning

**Terrain-Aware**（不用 Physics-Informed / Physics-Guided / PINN-based）。

理由：本项目目前只有地形**信息作为输入特征**，没有真正的物理约束或 PINN。用 "Physics-Informed" 会被审稿人立刻攻击（见 §12/§23）。

---

## FINAL VERDICT

**A. 是否有真正的可发表研究价值？**
**BORDERLINE**（偏向 YES，但必须定位为应用/实证论文，非方法论文）。

**B. 最强 1–3 个真实贡献**
1. 一个**严格因果、事件级未见台风**的 TC 降水临近预报评估协议（方法学贡献）。
2. **静态地形信息（DEM+梯度+海陆）对 TC 降水临近预报的独立贡献**的严格消融证据（实证贡献）。
3. （可选，若加 ERA5）地形强迫代理 q(V·∇h) 作为物理派生特征的增量价值。

**C. 弱新颖/伪新颖，不应营销**
- ConvLSTM + 残差 + 多源融合（伪新颖）。
- "PINN"/"physics-informed" 术语（会招致攻击）。
- q(V·∇h) 作为"降水率"（过度简化）。

**D. ERA5 = OPTIONAL**（核心问题不需；仅 orographic 扩展需）。

**E. q(V·∇h) 主要角色 = INPUT FEATURE**（若加 ERA5）；当前 = DIAGNOSTIC ONLY（无环境风）。

**F. Full PINN = NO-GO**（潜变量不可辨识，样本不足）。

**G. 唯一推荐主路径**
**Route A（物理派生特征）：Terrain-Aware Residual ConvLSTM + 严格地形消融（E3 vs E4）+ 事件级未见台风评估，定位 Terrain-Aware，MSE 主损失，TrajGRU 强 baseline。不碰 PINN，不强行加 ERA5。**

**H. 最终实验矩阵（7 个，现实）**
E0 Persistence / E1 Plain ConvLSTM / E2 ResConvLSTM / E3 ResConvLSTM+CMA / **E4 Terrain-Aware（主）** / E5 Terrain-Aware+Extreme / B1 TrajGRU。

**I. Research Gap（Introduction 可用一句话）**
> 现有台风降水临近预报工作鲜有在严格因果、事件级未见台风协议下系统隔离静态地形信息的独立贡献。

**J. Research Question（一句话）**
> 静态地形信息能否在降水+风暴轨迹基线之上提升 30 分钟台风降水临近预报，且收益是否集中于强降水和地形强迫降水？

---

## 附：执行边界声明

本报告仅做文献调研与方向决策。**未**修改代码、**未**训练模型、**未**下载 ERA5、**未**重建数据集、**未**修改划分与实验配置。标注 "ABSTRACT ONLY" 的论文仅基于摘要/元数据，未读全文，未编造 DOI。
