# PAPER_PLAN — 论文逐章规划（ars-plan / academic-paper `plan` 模式产物）

> 生成日期：2026-08-14
> 依据：`RESEARCH_AUDIT.md`（仓库审查）+ 用户确认的方向（**物理约束消融实证** + **英文**）
> 定位：本科第一篇正式论文 · 不追求顶刊顶会 · GeoAI / 气象 AI / 遥感降水交叉

---

## 0. 论文主张与 Research Question（已按新定位调整）

**Research Question（英文，暂定）**：

> Can terrain-aware physics guidance improve ConvLSTM-based typhoon precipitation nowcasting, and are the benefits particularly pronounced for heavy rainfall and terrain-forced precipitation?

**定位（中文）**：

- **主任务 = 台风降水临近预报（Typhoon Precipitation Nowcasting）**，预测**完整降水场**，主指标 MAE / RMSE / SSIM。
- **极端降水不是唯一目标**，而是困难场景 + 重点子分析：强降水区用 CSI / POD / FAR；高地形梯度区分析 terrain-aware physics guidance 是否更有效。
- **不预设答案**：Hypothesis 与 Conclusion 严格分离；是否"物理有效"由实验决定。

**这是实证/方法交叉论文，不是新算法论文。** 贡献的诚实边界：不宣称发明了 ConvLSTM/PINN/attention，而是**系统地、可复现地检验"地形感知物理引导"对台风降水临近预报的真实收益**。

---

## 1. 章节规划（IMRaD + ML 论文惯例）

| # | 章节 | 建议字数 | 核心内容 | 竞赛材料迁移来源 |
|---|------|:---:|------|------|
| 1 | Introduction | ~800 | 台风极端降水重要性 → 纯数据模型低估极端/忽略地形 → **研究缺口**（缺物理约束的系统实证）→ 本文贡献 | 竞赛"背景与动机"段（去修辞） |
| 2 | Related Work | ~900 | 降水 nowcasting（Shi 2015 ConvLSTM、TrajGRU、HKO-7、PredRNN）→ physics-informed ML（Raissi 2019 PINN 及气象应用）→ 极端事件预测 | **全新撰写** |
| 3 | Method | ~1200 | 问题定义 P_hat=ReLU(P_t+ΔP)；模型（ConvLSTM 编码 + 时序残差 + 可选 SE）；物理损失 L=L_rain+λ₁L_oro+λ₂L_smooth+λ₃L_extreme；多源数据管道 | 模型架构 + 物理损失（`src/` 修正后）+ 数据同化 |
| 4 | Experiments | ~900 | 数据与划分（年份/台风留出，防泄漏）；baselines（Persistence/PlainConvLSTM/ResConvLSTM/PI-ResConvLSTM）；指标套件；实现细节（seed/优化器/早停） | baseline 设计表 + 指标定义（`src/evaluation/metrics.py`） |
| 5 | Results & Discussion | ~1000 | 主对比表 → 消融表 → 极端阈值分层表 → 预测场可视化 → 讨论"物理约束何时有效" | **全新（当前无数字）** |
| 6 | Conclusion | ~300 | 总结贡献 + 局限 + 未来工作 | 新写（砍掉竞赛"人文反思"） |

**全文合计 ≈ 5100 词**（适合会议/workshop 或短篇应用型期刊）。

---

## 2. 各章核心论证（Argument Blueprint 摘要）

### Introduction 的三段论（CARS 结构）
1. **Establish the territory**：台风极端降水是沿海灾害核心；临近预报（nowcasting）是防灾刚需。
2. **Establish the niche**：现有纯数据驱动模型（ConvLSTM 类）以 MSE 为主，系统性**低估极端降水峰值/面积**；地形强迫抬升这一已知物理机制未被结构化注入。
3. **Occupy the niche**：本文提出 PI-ResConvLSTM，用留出台风验证 + 逐项消融实证"物理约束在极端/地形区的真实增益"。

**研究缺口（一句话）**：*尽管 physics-informed 天气模型大量涌现，却缺少对"物理约束损失在台风极端降水预报中、按极端阈值与地形区拆分的系统实证"。*

### Method 的"守底→修正"逻辑
- 主体是标准组件（ConvLSTM + 时序残差 + 软物理正则），**不伪装成新算法**。
- 物理损失三个有效项（修正后）：`L_oro`（地形抬升相关/阈值约束）、`L_extreme`（极端像素加权 MSE）、`L_smooth`（弱时空平滑）。**删除/改造**空约束 `L_nonneg`（输出已 ReLU，恒为 0）。

### Results 的主次（nowcasting 为主，physics 为次）
- **主结果**：PI-ResConvLSTM vs baselines 的**完整降水场指标** MAE/RMSE/SSIM（回答"模型好不好"）。
- **子分析 1（强降水）**：极端阈值 CSI/POD/FAR，回答"强降水区是否预测得更好"。
- **子分析 2（地形）**：按地形梯度分区的指标对比，回答"terrain-aware physics 是否在地形区更有效"。
- **核心消融**：逐项去掉 L_oro / L_extreme / DEM 通道 / 残差，观察上述指标的**变化方向**（不预设"一定变差"）。

---

## 3. INSIGHT 收集（关键结晶）

| INSIGHT | 内容 |
|---------|------|
| `research_question` | Can terrain-aware physics guidance improve ConvLSTM-based typhoon precipitation nowcasting, and are the benefits particularly pronounced for heavy rainfall and terrain-forced precipitation? |
| `primary_task` | 台风降水临近预报（完整降水场，MAE/RMSE/SSIM） |
| `secondary_analysis` | 强降水区（CSI/POD/FAR）+ 高地形梯度区（terrain-aware physics 是否更有效） |
| `contribution_claim` | **（待实验确定，不预设）** 提供"地形感知物理引导"的可复现实现 + 严格 baseline/消融，定位 GeoAI 应用/方法交叉论文 |
| `honest_boundary` | 不宣称新算法；"物理启发"非严格 PINN（无 PDE 残差），表述用 physics-informed / physics-guided / terrain-constrained |
| `key_risk` | Hypothesis 与 Conclusion 分离：若物理引导无显著增益，需诚实报告（负面结果也可成文，但需预先设计好"至少证明某一子假设"的兜底）；核心依赖实验重跑 |

---

## 4. 下一步（进入 `full` 模式前的先决条件）

> ⚠️ **IRON RULE**：plan 模式的产出必须先经你确认，才能进入 `full`（全文撰写）。

在写正文之前，需要先落地 4 件事（均来自 `RESEARCH_AUDIT.md`）：
1. **重建 `ConvLSTM_Dataset_128.h5`**（当前缺失；需在服务器用 `step2.1` 逻辑重建）。
2. **修 `physics_loss` 的通道索引 bug**（`trainer.py:29-32` 硬编码 6/7/9/10，但 `input_channels=4`），把 DEM/地形梯度真正加入输入通道。
3. **删/改空约束 L_nonneg**。
4. **重跑全部 baseline + 消融**，得到 `paper_experiment_metrics_summary.csv`。

在 1–4 完成之前，论文只能写到"方法 + 实验设计"，**写不了"结果"**。

---

## 5. 待你确认的问题

1. **目标载体**：会议/workshop（SIGSPATIAL workshop、AGU/EMS、中文英文均可）还是短篇期刊（Remote Sensing / Atmosphere / Water / Earth Science Informatics 等）？这决定篇幅与写作风格。
2. **是否现在就进入逐章细化**（Step 2 的章节级苏格拉底对话），还是先按本规划去补实验、回来再写？

---

*本文档是 plan 模式产物（Chapter Plan + INSIGHT），非正文草稿。*
