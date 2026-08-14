# 台风降水 PI-ResConvLSTM 项目汇总文档

> 基于多源数据同化、物理启发损失与残差 ConvLSTM 的论文转化方案


## 0. 一句话结论

这个项目比普通课程项目更有论文转化价值，适合包装成 **GeoAI / 气象 AI / 遥感降水建模 / 自然灾害风险模拟** 方向的应用型交叉论文。建议主线不要堆成“PINN + ConvLSTM + Attention + SHAP + 情景推演”的全家桶，而是收敛为一个干净、可解释、可验证的模型：

**PI-ResConvLSTM：Physics-informed Residual ConvLSTM for Typhoon Precipitation Simulation**

核心策略：

- 主体模型：Residual ConvLSTM。
- 物理增强：地形抬升约束、降水非负性约束、弱时空平滑约束、极端降水加权损失。
- Attention：先不要作为主贡献；可作为可选消融模块，优先用轻量 Channel Attention。
- 论文定位：短期更适合转成地理/气象/遥感 + AI 交叉论文；DMOEA 更适合作为长期算法论文。

---

## 1. 项目定位

### 1.1 当前项目价值

本项目来自北京高校数学建模竞赛一等奖成果，任务背景是台风极端降水时空推演。题目要求利用台风路径数据、卫星遥感降水数据，以及可自行补充的环境变量，建立能够刻画“路径—强度—环境—降水”关系的数学模型，并生成缺失降水记录台风以及未来虚拟台风情景下的降水时空分布。

这个任务天然具备论文价值，因为它不是单纯分类或回归，而是一个真实的跨学科建模问题：

- 有真实灾害场景：台风极端降水、防灾减灾、城市内涝、风险评估。
- 有真实多源数据：台风路径、强度、卫星降水、DEM、海陆掩膜、地形梯度等。
- 有时空预测任务：输入过去多帧降水和台风动力特征，输出下一时刻降水空间场。
- 有可解释需求：需要解释路径、气压、风速、地形等因素如何影响极端降水。
- 有情景推演需求：可构造路径偏移、强度增强、热力增强等未来气候变化情景。

### 1.2 论文定位建议

推荐论文题目方向：

中文：

**基于物理启发残差 ConvLSTM 的台风极端降水时空推演与气候情景模拟**

英文：

**Physics-informed Residual ConvLSTM for Spatiotemporal Typhoon Precipitation Simulation under Climate Scenario Perturbations**

更保守、更稳的版本：

**A Physics-guided Residual ConvLSTM Framework for Typhoon Rainfall Nowcasting and Scenario Simulation**

注意：如果没有明确 PDE 残差，不建议强行使用严格意义上的 “PINN” 表述。更稳的说法是：

- Physics-informed ConvLSTM
- Physics-guided ConvLSTM
- Terrain-constrained Residual ConvLSTM
- Physics-informed Residual ConvLSTM

---

## 2. PINN / Physics-informed 到底怎么融入

### 2.1 不要误解 PINN

PINN 不是“把气象变量丢进模型”就结束，也不是让模型完全按照一个物理方程计算。严格意义上，PINN 是把物理方程或物理规律写入损失函数，让模型既拟合数据，又尽量满足物理约束。

普通模型：

```text
Loss = 数据预测误差
```

Physics-informed 模型：

```text
Loss = 数据预测误差 + 物理约束误差
```

推荐总损失：

```text
L_total = L_rain + λ1 L_nonneg + λ2 L_oro + λ3 L_smooth + λ4 L_extreme
```

其中：

- `L_rain`：降水预测误差。
- `L_nonneg`：降水非负性约束。
- `L_oro`：地形抬升一致性约束。
- `L_smooth`：弱时空平滑约束。
- `L_extreme`：极端降水加权损失。

### 2.2 为什么不建议做重型 PINN

台风降水的严格物理过程涉及水汽输送、热力学、三维风场、垂直速度、温湿度、边界层过程等。若没有完整三维气象再分析数据，仅靠台风路径、降水图和地形数据，很难构造严格的水汽守恒或 Navier-Stokes 类 PDE 残差。

因此本项目更适合做轻量但可信的 Physics-informed 模型：

- 使用 DEM、地形梯度、海陆掩膜作为物理输入。
- 使用地形抬升约束作为物理损失。
- 使用降水非负性和时空平滑作为基本物理一致性约束。
- 使用极端降水加权损失提升暴雨峰值和暴雨面积预测能力。

---

## 3. 推荐模型：PI-ResConvLSTM

### 3.1 输入设计

建议输入过去 K 帧信息：

```text
历史降水序列：P_{t-K+1}, ..., P_t
台风中心相对坐标：dx, dy
台风移动速度：u_move, v_move
中心最低气压：P_c
最大风速：V_max
风速半径/强度等级：R_wind, category
登陆状态：landfall flag
DEM 地形高度：h
地形梯度：dh/dx, dh/dy
海陆掩膜：land-sea mask
距离台风中心半径：r
相对方位角：θ
```

输入张量可以组织为：

```text
X_t ∈ R^{K × C × H × W}
```

其中：

- `K`：历史时间步数量，例如 6、8、11。
- `C`：输入通道数。
- `H × W`：降水空间网格，例如 128 × 128。

### 3.2 输出设计

不建议直接预测完整降水场 `P_{t+1}`。推荐使用残差学习：

```text
ΔP_hat = Model(X_t)
P_hat_{t+1} = P_t + ΔP_hat
```

优势：

- 降低直接预测完整降水场的难度。
- 更符合降水场连续演化特征。
- 更容易学习台风降水雨带移动、增强、衰减的变化量。
- 对短序列、小样本更友好。

---

## 4. 物理约束设计

### 4.1 降水预测误差 L_rain

基础版本：

```text
L_rain = MSE(P_hat, P_true)
```

更推荐：对强降水区域加权。

```text
L_rain = mean(w(P_true) · (P_hat - P_true)^2)
```

权重函数示例：

```text
w(P_true) = 1 + α · I(P_true > threshold)
```

目的：避免模型只预测大片小雨，而忽略极端暴雨核心。

### 4.2 非负性约束 L_nonneg

降水不能为负：

```text
L_nonneg = mean(ReLU(-P_hat)^2)
```

如果模型最后输出层已经使用 ReLU 或 Softplus，这一项可以降低权重或作为安全约束。

### 4.3 地形抬升约束 L_oro

地形抬升是最适合本项目的物理先验。可以近似构造：

```text
O = u · dh/dx + v · dh/dy
```

其中：

- `u, v`：台风移动方向或近似低层气流方向。
- `h`：DEM 地形高度。
- `dh/dx, dh/dy`：地形坡度。
- `O`：迎风坡抬升强度。

物理含义：当气流迎坡上升，水汽更容易凝结形成强降水。

推荐软约束：

```text
mask_oro = I(O > percentile_80(O))
L_oro = mean(mask_oro · ReLU(τ - P_hat)^2)
```

解释：在地形抬升强的区域，模型不应完全预测为无雨或极弱雨。

更稳的版本：

```text
L_oro = - corr(P_hat, ReLU(O))
```

或：

```text
L_oro = MSE(normalize(P_hat · mask_oro), normalize(ReLU(O) · mask_oro))
```

注意：不要强行要求降水完全等于地形抬升项。地形抬升只是降水增强因子之一，不是全部机制。

### 4.4 弱时空平滑约束 L_smooth

降水场不应像噪声一样剧烈跳变：

```text
L_spatial = mean(|∂P_hat/∂x| + |∂P_hat/∂y|)
L_temporal = mean(|P_hat_{t+1} - P_t|)
```

但平滑权重要小，否则会抹平极端降水峰值。

### 4.5 极端降水约束 L_extreme

极端降水是论文重点，建议单独强调。

```text
L_extreme = MSE(P_hat[heavy_rain_mask], P_true[heavy_rain_mask])
```

heavy_rain_mask 可按阈值定义：

```text
P_true > 10 mm/h
P_true > 20 mm/h
P_true > 50 mm/h
```

根据数据单位和任务设定调整。

---

## 5. Attention 要不要加

### 5.1 结论

Attention 可以加，但不要作为第一主线。当前项目已经有多模态输入、残差学习、物理约束、极端降水损失、情景推演和可解释性分析。如果再加入复杂 Attention，容易显得模块堆叠。

推荐原则：

```text
先证明 Physics-informed loss 有用。
再把 Attention 作为可选消融模块。
```

### 5.2 最推荐的 Attention：Channel Attention

因为输入通道很多，通道注意力最自然。

目标：让模型自动学习不同变量的重要性，例如：

- 降水历史帧。
- 中心气压。
- 最大风速。
- DEM。
- 地形梯度。
- 海陆掩膜。
- 距离台风中心。
- 台风移动方向。

推荐模块：SE Block / Channel Attention。

表达方式：

```text
ConvLSTM hidden feature
↓
Global Average Pooling
↓
MLP / sigmoid
↓
Channel weights
↓
Reweighted feature
```

论文里可以写成：

**A lightweight channel attention module is introduced to adaptively recalibrate meteorological and topographic feature channels.**

### 5.3 不建议优先使用复杂时空 Transformer

原因：

- 样本量可能不足。
- 训练成本高。
- 模块解释难度变大。
- 审稿人可能认为堆模型。
- ConvLSTM 本身已经建模时间依赖。

因此 Attention 只作为增强模块，不作为核心卖点。

---

## 6. 最终推荐实验设计

### 6.1 Baseline 对比

至少需要以下 baseline：

```text
1. Persistence：直接使用上一时刻降水作为下一时刻预测。
2. ConvLSTM：基础时空预测模型。
3. ResConvLSTM：加入残差学习。
4. ResConvLSTM + DEM：加入地形输入。
5. PI-ResConvLSTM：加入物理启发损失。
6. PIA-ResConvLSTM：加入轻量 Channel Attention，可选。
```

### 6.2 消融实验

重点消融：

```text
去掉残差学习
去掉 DEM
去掉地形梯度
去掉海陆掩膜
去掉地形抬升损失
去掉极端降水加权损失
去掉 Channel Attention
```

### 6.3 评价指标

不要只用 MSE/RMSE。建议至少包含：

```text
MAE
RMSE
SSIM
CSI / Threat Score
POD 命中率
FAR 虚警率
HSS
峰值降水误差
暴雨面积误差
降水中心偏移误差
极端降水阈值命中率
```

### 6.4 训练/验证划分

推荐：

```text
2014-2022：训练
2023：验证
2024：测试
```

或者：

```text
Leave-one-typhoon-out cross validation
```

期刊论文必须避免只展示少数个例，应该尽量做多台风独立测试。

---

## 7. 情景推演设计

可构造 4 类虚拟台风情景：

```text
1. 路径北抬：V-PATH-NORTH
2. 中心气压降低：V-LOW-PRESSURE
3. 风速增强：V-WIND-INTENSE
4. 热力增强：V-THERMAL
```

每类情景需要输出：

- 峰值降水变化。
- 暴雨面积变化。
- 持续时间变化。
- 降水中心偏移。
- 高风险区域空间分布。

重点不要只展示漂亮图，要给出可量化指标。

---

## 8. 和 DMOEA 的关系

### 8.1 台风项目

优势：

- 应用价值强。
- 可视化强。
- 有北京市一等奖背书。
- 更适合短期转论文。
- 更适合地理/气象/遥感 + AI 交叉方向。
- 对港硕申请展示非常直观。

风险：

- ConvLSTM 本身不新。
- 必须补 baseline、独立测试和专业气象指标。
- PINN 表述不能过度夸张。

### 8.2 DMOEA 项目

优势：

- 算法原创性更高。
- 你是负责人。
- 更适合智能优化/进化计算方向。
- 长期论文上限更高。

风险：

- SOTA 对比难。
- 消融和统计检验要求高。
- 算法期刊审稿会更严格。

### 8.3 最终定位

```text
短期论文转化：台风 PI-ResConvLSTM 优先。
长期算法科研：DMOEA 继续打磨。
申请材料：两者都保留，分别代表应用科研和算法科研。
```

---

## 9. 简历/申请包装

推荐中文表达：

**北京市高校数学建模一等奖，构建基于多源数据同化与物理启发残差 ConvLSTM 的台风极端降水时空推演模型，融合台风路径、强度、卫星降水、DEM 地形与海陆掩膜数据，实现降水场预测、未来气候情景模拟与关键因子可解释性分析。**

推荐英文表达：

**Won the Beijing Intercollegiate Mathematical Modeling First Prize by developing a physics-informed residual ConvLSTM framework for typhoon precipitation simulation. The model integrates typhoon trajectory, intensity, satellite precipitation, DEM terrain, and land-sea mask data to predict spatiotemporal rainfall fields and simulate future climate perturbation scenarios.**

---

## 10. 给 Agent 的执行任务

```text
请基于现有台风降水数学建模项目，重构为可发表论文方向的 PI-ResConvLSTM 框架。

目标：
将原竞赛模型升级为 Physics-informed Residual ConvLSTM，用于台风降水时空预测和未来情景推演。

一、数据处理
1. 读取 CMABST 台风路径数据和 GPM 降水时空分布数据。
2. 将降水图统一重采样到 128×128 网格。
3. 对齐台风路径时间戳和 GPM 降水时间戳。
4. 构造过去 K 帧输入，预测下一时刻降水图。
5. 加入 DEM、地形梯度、海陆掩膜、台风中心相对坐标、移动速度、中心气压、最大风速等通道。

二、模型实现
1. 实现 ConvLSTM baseline。
2. 实现 Residual ConvLSTM：预测 ΔP，然后 P_hat = P_t + ΔP。
3. 实现 PI-ResConvLSTM：加入物理启发损失。
4. 可选实现 Channel Attention，用于通道权重自适应。

三、物理损失
1. L_rain：降水预测误差，建议使用 Weighted MSE。
2. L_nonneg：降水非负性约束。
3. L_oro：地形抬升一致性约束，基于 O = u·dh/dx + v·dh/dy。
4. L_smooth：弱时空平滑约束。
5. L_extreme：极端降水区域加权误差。

四、实验设计
1. 对比 Persistence、ConvLSTM、ResConvLSTM、ResConvLSTM+DEM、PI-ResConvLSTM、PIA-ResConvLSTM。
2. 做消融实验：去掉残差、去掉 DEM、去掉地形梯度、去掉物理损失、去掉 attention。
3. 使用 MAE、RMSE、SSIM、CSI、POD、FAR、HSS、峰值误差、暴雨面积误差、降水中心偏移误差进行评估。
4. 对强降水阈值分层评估。
5. 输出每个测试台风的可视化降水图、误差图和时间序列曲线。

五、情景推演
1. 构造路径北抬、中心气压降低、风速增强、热力增强四类虚拟台风情景。
2. 输出对应降水分布、峰值降水、暴雨面积、持续时间和高风险区域。

六、论文输出
1. 生成英文摘要。
2. 生成方法流程图。
3. 生成模型结构图。
4. 生成消融实验表。
5. 生成情景推演结果图。
6. 生成论文初稿目录。
```

---

## 11. 最终建议

不要把项目写成模块堆砌。最稳的主线是：

```text
Residual ConvLSTM
+ 多源气象/地理输入
+ 地形抬升物理约束
+ 极端降水加权损失
+ 情景推演
```

Attention 可以作为加分项，但不是必须。最重要的是补齐 baseline、消融实验、专业气象指标和独立测试集。

最终论文主张：

**本文提出一种物理启发式残差 ConvLSTM 框架，通过融合台风动力特征、卫星降水、地形 DEM 与地形抬升约束，提高台风极端降水时空推演的物理一致性和极端事件预测能力。**
