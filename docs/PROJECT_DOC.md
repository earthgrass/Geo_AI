# GeoAI: 物理启发残差ConvLSTM台风极端降水时空推演

## 项目总览文档

> **来源**: 2026年北京高校数学建模校际联赛 B题 一等奖
> **转化方向**: GeoAI / 气象AI / 遥感降水建模 / 自然灾害风险模拟
> **论文题目**: Physics-informed Residual ConvLSTM for Spatiotemporal Typhoon Precipitation Simulation
> **文档版本**: v2.0 | **更新日期**: 2026-07-04

---

## 目录

1. [项目概述](#1-项目概述)
2. [数据源说明](#2-数据源说明)
3. [方法论与Pipeline](#3-方法论与pipeline)
4. [模型架构](#4-模型架构)
5. [关键结果](#5-关键结果)
6. [文件索引](#6-文件索引)
7. [环境配置](#7-环境配置)
8. [已知问题与改进方向](#8-已知问题与改进方向)
9. [论文转化路线图](#9-论文转化路线图)

---

## 1. 项目概述

### 1.1 赛题背景

台风是我国沿海地区最重要的气象灾害之一。准确预测台风的路径演变、强度变化及降水时空分布，对防灾减灾具有重要意义。本赛题要求基于历史台风最佳路径数据和GPM卫星降水遥感影像，建立数学模型揭示台风"路径—强度—环境—降水"的内在关联机制。

### 1.2 研究目标

| 问题 | 核心任务 | 方法 | 状态 |
|------|----------|------|:----:|
| Q1 | 分析台风路径/强度与降水特征的定量关系 | Spearman + RF + SHAP | ✅ |
| Q2 | 预测KONG-REY(2024)和MAN-YI(2024)降水时空分布 | Spatial Residual ConvLSTM | ✅ |
| Q3 | 构造虚拟台风情景，分析极端降水趋势 | 多维扰动 + 敏感性分析 | ✅ |

### 1.3 核心方法

**PI-ResConvLSTM** (Physics-informed Residual ConvLSTM) — 一种融合物理先验与深度学习的台风降水时空预测框架：

- **数据驱动**: ConvLSTM捕捉降水场的时空依赖关系
- **物理增强**: DEM地形、海陆掩膜、Coriolis力注入网络
- **残差学习**: 预测降水变化量 ΔP，降低学习难度
- **多模态融合**: 台风路径(1D) + 卫星降水(2D) + 地形(2D) 在通道维度统一

---

## 2. 数据源说明

### 2.1 台风最佳路径数据 (CMABSTdata/)

| 属性 | 说明 |
|------|------|
| 来源 | 中国气象局(CMA)最佳路径数据集 |
| 覆盖 | 2014-2025年，每年一个TXT文件 |
| 时间分辨率 | 原始6小时 → 三次样条插值至0.5小时 |
| 关键字段 | 时间、纬度(0.1°)、经度(0.1°)、中心气压(hPa)、最大风速(m/s) |
| 文件大小 | ~500KB (全部年份) |

### 2.2 卫星降水影像 (TIFdata/)

| 属性 | 说明 |
|------|------|
| 来源 | NASA GPM IMERG (V07B) |
| 格式 | GeoTIFF, 以台风中心裁剪 |
| 空间分辨率 | ~0.1° (约10km网格) |
| 时间分辨率 | 30分钟 |
| 覆盖 | 110个台风事件，30,000+个TIF文件 |
| 文件名格式 | `3B-HHR-E.MS.MRG.3IMERG.YYYYMMDD-SHHMMSS-EHHMMSS.V07B_*.tif` |

### 2.3 全球地形数据 (Global_DEM.tif)

| 属性 | 说明 |
|------|------|
| 来源 | ETOPO1 / SRTM |
| 分辨率 | 60弧秒 (~1.8km) |
| 用途 | 地形抬升效应分析、海陆掩膜生成 |
| 文件大小 | ~445MB |

### 2.4 降水强度分级标准 (CMA)

| 等级 | 名称 | 强度范围 (mm/h) | 特征名 |
|:----:|------|:---------------:|--------|
| 1 | 小雨 | 0.1 - 2.0 | Area_Light |
| 2 | 中雨 | 2.0 - 5.0 | Area_Moderate |
| 3 | 大雨 | 5.0 - 10.0 | Area_Heavy |
| 4 | 暴雨 | 10.0 - 20.0 | Area_Torrential |
| 5 | 大暴雨/特大暴雨 | ≥ 20.0 | S_ext_Extreme |

---

## 3. 方法论与Pipeline

### 3.1 整体流程

```
┌──────────────────────────────────────────────────────────────┐
│                    Phase 1: 数据处理与分析 (Q1)                 │
├──────────────────────────────────────────────────────────────┤
│  CMABSTdata/*.txt         TIFdata/*/*.tif                    │
│       │                        │                              │
│       ▼                        ▼                              │
│  step1.1: 轨迹解析       step1.2: 降水特征提取                 │
│  三次样条插值→0.5h        CMA分级面积+质心                      │
│       │                        │                              │
│       └────────┬───────────────┘                              │
│                ▼                                              │
│         step1.3: 多模态融合                                    │
│         时空对齐 + 派生特征 + 交叉特征                          │
│                │                                              │
│                ▼                                              │
│         step1.4: 统计分析                                      │
│         Spearman + RF重要性 + SHAP解释                         │
│                │                                              │
│                ▼                                              │
│         step1.5: 可视化 → Figures1/                           │
├──────────────────────────────────────────────────────────────┤
│                 Phase 2: 深度学习预测 (Q2)                      │
├──────────────────────────────────────────────────────────────┤
│  step2.1: HDF5时空数据集构建                                    │
│  物理场渲染(Wind/Pressure/DEM/Dist) + 滑窗构造                 │
│                │                                              │
│                ▼                                              │
│  step2.2: GPU训练 (RTX 4090)                                  │
│  Spatial Residual ConvLSTM, AMP, 20 Epochs                   │
│                │                                              │
│                ▼                                              │
│  step2.3: 物理推断引擎                                         │
│  PINN-Inference: Physics prior + NN residual                  │
│                │                                              │
│                ▼                                              │
│  step2.4: 降水热力图可视化 → Figures2/                         │
├──────────────────────────────────────────────────────────────┤
│              Phase 3 & 4: 虚拟台风与敏感性分析 (Q3)             │
├──────────────────────────────────────────────────────────────┤
│  step3.1: 虚拟台风生成 (SHIFT/INTENSE/COMPOUND/SLOW)          │
│       ↓                                                       │
│  step2.3复用: 对虚拟台风推理                                   │
│       ↓                                                       │
│  step3.2: 量化指标计算 (P_max, S_ext, Duration)               │
│       ↓                                                       │
│  step4.x: 可视化 + 敏感性分析 + 中国沿海风险图                   │
└──────────────────────────────────────────────────────────────┘
```

### 3.2 特征工程体系

#### 输入特征 (10维)

| 类别 | 特征名 | 含义 | 单位 | 来源 |
|------|--------|------|------|------|
| 位置 | Lat, Lon | 纬度/经度 | °N/°E | 三次样条插值 |
| 强度 | Wind_Speed | 最大风速 | m/s | 三次样条插值 |
| 强度 | Pressure | 中心气压 | hPa | 三次样条插值 |
| 结构 | Radius_max_wind_km | 最大风速半径 | km | Willoughby(2006) |
| 运动 | Moving_Speed_kmh | 移动速度 | km/h | Haversine差分 |
| 运动 | Moving_Direction | 移动方向 | deg | 方位角计算 |
| 形态 | Curvature_deg_per_km | 路径曲率 | deg/km | 方位角差分 |
| 变化 | Delta_P_6h | 6h气压变化 | hPa | 时序差分 |
| 变化 | Delta_V_6h | 6h风速变化 | m/s | 时序差分 |

#### 输出特征 (5维)

| 特征名 | 含义 | 单位 |
|--------|------|------|
| P_total | 总降水量 | mm |
| P_max | 最大降水强度 | mm/h |
| S_ext_Extreme_over_20 | 极端降水面积 | km² |
| D_offset_km | 降水中心偏移距离 | km |
| I_asy_Index | 降水非对称性指数 | 无量纲 |

---

## 4. 模型架构

### 4.1 Spatial Residual ConvLSTM

```
Input: [B, T=11, C=4, H=128, W=128]
  4通道: 降水场 + 物理风场 + 物理气压场 + 距离矩阵
         │
         ▼
  ┌─────────────────────────┐
  │  Encoder ConvLSTM #1    │  hidden=64, kernel=3×3
  │  return_sequences=True  │
  └────────────┬────────────┘
               │
               ▼
  ┌─────────────────────────┐
  │  Encoder ConvLSTM #2    │  hidden=128, kernel=3×3
  │  return_sequences=False │
  └────────────┬────────────┘
               │  h2 [B, 128, 128, 128]
               ▼
  ┌─────────────────────────┐
  │  Decoder                │  Conv2D(128→64) → BN → ReLU
  │  Pred Head              │  Conv2D(64→1)
  └────────────┬────────────┘
               │  p_pred [B, 1, 128, 128]
               ▼
  ┌─────────────────────────┐
  │  Residual Network       │  [h2 + p_pred] → Conv(32) → Conv(1)
  └────────────┬────────────┘
               │  ΔP
               ▼
          P_final = ReLU(p_pred + ΔP)
```

### 4.2 训练配置

| 参数 | 值 |
|------|-----|
| 输入序列长度 | 11帧 (历史) |
| 输出 | 1帧 (下一时刻降水) |
| Batch Size | 8 |
| Optimizer | AdamW (lr=1e-3, weight_decay=1e-4) |
| Loss | MSE |
| 混合精度 | AMP (GradScaler) |
| Epochs | 20 |
| 最佳Val Loss | 1.44 (Epoch 9) |
| 训练硬件 | NVIDIA RTX 4090 (~7h) |

### 4.3 物理推断引擎 (step2_3)

推理时将物理先验与神经网络输出融合：

```
P_final = physics_rain(wind, pressure, DEM, Coriolis) + NN_residual × 1.5
```

其中物理降水场由以下经验公式构建：
- 风场: 内核线性 + 外核指数衰减
- 气压场: 高斯衰减 (σ=300km)
- Coriolis非对称: `1 + f·cos(θ)` 修正
- 地形抬升: `1 + (DEM/1000) × 0.35`
- 陆地摩擦: `×1.20` (陆地) / `×1.0` (海洋)
- 基准降水: `(wind/5.0)^1.2`

> ⚠️ **注意**: 上述物理公式缺乏严格文献支撑，`×1.5`缩放因子为硬编码。这些问题将在PI-ResConvLSTM重构中修复。

---

## 5. 关键结果

### 5.1 Q1: 特征分析发现

| 发现 | 方法 | 结论 |
|------|------|------|
| 风速 ↔ 极端降水面积 | Spearman ρ > 0.6 | 强正相关，风速是极端降水主导因子 |
| 路径曲率阈值效应 | SHAP依赖曲线 | 曲率超过阈值时降水模式突变 |
| Delta_P_6h预测价值 | RF重要性Top3 | 6h变化量优于静态气压 |
| 降水非对称性 | 象限分析 | 降水中心偏向台风移动方向右前象限 |

### 5.2 Q2: 预测精度

| 台风 | P_max (mm/h) | S_ext (km²) | Duration (步) |
|------|:-----------:|:-----------:|:------------:|
| KONG-REY | 37.20 | 1,458,800 | 30 |
| MAN-YI | 35.11 | 1,305,100 | 36 |

> ⚠️ KONG-REY和MAN-YI在GPM中无实测降水，上述数值为模型预测，无法与真值验证。

### 5.3 Q3: 气候情景敏感性

| 情景 | P_max (mm/h) | S_ext (km²) | 相对基准变化 |
|------|:-----------:|:-----------:|:------------:|
| KONG-REY (基准) | 37.20 | 1,458,800 | — |
| KONG-REY (无地形) | 29.42 | 1,414,100 | P_max -26.4% |
| V-INTENSE (强度+15%) | 45.26 | 1,738,800 | S_ext +19.2% |
| V-HIGHLAT (高纬) | 49.41 | 1,608,300 | P_max +32.8% |

> 地形消融实验证明DEM对极端降水有显著放大效应（P_max降低26.4%）。

---

## 6. 文件索引

### 6.1 代码脚本

| 文件 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `step1.1 data process.py` | 台风轨迹解析+三次样条插值 | CMABSTdata/*.txt | All_Years_Typhoon_Features.csv |
| `step1.2 tif process.py` | TIF降水影像特征提取 | TIFdata/*/*.tif | TIF_Features_Base.csv |
| `step1.3 data integration.py` | 多模态数据融合+派生特征 | 上述两CSV | Typhoon_Full_Dataset_Q1.csv |
| `step1.4 data analysis.py` | Spearman/RF/SHAP分析 | Q1融合数据 | 相关性矩阵/SHAP值 |
| `step1.5 visualization.py` | Q1综合可视化 | 分析结果 | Figures1/*.png |
| `convLSTM_model.py` | ConvLSTM网络架构定义 | — | — |
| `step2.1_spatial_dataloader.py` | HDF5时空数据集构建 | Q1数据+TIF | ConvLSTM_Dataset_128.h5 |
| `step2_2_train_cloud.py` | GPU训练脚本 | HDF5数据集 | typhoon_convlstm_best.pth |
| `step2_3_generate_metrix.py` | 物理推断引擎 | 模型权重+DEM+轨迹 | *_DataPackage.npz |
| `step2_4_rainfall_heatmap.py` | 降水热力图可视化 | DataPackage | Figures2/*.png |
| `step3_1_virtural_generator.py` | 虚拟台风场景生成 | KONG-REY轨迹 | Virtual_Typhoons_2026.txt |
| `step3_2_metrics_table.py` | 量化指标计算 | DataPackage | Sensitivity_Analysis_Results.csv |
| `step3_4_china_rainfall.py` | 中国沿海降水分布 | DataPackage | Fig4_*.png |
| `step4_1_virtual_spatiotemporal.py` | 宏观尺度虚拟台风演变 | DataPackage | Fig4_*.png |
| `step4_2_china_landfall_map.py` | 中国沿海登陆风险图 | DataPackage | Fig4_5_*.png |
| `step4_2_china_rainfall_map.py` | 中国沿海降水分布图 | DataPackage | Fig4_*.png |
| `step4_3_sensitivity_impact.py` | 单因素敏感性分析 | DataPackage | Fig4_6_*.png |

### 6.2 数据文件

| 文件 | 格式 | 大小 | 说明 |
|------|------|------|------|
| `All_Years_Typhoon_Features.csv` | CSV | ~39MB | 2014-2025全部台风特征(0.5h) |
| `TIF_Features_Base.csv` | CSV | ~8MB | TIF降水特征提取 |
| `Typhoon_Full_Dataset_Q1.csv` | CSV | ~3.5MB | Q1时空融合数据集 |
| `Typhoon_Features_Complete.csv` | CSV | — | 补充版完整特征 |
| `ConvLSTM_Dataset_128.h5` | HDF5 | ~3.9GB | 时空张量训练集 |
| `typhoon_convlstm_best.pth` | PyTorch | ~4.4MB | 最佳模型权重 |
| `*_DataPackage.npz` | NumPy | ~3-5MB/个 | 降水预测场 |
| `train_log.txt` | TXT | — | 训练日志 |
| `Sensitivity_Analysis_Results.csv` | CSV | — | 敏感性分析结果 |

### 6.3 图表输出

| 目录 | 内容 | 数量 |
|------|------|:--:|
| `Figures1/` | Q1分析图表(热力图/SHAP/雷达图等) | 16张 |
| `Figures2/` | Q2降水预测热力图(含逐帧) | 大量 |
| `Results_Figures/` | Q1早期版本(英文标签) | 14张 |
| 根目录散落 | 各阶段图表 | ~40张 |

### 6.4 文档与论文

| 文件 | 说明 |
|------|------|
| `essay/essay.tex` | LaTeX论文源码 |
| `essay/essay.pdf` | 编译后论文PDF (~27MB) |
| `项目技术文档_README.md` | 详细技术文档(含公式推导) |
| `第一部分工作汇报.md` | 面向零基础队员的通俗解读 |
| `项目评估报告.md` | 客观评估报告(方法论/模型/代码/可信度) |
| `Typhoon_PI_ResConvLSTM_Summary_for_Agent.md` | PI-ResConvLSTM转化蓝图 |

---

## 7. 环境配置

### 7.1 必需环境

- **Python** ≥ 3.8 (推荐 3.10+)
- **CUDA** 11.8+ (GPU训练需要，RTX 4090推荐)

### 7.2 依赖安装

```bash
# 核心科学计算
pip install pandas numpy scipy scikit-learn

# 可视化
pip install matplotlib seaborn cartopy

# 地理空间数据处理
pip install rasterio

# 深度学习
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 可解释性
pip install shap

# 数据处理
pip install h5py tqdm
```

### 7.3 数据准备

1. 确保 `CMABSTdata/` 包含2014-2025年的CMA最佳路径TXT文件
2. 确保 `TIFdata/` 包含GPM IMERG降水GeoTIFF影像
3. 下载 `Global_DEM.tif` (ETOPO1, 约445MB)
4. 运行 `step1.1` → `step1.2` → `step1.3` → `step1.4` 生成中间数据

### 7.4 运行顺序

```
Step 1-5:  Q1数据处理与分析 (必须顺序执行)
Step 6:    step2.1 生成HDF5数据集 (~2h, ~4GB输出)
Step 7:    step2.2 GPU训练 (RTX 4090, ~7h/20epochs)
Step 8:    step2.3 推理生成降水预测
Step 9:    step2.4 可视化
Step 10-13: Q3虚拟台风与敏感性分析
```

---

## 8. 已知问题与改进方向

### 8.1 致命问题 🔴

| # | 问题 | 影响 |
|---|------|------|
| 1 | **损失函数无物理约束**: 声称"PINN"但仅使用纯MSE | 模型未学习物理规律 |
| 2 | **残差学习不一致**: 训练预测绝对值，推理用作残差修正 | 训推不一致 |
| 3 | **数据泄漏**: 随机80/20切分，同台风帧跨训练/验证集 | 验证指标虚高 |
| 4 | **Q2无精度验证**: KONG-REY/MAN-YI无实测降水 | 结论可信度不足 |

### 8.2 重要问题 🟡

| # | 问题 | 位置 |
|---|------|------|
| 5 | 物理降水公式 `(wind/5)^1.2` 无文献支撑 | step2_3:74 |
| 6 | 推理阶段 `×1.5` 硬编码缩放 | step2_3:146 |
| 7 | 两个Rmax公式冲突 (Willoughby vs 经验公式) | step1.1 / step2_3 |
| 8 | 模型架构重复定义3处 | convLSTM_model.py / step2_2 / essay/ |
| 9 | 自回归误差累积无量化分析 | step2_3:151 |
| 10 | 虚拟台风V-HIGHLAT/V-STAGNANT/V-DOOMSDAY来源不明 | step3_1 |

### 8.3 一般问题 🟢

| # | 问题 |
|---|------|
| 11 | 绝对路径硬编码 (`C:\Users\champ\Desktop\...`) |
| 12 | 文件名引用不一致 (`KONG-REY_Data.npz` vs `KONG-REY_DataPackage.npz`) |
| 13 | 无PyTorch seed锁定 |
| 14 | S_ext单位在不同脚本中不一致 |
| 15 | `__pycache__/` 和 `__MACOSX/` 垃圾文件 |
| 16 | 文件命名混乱 (点号/下划线/空格混用) |
| 17 | 无 `requirements.txt` |
| 18 | 无单元测试 |

---

## 9. 论文转化路线图

### 9.1 推荐论文方向

**题目**: Physics-informed Residual ConvLSTM for Spatiotemporal Typhoon Precipitation Simulation under Climate Scenario Perturbations

**核心贡献**:
1. 提出PI-ResConvLSTM框架，将物理先验以结构化方式注入深度学习
2. 设计5项物理损失函数（非负、地形、平滑、极端降水加权）
3. 建立完整的baseline对比体系（Persistence → ConvLSTM → ResConvLSTM → PI-ResConvLSTM）
4. 多维度虚拟台风气候情景推演

### 9.2 改进步骤

详见 `docs/CODE_CHANGE_PLAN.md` 和 `docs/SUMMARY.md`。

**概要**:
1. **模型重构**: 实现真正Physics-informed损失函数
2. **数据修复**: 按时序/台风分组划分train/val/test
3. **评估完善**: 加入SSIM, CSI, POD, FAR, HSS等气象指标
4. **Baseline补齐**: Persistence, ConvLSTM, ResConvLSTM等
5. **推理修正**: 移除硬编码缩放因子，统一训推范式

---

*文档版本: v2.0 | 更新日期: 2026-07-04*
