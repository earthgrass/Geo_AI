# RESEARCH_READY_AUDIT — 研究就绪审计

> 生成日期：2026-08-14
> 范围：Final Data Validation → Coverage → Split → Production Build → Normalization → Experiment-Ready Freeze

## 1. Repository state
- HEAD: `b87f6fe`（本轮提交前）
- 分支 main，远程 `https://github.com/earthgrass/Geo_AI.git`

## 2. Tests
- `python -m pytest tests/` → **27 passed**
- 本轮修复的测试脚本 bug：3 个陈旧断言 + 2 个 h5py 键名/索引 off-by-one；新增 3 个回归测试（因果地理合法性、经度归一化、日界线 u_move）。

## 3. GPM grid audit
- 扫描 400 文件：CRS 唯一（EPSG:4326）、分辨率唯一（0.1°）、尺寸唯一（201×201）、nodata 唯一（-9999）。
- transform 唯一 400 个（**每文件是台风中心瓦片**，非单一全局网格）。
- 处理：builder 改用逐文件 `rasterio.warp.reproject` 到规范北向锚点网格。

## 4. Coverage audit
- 302 个 CMA 台风，110 个有 TIF 数据，**36 个有可用窗口**。
- EXACT == T15（无 ±15min 系统偏移）→ **MATCH_POLICY = EXACT**。

## 5. Timestamp policy
- **EXACT**（tolerance=0）。GPM 时间戳与 CMA 0.5h 网格精确对齐，近邻匹配无收益。

## 6. Event split（`configs/splits_v1.yaml`）
- train=25 / val=7 / test=4（时间外推，交集为空）。
- 样本数：train=4894 / val=1266 / test=707。

## 7. Production HDF5
- `ConvLSTM_Dataset_128.h5`：6867 样本，~1.2GB，schema v2（12 通道）。

## 8. Dataset validation
- `validate_paper_dataset.py` → **PASS（67 项）**。

## 9. Normalization
- `configs/normalization_v1.json`：train-only 拟合，降水 + track + terrain 统计齐全，land_mask 不归一化。

## 10. Model input smoke test
- train/val/test 均能加载，X `[11,12,128,128]`、Y `[1,128,128]`。
- Persistence / PlainConvLSTM / ResConvLSTM / PIResConvLSTM 前向均输出 `[1,1,128,128]`，张量兼容。

## 11. Experiment configs prepared
- `configs/experiments/01-06`：E0-E5 实验矩阵（channel 选择为训练阶段 TODO）。

## 12. Remaining blockers
- **无阻塞项。** 下一步是文献调研，决定是否引入 ERA5 环境风场、最终实验矩阵与 research gap。

---

## Final classification

# RESEARCH_READY

所有关键检查通过：测试全绿、GPM 网格预检通过、覆盖已理解、事件划分已冻结、生产 HDF5 已构建、验证器通过、归一化仅用 train、模型前向冒烟通过。
