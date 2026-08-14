# GeoAI 代码改动方案

## PI-ResConvLSTM 完整重构计划

> 基于 `Typhoon_PI_ResConvLSTM_Summary_for_Agent.md` 蓝图
> 目标: 将竞赛代码转化为可发表的GeoAI论文级代码框架

---

## 目录

1. [改动总览](#1-改动总览)
2. [模型架构改动](#2-模型架构改动)
3. [物理损失函数](#3-物理损失函数)
4. [数据管道改动](#4-数据管道改动)
5. [训练框架改动](#5-训练框架改动)
6. [评估框架新建](#6-评估框架新建)
7. [推理引擎改动](#7-推理引擎改动)
8. [文件对照表](#8-文件对照表)
9. [执行优先级](#9-执行优先级)

---

## 1. 改动总览

### 1.1 从"竞赛代码"到"论文代码"的核心变化

| 维度 | 当前竞赛代码 | 目标论文代码 |
|------|-------------|-------------|
| **模型** | SpatialResidualConvLSTM (内部残差) | PI-ResConvLSTM (时序残差 ΔP) |
| **输入** | 4通道 (降水+风场+气压+距离) | ~13通道 (降水+dx+dy+u+v+Pc+Vmax+RMW+DEM+dh/dx+dh/dy+landmask+r+θ) |
| **损失** | 纯MSE | L_total = L_rain + λ1·L_nonneg + λ2·L_oro + λ3·L_smooth + λ4·L_extreme |
| **残差** | 内部残差 p_pred+Δp | 时序残差 P_t+ΔP |
| **Attention** | 无 | 可选SE Channel Attention |
| **数据划分** | 随机80/20 | 2014-2022 train / 2023 val / 2024 test |
| **Baseline** | 无 | Persistence, ConvLSTM, ResConvLSTM, ResConvLSTM+DEM |
| **评估** | 仅MSE | MAE, RMSE, SSIM, CSI, POD, FAR, HSS, 峰值/面积/中心误差 |
| **推理** | base_rain + NN_residual × 1.5 | 纯模型输出，无硬编码缩放 |
| **路径** | 绝对路径硬编码 | config.py / argparse |
| **可复现** | 无seed | 全seed锁定 |

### 1.2 文件改动汇总

| 操作 | 数量 | 说明 |
|------|:----:|------|
| **新建** | ~15个.py | src/ 下全部新代码 |
| **归档** | ~30个文件 | 旧脚本移至 archive/competition/old_scripts/ |
| **保留引用** | ~10个 | 数据文件(.h5/.npz/.csv)保留原位 |
| **删除** | 4项 | __MACOSX/, __pycache__/, ~$*, B.zip |

---

## 2. 模型架构改动

### 2.1 核心改动: 从内部残差到时序残差

**当前** (`convLSTM_model.py`):
```python
# 模型输出 = 内部残差修正
p_pred = pred_head(decoder(h2))
delta_p = residual_net(cat(h2, p_pred))
output = ReLU(p_pred + delta_p)  # 内部残差
```

**目标** (`src/models/pi_res_convlstm.py`):
```python
# 模型输出 = 降水变化量
delta_p = Model(input_sequence)
output = ReLU(last_precip_frame + delta_p)  # 时序残差: P_hat = P_t + ΔP
```

**区别**:
- 旧: 模型学习"修正自己的预测" (self-correction)
- 新: 模型学习"降水如何从前一帧变化" (temporal delta) — 任务更简单、物理更合理

### 2.2 新建文件: `src/models/convlstm_cell.py`

从旧 `convLSTM_model.py` 提取ConvLSTMCell类，保持核心逻辑不变：

```python
class ConvLSTMCell(nn.Module):
    """标准ConvLSTM单元 (Shi et al., 2015)"""
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        # 输入门、遗忘门、输出门、候选状态 — 卷积实现
        # 支持peephole连接 (W_ci ⊙ C_{t-1})
    
    def forward(self, x, cur_state):
        # 返回 (h_next, c_next)
    
    def init_hidden(self, batch_size, image_size):
        # 返回 (h0, c0)
```

### 2.3 新建文件: `src/models/channel_attention.py`

SE-Net风格的通道注意力：

```python
class ChannelAttention(nn.Module):
    """SE Block: 自适应学习多模态通道重要性"""
    def __init__(self, n_channels, reduction=16):
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(n_channels, n_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(n_channels // reduction, n_channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: [B, C, H, W] -> [B, C, H, W] (reweighted)
```

### 2.4 新建文件: `src/models/pi_res_convlstm.py`

核心模型，支持配置化的架构：

```python
class PIResConvLSTM(nn.Module):
    """
    Physics-informed Residual ConvLSTM
    
    Args:
        input_channels: int = 13  # 输入通道数
        hidden_dims: list = [64, 128]  # ConvLSTM隐藏维度
        kernel_size: int = 3
        use_attention: bool = True  # 是否启用Channel Attention
        attention_reduction: int = 16
        dropout: float = 0.2
    """
    def __init__(self, input_channels=13, hidden_dims=[64,128], ...):
        # Encoder: 堆叠ConvLSTM
        self.encoder1 = ConvLSTMCell(input_channels, hidden_dims[0], kernel_size)
        self.encoder2 = ConvLSTMCell(hidden_dims[0], hidden_dims[1], kernel_size)
        
        # Channel Attention (可选)
        if use_attention:
            self.channel_attn = ChannelAttention(hidden_dims[1])
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dims[1], 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pred_head = nn.Conv2d(64, 1, 1)
        
        # 残差精炼网络
        self.refine_net = nn.Sequential(
            nn.Conv2d(hidden_dims[1] + 1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1)
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, T, C, H, W]  输入序列
        Returns:
            delta_p: [B, 1, H, W]  降水变化量
            p_pred:  [B, 1, H, W]  粗预测 (中间量)
        """
        B, T, C, H, W = x.shape
        
        # 提取最后一帧降水 (通道0)
        last_precip = x[:, -1, 0:1, :, :]  # [B, 1, H, W]
        
        # ConvLSTM编码
        h1, c1 = self.encoder1.init_hidden(B, (H, W))
        h2, c2 = self.encoder2.init_hidden(B, (H, W))
        for t in range(T):
            h1, c1 = self.encoder1(x[:, t], (h1, c1))
            h2, c2 = self.encoder2(h1, (h2, c2))
        
        # Channel Attention
        if self.channel_attn is not None:
            h2 = self.channel_attn(h2)
        
        # 解码
        decoded = self.decoder(h2)
        p_pred = self.pred_head(decoded)
        
        # 残差精炼
        refine_input = torch.cat([h2, p_pred], dim=1)
        delta_p = self.refine_net(refine_input)
        
        return delta_p, p_pred
    
    def predict(self, x):
        """完整推理: 返回 P_hat = ReLU(P_t + ΔP)"""
        last_precip = x[:, -1, 0:1, :, :]
        delta_p, _ = self.forward(x)
        return torch.relu(last_precip + delta_p)
```

### 2.5 新建文件: `src/models/baselines.py`

```python
class PersistenceModel:
    """持续预报: P_hat_{t+1} = P_t"""
    def predict(self, x):
        return x[:, -1, 0:1, :, :]  # 直接返回最后一帧降水

class ConvLSTM(nn.Module):
    """基础ConvLSTM (无残差、无物理约束)"""
    # 与PI-ResConvLSTM相同架构但无残差、无attention

class ResConvLSTM(nn.Module):
    """残差ConvLSTM (有时序残差、无物理损失)"""
    # PI-ResConvLSTM去掉attention和物理损失
```

---

## 3. 物理损失函数

### 3.1 新建文件: `src/training/physics_loss.py`

```python
class PhysicsInformedLoss(nn.Module):
    """
    L_total = L_rain + λ1·L_nonneg + λ2·L_oro + λ3·L_smooth + λ4·L_extreme
    
    Args:
        lambda_nonneg: float = 0.1
        lambda_oro: float = 0.1
        lambda_smooth: float = 0.01
        lambda_extreme: float = 0.5
        extreme_threshold: float = 10.0  # mm/h
        extreme_weight: float = 2.0
    """
    
    def forward(self, p_hat, p_true, terrain_gradient=None,
                wind_u=None, wind_v=None, last_frame=None):
        """
        Args:
            p_hat: [B, 1, H, W]  预测降水
            p_true: [B, 1, H, W]  真实降水
            terrain_gradient: [B, 2, H, W]  (dh/dx, dh/dy) — 可选
            wind_u, wind_v: [B, 1, H, W]  风场分量 — 可选
            last_frame: [B, 1, H, W]  前一帧降水 — 可选
        
        Returns:
            total_loss: scalar
            loss_dict: {name: value}  各项损失明细
        """
        # L_rain: 加权MSE
        weight = 1.0 + self.extreme_weight * (p_true > self.extreme_threshold).float()
        l_rain = (weight * (p_hat - p_true)**2).mean()
        
        # L_nonneg: 非负约束
        l_nonneg = (torch.relu(-p_hat)**2).mean()
        
        # L_oro: 地形抬升约束 (需要dh/dx, dh/dy, u, v)
        if terrain_gradient is not None and wind_u is not None:
            oro = wind_u * terrain_gradient[:, 0:1] + wind_v * terrain_gradient[:, 1:2]
            oro_mask = (oro > torch.quantile(oro[oro > 0], 0.8)).float()
            l_oro = (oro_mask * torch.relu(1.0 - p_hat)**2).mean()
        else:
            l_oro = torch.tensor(0.0, device=p_hat.device)
        
        # L_smooth: 时空平滑
        dx = torch.abs(p_hat[:, :, :, 1:] - p_hat[:, :, :, :-1]).mean()
        dy = torch.abs(p_hat[:, :, 1:, :] - p_hat[:, :, :-1, :]).mean()
        l_spatial = dx + dy
        if last_frame is not None:
            l_temporal = torch.abs(p_hat - last_frame).mean()
        else:
            l_temporal = torch.tensor(0.0, device=p_hat.device)
        l_smooth = l_spatial + l_temporal
        
        # L_extreme: 极端降水独立加权
        extreme_mask = (p_true > self.extreme_threshold).float()
        if extreme_mask.sum() > 0:
            l_extreme = ((p_hat - p_true)**2 * extreme_mask).sum() / extreme_mask.sum()
        else:
            l_extreme = torch.tensor(0.0, device=p_hat.device)
        
        # 总损失
        total = (l_rain 
                 + self.lambda_nonneg * l_nonneg 
                 + self.lambda_oro * l_oro 
                 + self.lambda_smooth * l_smooth 
                 + self.lambda_extreme * l_extreme)
        
        return total, {
            'l_rain': l_rain.item(),
            'l_nonneg': l_nonneg.item(),
            'l_oro': l_oro.item(),
            'l_smooth': l_smooth.item(),
            'l_extreme': l_extreme.item(),
            'total': total.item()
        }
```

### 3.2 损失权重建议

| 损失项 | 默认λ | 物理意义 | 调参建议 |
|--------|:-----:|----------|----------|
| L_rain | 1.0 | 主损失（加权MSE） | 不动 |
| L_nonneg | 0.1 | 降水非负 | 如输出已ReLU可降为0.01 |
| L_oro | 0.1 | 地形抬升约束 | 复杂地形可增到0.2 |
| L_smooth | 0.01 | 时空平滑 | 不要过大，会模糊峰值 |
| L_extreme | 0.5 | 极端降水加权 | 论文重点时可增到1.0 |

---

## 4. 数据管道改动

### 4.1 新建文件: `src/data/dataset.py`

**关键改进**: 按台风事件分组，防止数据泄漏

```python
class TyphoonDataset(Dataset):
    """
    台风降水时空数据集
    
    特性:
    - 从HDF5加载预构建的时空张量
    - 支持按台风事件/年份分组划分
    - 返回 (X, Y, metadata) 元组
    """
    
    def __init__(self, h5_path, typhoon_ids=None, seq_len=11,
                 transform=None, return_metadata=False):
        """
        Args:
            h5_path: HDF5文件路径
            typhoon_ids: 指定台风ID列表 (用于train/val/test划分)
            seq_len: 输入序列长度
            transform: 数据变换
            return_metadata: 是否返回台风ID和时间信息
        """
    
    def __getitem__(self, idx):
        X = self.samples[idx, :seq_len]     # [seq_len, C, H, W]
        Y = self.samples[idx, seq_len, 0:1] # [1, H, W] 仅降水
        return X, Y
    
    @staticmethod
    def split_by_year(h5_path, train_years=(2014,2022),
                      val_years=(2023,2023), test_years=(2024,2024)):
        """按年份划分train/val/test"""
    
    @staticmethod
    def split_by_typhoon(h5_path, train_ratio=0.7, val_ratio=0.15):
        """按台风事件随机划分 (Leave-one-typhoon-out)"""
```

### 4.2 新建文件: `src/data/transforms.py`

```python
class MinMaxNormalize:
    """降水场归一化到 [0, 1]"""
    def __init__(self, vmin=0.0, vmax=100.0):
        self.vmin = vmin
        self.vmax = vmax
    
    def __call__(self, x):
        return (x - self.vmin) / (self.vmax - self.vmin)

class LogTransform:
    """对数变换: x -> log(1 + x), 压缩极端值"""
    def __call__(self, x):
        return torch.log1p(x)

class Compose:
    """组合变换"""
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x
```

---

## 5. 训练框架改动

### 5.1 新建文件: `src/training/trainer.py`

统一训练循环，支持：
- 多模型对比训练
- 混合精度 (AMP)
- Early stopping
- 学习率调度 (ReduceLROnPlateau / CosineAnnealing)
- Checkpoint保存与恢复
- TensorBoard日志
- 训练曲线记录

```python
class Trainer:
    """统一训练器"""
    def __init__(self, model, train_loader, val_loader, config):
        """
        Args:
            model: nn.Module
            train_loader, val_loader: DataLoader
            config: dict 训练配置
        """
        self.model = model
        self.criterion = PhysicsInformedLoss(...) if config['use_physics_loss'] else nn.MSELoss()
        self.optimizer = AdamW(model.parameters(), lr=config['lr'], weight_decay=config['wd'])
        self.scaler = GradScaler()  # AMP
        self.scheduler = ReduceLROnPlateau(self.optimizer, patience=5)
        self.early_stopping = EarlyStopping(patience=10)
    
    def train_epoch(self):
        """一个epoch的训练"""
    
    def validate(self):
        """验证集评估"""
    
    def train(self, epochs):
        """完整训练循环"""
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            self.scheduler.step(val_loss)
            if self.early_stopping(val_loss):
                break
            self.save_checkpoint(epoch, val_loss)

def run_benchmark(config):
    """
    运行完整benchmark:
    1. Persistence
    2. ConvLSTM
    3. ResConvLSTM
    4. ResConvLSTM + DEM
    5. PI-ResConvLSTM
    6. PIA-ResConvLSTM (with Attention)
    
    每个模型训练并记录结果，最后生成对比表。
    """
```

### 5.2 新建文件: `src/training/configs/default.yaml`

```yaml
# PI-ResConvLSTM 默认训练配置

data:
  h5_path: "ConvLSTM_Dataset_128.h5"
  seq_len: 11
  train_years: [2014, 2022]
  val_years: [2023, 2023]
  test_years: [2024, 2024]
  num_workers: 4

model:
  name: "pi_res_convlstm"
  input_channels: 13
  hidden_dims: [64, 128]
  kernel_size: 3
  use_attention: true
  attention_reduction: 16
  dropout: 0.2

training:
  batch_size: 8
  epochs: 100
  lr: 0.0001
  weight_decay: 0.0001
  amp: true
  early_stopping_patience: 15
  scheduler_patience: 5

physics_loss:
  use_physics_loss: true
  lambda_nonneg: 0.1
  lambda_oro: 0.1
  lambda_smooth: 0.01
  lambda_extreme: 0.5
  extreme_threshold: 10.0
  extreme_weight: 2.0

reproducibility:
  seed: 42
  deterministic: true
```

---

## 6. 评估框架新建

### 6.1 新建文件: `src/evaluation/metrics.py`

```python
class TyphoonMetrics:
    """台风降水预测评估指标集"""
    
    @staticmethod
    def mae(pred, true):
        """平均绝对误差"""
        return F.l1_loss(pred, true).item()
    
    @staticmethod
    def rmse(pred, true):
        """均方根误差"""
        return torch.sqrt(F.mse_loss(pred, true)).item()
    
    @staticmethod
    def ssim(pred, true, window_size=11):
        """结构相似性指数"""
        # 使用 kornia 或 手写实现
    
    @staticmethod
    def csi(pred, true, threshold):
        """
        Critical Success Index (Threat Score)
        CSI = TP / (TP + FP + FN)
        """
        pred_yes = pred > threshold
        true_yes = true > threshold
        tp = (pred_yes & true_yes).sum().float()
        fp = (pred_yes & ~true_yes).sum().float()
        fn = (~pred_yes & true_yes).sum().float()
        return (tp / (tp + fp + fn + 1e-8)).item()
    
    @staticmethod
    def pod(pred, true, threshold):
        """Probability of Detection (命中率): POD = TP / (TP + FN)"""
    
    @staticmethod
    def far(pred, true, threshold):
        """False Alarm Ratio (虚警率): FAR = FP / (TP + FP)"""
    
    @staticmethod
    def hss(pred, true, threshold):
        """Heidke Skill Score"""
        # HSS = 2*(TP*TN - FP*FN) / ((TP+FN)*(FN+TN) + (TP+FP)*(FP+TN))
    
    @staticmethod
    def peak_error(pred, true):
        """峰值降水误差: |max(pred) - max(true)|"""
        return abs(pred.max().item() - true.max().item())
    
    @staticmethod
    def area_error(pred, true, threshold=10.0):
        """暴雨面积误差 (像素计数法)"""
        pred_area = (pred > threshold).sum().item()
        true_area = (true > threshold).sum().item()
        return abs(pred_area - true_area) / (true_area + 1e-8)
    
    @staticmethod
    def center_offset(pred, true):
        """降水中心偏移 (质心距离)"""
        # 计算降水质心的大圆距离
    
    @classmethod
    def compute_all(cls, pred, true, thresholds=[1.0, 5.0, 10.0, 20.0, 50.0]):
        """计算所有指标"""
        results = {
            'MAE': cls.mae(pred, true),
            'RMSE': cls.rmse(pred, true),
            'SSIM': cls.ssim(pred, true),
            'Peak_Error': cls.peak_error(pred, true),
        }
        for thresh in thresholds:
            results[f'CSI_{thresh}mm'] = cls.csi(pred, true, thresh)
            results[f'POD_{thresh}mm'] = cls.pod(pred, true, thresh)
            results[f'FAR_{thresh}mm'] = cls.far(pred, true, thresh)
        results['Area_Error_10mm'] = cls.area_error(pred, true, 10.0)
        results['Center_Offset_km'] = cls.center_offset(pred, true)
        return results
```

---

## 7. 推理引擎改动

### 7.1 新建文件: `src/inference/infer.py`

**关键修复**: 移除硬编码缩放因子

```python
class InferenceEngine:
    """
    修正版推理引擎
    
    与旧版(step2_3)的关键区别:
    1. 不使用 base_rain + nn_residual * 1.5 的hybrid模式
    2. 纯模型输出: P_hat = ReLU(P_t + ΔP)
    3. 自回归推理 + 误差传播分析
    """
    
    def __init__(self, model_path, model_config, device='cuda'):
        self.model = PIResConvLSTM(**model_config)
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(device)
        self.model.eval()
        self.device = device
    
    def run_autoregressive(self, initial_sequence, n_steps, 
                           dynamic_features_fn=None):
        """
        自回归推理
        
        Args:
            initial_sequence: [T, C, H, W]  初始序列 (来自历史数据)
            n_steps: 预测步数
            dynamic_features_fn: 获取每步动态特征的回调函数
                (step_idx, prev_pred) -> dynamic_channels
        
        Returns:
            predictions: [n_steps, H, W]  降水预测序列
            error_estimates: [n_steps]  累积误差估计
        """
        predictions = []
        current_seq = initial_sequence.clone()
        
        for step in range(n_steps):
            with torch.no_grad():
                batch_seq = current_seq.unsqueeze(0)  # [1, T, C, H, W]
                
                # 模型预测
                delta_p, _ = self.model(batch_seq)
                last_precip = current_seq[-1, 0:1]  # [1, H, W]
                p_next = torch.relu(last_precip + delta_p.squeeze(0))
                
            predictions.append(p_next.cpu().numpy())
            
            # 更新序列 (滑窗)
            # 构建新帧: [p_next, dynamic_features...]
            new_frame = self._build_frame(p_next, step, dynamic_features_fn)
            current_seq = torch.cat([current_seq[1:], new_frame.unsqueeze(0)])
        
        return np.array(predictions)
    
    def _build_frame(self, precip, step_idx, dynamic_fn):
        """构建包含所有通道的新帧"""
        # Channel 0: 降水 (模型输出)
        # Channel 1-12: 从dynamic_features_fn获取
        pass
```

---

## 8. 文件对照表

| 旧文件 | 新文件 | 改动类型 |
|--------|--------|:------:|
| `convLSTM_model.py` | `src/models/convlstm_cell.py` + `src/models/pi_res_convlstm.py` | 提取+重写 |
| — | `src/models/channel_attention.py` | 新建 |
| — | `src/models/baselines.py` | 新建 |
| `step2.1_spatial_dataloader.py` | `src/data/dataset.py` | 重写 (时序split) |
| — | `src/data/transforms.py` | 新建 |
| `step2_2_train_cloud.py` | `src/training/trainer.py` | 重写 (统一框架) |
| — | `src/training/physics_loss.py` | 新建 |
| — | `src/training/configs/default.yaml` | 新建 |
| — | `src/evaluation/metrics.py` | 新建 |
| `step2_3_generate_metrix.py` | `src/inference/infer.py` | 重写 (去除hardcode) |
| `step2_4_rainfall_heatmap.py` | `src/visualization/plot.py` | 重写 |
| — | `src/config.py` | 新建 |
| — | `configs/default.yaml` | 新建 |
| — | `requirements.txt` | 新建 |
| `README.md` | `README.md` | 重写 |
| 多个文档 | `docs/PROJECT_DOC.md` | 整合 |
| — | `docs/CODE_CHANGE_PLAN.md` | 本文件 |

### 保留不移动的文件

| 文件 | 原因 |
|------|------|
| `CMABSTdata/` | 原始数据 |
| `TIFdata/` | 原始数据 |
| `Global_DEM.tif` | 核心数据 |
| `ConvLSTM_Dataset_128.h5` | 训练数据 |
| `typhoon_convlstm_best.pth` | 旧模型权重(参考) |
| `*_DataPackage.npz` | 旧预测结果(参考) |
| `*.csv` (数据文件) | 中间数据 |
| `train_log.txt` | 训练记录(参考) |

---

## 9. 执行优先级

### Phase 4a: 模型层 (最高优先级)
1. `src/models/convlstm_cell.py` — 从旧代码提取，保持兼容
2. `src/models/channel_attention.py` — 独立模块，无依赖
3. `src/models/pi_res_convlstm.py` — 核心模型
4. `src/models/baselines.py` — 依赖convlstm_cell

### Phase 4b: 数据和训练层
5. `src/data/transforms.py` — 独立
6. `src/data/dataset.py` — 依赖现有HDF5
7. `src/training/physics_loss.py` — 独立
8. `src/training/trainer.py` — 依赖模型+数据+损失

### Phase 4c: 评估和推理层
9. `src/evaluation/metrics.py` — 独立
10. `src/inference/infer.py` — 依赖模型
11. `src/visualization/plot.py` — 独立
12. `src/config.py` — 依赖所有模块的路径整合
13. `configs/default.yaml`
14. `requirements.txt`
15. `README.md`

---

*文档版本: v1.0 | 更新日期: 2026-07-04*
*对应蓝图: docs/SUMMARY.md (Typhoon_PI_ResConvLSTM_Summary_for_Agent.md)*
