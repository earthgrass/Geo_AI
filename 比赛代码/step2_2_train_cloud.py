"""此部分云端运行
Typhoon Precipitation Spatio-Temporal Prediction (Step 2.2 - Cloud GPU Version)
Architecture: Spatial Residual ConvLSTM Network (Full Capacity [64, 128])
Author: Zhang Jiahao (ZJH) 云端运行
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
import h5py
import numpy as np
from tqdm import tqdm

# ==========================================
# 1. 云端数据读取 (HDF5 惰性安全加载)
# ==========================================
class TyphoonCloudDataset(Dataset):
    def __init__(self, h5_file_path):
        self.h5_path = h5_file_path
        print(f"[Data] 正在连接云端 HDF5 数据集: {h5_file_path} ...")
        with h5py.File(self.h5_path, 'r') as f:
            self.length = f['data'].shape[0]
        print(f"[Data] 成功挂载！总滑窗样本数: {self.length}")
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 每次单独打开句柄，完美支持 DataLoader 的多进程并发 (num_workers > 0)
        with h5py.File(self.h5_path, 'r') as f:
            sample = f['data'][idx]
            
        # 分离特征和标签
        # X: [Seq=11, C=4, H=128, W=128]
        X = torch.tensor(sample[:-1, :, :, :], dtype=torch.float32)
        # Y: 仅取最后时刻的降水场 [C=1, H=128, W=128]
        Y_precip = torch.tensor(sample[-1, 0, :, :], dtype=torch.float32).unsqueeze(0)
        
        return X, Y_precip

# ==========================================
# 2. 核心网络架构 (空间残差 ConvLSTM)
# ==========================================
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(in_channels=input_dim + hidden_dim,
                              out_channels=4 * hidden_dim,
                              kernel_size=kernel_size,
                              padding=kernel_size // 2,
                              bias=bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        cc_i, cc_f, cc_o, cc_g = torch.split(self.conv(combined), self.hidden_dim, dim=1)
        
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        device = self.conv.weight.device
        return (torch.zeros(batch_size, self.hidden_dim, *image_size, device=device),
                torch.zeros(batch_size, self.hidden_dim, *image_size, device=device))

class SpatialResidualConvLSTM(nn.Module):
    def __init__(self, input_channels=4, hidden_dims=[64, 128], kernel_size=3):
        super(SpatialResidualConvLSTM, self).__init__()
        # 满血高维特征提取
        self.encoder1 = ConvLSTMCell(input_channels, hidden_dims[0], kernel_size)
        self.encoder2 = ConvLSTMCell(hidden_dims[0], hidden_dims[1], kernel_size)
        
        # 解码器降维
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dims[1], hidden_dims[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.ReLU(inplace=True)
        )
        self.pred_head = nn.Conv2d(hidden_dims[0], 1, kernel_size=1)
        
        # 残差物理修补网络
        self.residual_net = nn.Sequential(
            nn.Conv2d(hidden_dims[1] + 1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)
        )

    def forward(self, x):
        b, seq_len, _, h, w = x.size()
        h1, c1 = self.encoder1.init_hidden(b, (h, w))
        h2, c2 = self.encoder2.init_hidden(b, (h, w))
        
        # 序列时间步迭代
        for t in range(seq_len):
            h1, c1 = self.encoder1(x[:, t, :, :, :], (h1, c1))
            h2, c2 = self.encoder2(h1, (h2, c2))
            
        dec_feat = self.decoder(h2)
        p_pred = self.pred_head(dec_feat)
        
        # 将深层空间特征与初始预测拼接，学习空间残差
        res_input = torch.cat([h2, p_pred], dim=1) 
        delta_p = self.residual_net(res_input)
        
        # ReLU 施加物理非负约束
        return F.relu(p_pred + delta_p)

# ==========================================
# 3. 训练引擎与主程序
# ==========================================
if __name__ == "__main__":
    # ---------------- 核心配置 ----------------
    H5_PATH = "ConvLSTM_Dataset_128.h5"  # 确保文件名和上传的一致
    BATCH_SIZE = 8   # 对于 RTX 4090 24G，8 到 16 是个好数字
    EPOCHS = 20      # 数模比赛时间紧，20轮通常能看到明显收敛
    LR = 1e-3
    NUM_WORKERS = 4  # 开启多线程加速硬盘读取
    # ------------------------------------------

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[System] 启动训练引擎! 核心设备: {device}")
    if torch.cuda.is_available():
        print(f"[System] 显卡型号: {torch.cuda.get_device_name(0)}")

    # 1. 挂载数据并切分
    dataset = TyphoonCloudDataset(H5_PATH)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # 2. 部署模型与优化器
    model = SpatialResidualConvLSTM().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scaler = GradScaler() # 初始化混合精度加速器

    best_val_loss = float('inf')
    
    print("\n" + "="*45)
    print("🚀 开始冲刺训练 (Training Started)")
    print("="*45)
    
    # 3. 开始训练循环
    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        
        # 训练批次
        for batch_idx, (X, Y) in enumerate(train_loader):
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            
            # 开启 AMP 自动混合精度
            with autocast():
                predictions = model(X)
                loss = criterion(predictions, Y)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * X.size(0)
            
            # 每 50 个 batch 打印一次，防止日志过长
            if batch_idx % 50 == 0:
                print(f"  [Epoch {epoch+1}/{EPOCHS}] Batch {batch_idx}/{len(train_loader)} | Current MSE: {loss.item():.4f}")
            
        epoch_train_loss = train_loss / train_size
        
        # 验证批次
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, Y_val in val_loader:
                X_val, Y_val = X_val.to(device), Y_val.to(device)
                with autocast():
                    val_preds = model(X_val)
                    v_loss = criterion(val_preds, Y_val)
                val_loss += v_loss.item() * X_val.size(0)
                
        epoch_val_loss = val_loss / val_size
        time_taken = time.time() - start_time
        
        print(f"🏁 Epoch {epoch+1}/{EPOCHS} 总结 | 耗时: {time_taken:.1f}s")
        print(f"   Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")
        
        # 判定与保存最优权重
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), 'typhoon_convlstm_best.pth')
            print("   🌟 发现新低验证集 Loss，权重已更新保存!")
            
    print("\n🎉 训练圆满结束！请下载 'typhoon_convlstm_best.pth' 文件备用。")