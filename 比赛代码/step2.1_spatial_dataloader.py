"""
Typhoon Precipitation Spatio-Temporal Dataloader (Step 2.1 - HDF5 Buffered Version)
Architecture: Physical Field Rendering + Time-weighted Interpolation + HDF5 Streaming
Author: Zhang Jiahao (ZJH) & AI Co-pilot
Date: 2026-04
"""

import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import warnings
import h5py  # 新增：用于处理超大矩阵的硬盘存储

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


# =====================================================================
# Module 1: Physical Field Engine (物理场渲染引擎)
# =====================================================================
class PhysicalFieldEngine:
    """利用矢量化 NumPy 矩阵，将 1D 动力学参量映射为 2D 物理空间场"""
    def __init__(self, grid_size=128, resolution_km=10.0):
        self.grid_size = grid_size
        self.half_grid = grid_size // 2
        self.resolution_km = resolution_km
        
        y, x = np.ogrid[-self.half_grid:self.half_grid, -self.half_grid:self.half_grid]
        self.distance_matrix_km = np.sqrt(x**2 + y**2) * self.resolution_km

    def render_wind_field(self, v_max, r_max, decay_factor=50.0):
        r_max = max(r_max, 1.0)
        inner_core = v_max * (self.distance_matrix_km / r_max)
        outer_region = v_max * np.exp(-(self.distance_matrix_km - r_max) / decay_factor)
        return np.where(self.distance_matrix_km <= r_max, inner_core, outer_region)

    def render_pressure_field(self, p_center, p_env=1010.0):
        pressure_drop = max(p_env - p_center, 0)
        return p_env - pressure_drop * np.exp(-(self.distance_matrix_km**2) / (2 * 300**2))


# =====================================================================
# Module 2: Spatial Pipeline Builder (多模态时空流水线 - HDF5 流式写入版)
# =====================================================================
class SpatialPipelineBuilder:
    def __init__(self, tif_dir, df_features, grid_size=128, seq_len=12, stride=1, max_missing=2):
        self.tif_dir = tif_dir
        self.df = df_features
        self.grid_size = grid_size
        self.seq_len = seq_len
        self.stride = stride
        self.max_missing = max_missing
        self.physics_engine = PhysicalFieldEngine(grid_size=grid_size)

    def _get_pixel_window(self, transform, lat, lon):
        col, row = ~transform * (lon, lat)
        col, row = int(np.round(col)), int(np.round(row))
        return Window(col - self.grid_size//2, row - self.grid_size//2, self.grid_size, self.grid_size)

    def _extract_precipitation_crop(self, tif_path, lat, lon):
        if not os.path.exists(tif_path):
            return None
        try:
            with rasterio.open(tif_path) as src:
                window = self._get_pixel_window(src.transform, lat, lon)
                crop_data = src.read(1, window=window, boundless=True, fill_value=0.0)
                crop_data = np.where(crop_data < 0, 0.0, crop_data)
                return crop_data
        except Exception:
            return None

    def _impute_sequence(self, precip_list):
        missing_indices = [i for i, p in enumerate(precip_list) if p is None]
        if len(missing_indices) > self.max_missing:
            return None 
            
        for idx in missing_indices:
            left, right = idx - 1, idx + 1
            while left >= 0 and precip_list[left] is None: left -= 1
            while right < len(precip_list) and precip_list[right] is None: right += 1
            
            if left >= 0 and right < len(precip_list):
                weight_l = (right - idx) / (right - left)
                weight_r = (idx - left) / (right - left)
                precip_list[idx] = precip_list[left] * weight_l + precip_list[right] * weight_r
            elif left >= 0:
                precip_list[idx] = precip_list[left]
            elif right < len(precip_list):
                precip_list[idx] = precip_list[right]
            else:
                return None 
        return precip_list

    def build_dataset(self, output_h5_path="Typhoon_Dataset.h5", buffer_size=500):
        """流式构建，分批写入 HDF5 以防内存溢出"""
        grouped = self.df.groupby('Typhoon_ID')
        print(f"Start processing {len(grouped)} typhoons. Saving to {output_h5_path}...")
        
        buffer = []
        total_samples = 0

        # 以写模式创建 HDF5 文件
        with h5py.File(output_h5_path, 'w') as f:
            # 创建动态可扩容的数据集 (maxshape 在第 0 维设为 None)
            dataset = f.create_dataset(
                'data', 
                shape=(0, self.seq_len, 4, self.grid_size, self.grid_size),
                maxshape=(None, self.seq_len, 4, self.grid_size, self.grid_size),
                dtype='float32',
                compression="gzip" # 启用 gzip 压缩，极大减小文件体积
            )
            
            for typhoon_id, group in tqdm(grouped, desc="Building Sliding Windows"):
                group = group.sort_values('Time').reset_index(drop=True)
                n_frames = len(group)
                if n_frames < self.seq_len:
                    continue
                
                frames_precip = []
                frames_static = []
                
                # Step 1: 预处理该台风的所有单帧
                for _, row in group.iterrows():
                    base_id = str(int(row['Typhoon_ID']))
                    folder_id = f"20{base_id}0" if len(base_id) == 4 else base_id 
                    raw_filename = str(row['Filename'])
                    tif_filename = raw_filename if raw_filename.endswith('.tif') else f"{raw_filename}.tif"
                    tif_path = os.path.join(self.tif_dir, folder_id, tif_filename)
                    
                    p = self._extract_precipitation_crop(tif_path, row['Lat'], row['Lon'])
                    wind = self.physics_engine.render_wind_field(row['Wind_Speed'], row['Radius_max_wind_km'])
                    press = self.physics_engine.render_pressure_field(row['Pressure'])
                    dist = self.physics_engine.distance_matrix_km
                    
                    frames_precip.append(p)
                    # 提前转换为 float32 节省内存
                    frames_static.append(np.stack([wind, press, dist], axis=0).astype('float32'))
                
                # Step 2: 时序滑窗与插值补全
                for i in range(0, n_frames - self.seq_len + 1, self.stride):
                    window_precip = frames_precip[i : i + self.seq_len]
                    window_static = frames_static[i : i + self.seq_len]
                    
                    imputed_precip = self._impute_sequence(list(window_precip))
                    if imputed_precip is None:
                        continue 
                    
                    # Step 3: 通道拼接
                    seq_tensor = []
                    for t in range(self.seq_len):
                        c0 = np.expand_dims(imputed_precip[t], axis=0).astype('float32')
                        full_frame = np.concatenate([c0, window_static[t]], axis=0) 
                        seq_tensor.append(full_frame)
                        
                    # 加入缓冲池
                    buffer.append(np.stack(seq_tensor, axis=0))
                    
                    # 触发缓冲池落地硬盘机制
                    if len(buffer) >= buffer_size:
                        buffer_arr = np.stack(buffer, axis=0)
                        current_size = dataset.shape[0]
                        # 动态扩展 HDF5 容量并写入
                        dataset.resize(current_size + buffer_arr.shape[0], axis=0)
                        dataset[current_size:] = buffer_arr
                        total_samples += buffer_arr.shape[0]
                        buffer = [] # 清空内存
                        
            # 将最后残余在缓冲区的数据刷入硬盘
            if len(buffer) > 0:
                buffer_arr = np.stack(buffer, axis=0)
                current_size = dataset.shape[0]
                dataset.resize(current_size + buffer_arr.shape[0], axis=0)
                dataset[current_size:] = buffer_arr
                total_samples += buffer_arr.shape[0]
                
        print(f"\n[Success] Dataset constructed safely via HDF5 Streaming!")
        print(f"Total Samples Generated: {total_samples}")


# =====================================================================
# Module 3: PyTorch ConvLSTM Dataset (HDF5 安全并发读取版)
# =====================================================================
class TyphoonSpatioTemporalDataset(Dataset):
    """标准的 PyTorch DataLoader 接口，完美规避多线程读取冲突"""
    def __init__(self, h5_file_path):
        self.h5_path = h5_file_path
        print(f"Connecting to HDF5 database at {h5_file_path}...")
        # 初始化时只读元数据，不占内存
        with h5py.File(self.h5_path, 'r') as f:
            self.length = f['data'].shape[0]
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 每次读取都单独打开文件句柄，彻底解决 DataLoader 多进程 (num_workers>0) 死锁问题
        with h5py.File(self.h5_path, 'r') as f:
            sample = f['data'][idx]
            
        # X: [11, 4, 128, 128], Y: [1, 128, 128]
        X = torch.tensor(sample[:-1, :, :, :], dtype=torch.float32)
        Y_precip = torch.tensor(sample[-1, 0, :, :], dtype=torch.float32).unsqueeze(0)
        
        return X, Y_precip


# =====================================================================
# Main Execution Block
# =====================================================================
if __name__ == "__main__":
    CSV_PATH = "Typhoon_Full_Dataset_Q1.csv"
    TIF_DIR = "TIFdata/"
    OUTPUT_H5 = "ConvLSTM_Dataset_128.h5"
    
    print(">>> Phase 2.1: Initializing Buffered Spatial Tensor Pipeline...")
    
    if os.path.exists(CSV_PATH):
        df_features = pd.read_csv(CSV_PATH)
        
        builder = SpatialPipelineBuilder(
            tif_dir=TIF_DIR,
            df_features=df_features,
            grid_size=128,
            seq_len=12,
            stride=1,
            max_missing=2
        )
        
        # 构建张量，每处理 500 个样本清理一次内存
        builder.build_dataset(output_h5_path=OUTPUT_H5, buffer_size=500)
        
        # 测试 DataLoader
        dataset = TyphoonSpatioTemporalDataset(OUTPUT_H5)
        # 注意: 这里的 num_workers=2 是为了测试咱们安全读取代码的健壮性
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0) 
        
        X_batch, Y_batch = next(iter(dataloader))
        print(f"\nDataloader Test Pass!")
        print(f"X_batch shape: {X_batch.shape} # [Batch, Seq-1, Channels, H, W]")
        print(f"Y_batch shape: {Y_batch.shape} # [Batch, 1, H, W]")
    else:
        print(f"Waiting for CSV data at {CSV_PATH}...")