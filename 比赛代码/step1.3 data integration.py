import pandas as pd
import numpy as np
import math
import os
import rasterio

# ==================== 1. 基础数学工具 ====================

def haversine_distance(lat1, lon1, lat2, lon2):
    """计算地球表面两点间的大圆距离 (km) -> 用于计算 D_offset"""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    a = math.sin((lat2 - lat1)/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1)/2)**2
    return R * 2 * math.asin(math.sqrt(a))

def calculate_asymmetry_index(tif_path, center_lat, center_lon):
    """
    计算降水非对称性指数 (I_asy)
    逻辑：以台风中心为原点划分四大象限，计算降水量的变异系数 (CV = std / mean)
    """
    try:
        with rasterio.open(tif_path) as dataset:
            rain_matrix = dataset.read(1)
            transform = dataset.transform
            
            # 处理异常值
            rain_matrix = np.where(rain_matrix == -9999, 0, rain_matrix)
            rain_matrix = np.clip(rain_matrix, 0, None)
            
            if np.sum(rain_matrix) == 0:
                return 0.0
                
            # 获取经纬度网格
            cols, rows = np.meshgrid(np.arange(dataset.width), np.arange(dataset.height))
            lons, lats = rasterio.transform.xy(transform, rows, cols)
            lons = np.array(lons).reshape(rain_matrix.shape)
            lats = np.array(lats).reshape(rain_matrix.shape)
            
            # 划分四大象限并求和
            q1_ne = rain_matrix[(lons >= center_lon) & (lats >= center_lat)].sum() # 东北
            q2_nw = rain_matrix[(lons < center_lon)  & (lats >= center_lat)].sum() # 西北
            q3_sw = rain_matrix[(lons < center_lon)  & (lats < center_lat)].sum()  # 西南
            q4_se = rain_matrix[(lons >= center_lon) & (lats < center_lat)].sum()  # 东南
            
            quadrants = np.array([q1_ne, q2_nw, q3_sw, q4_se])
            
            # 变异系数 CV = 标准差 / 平均值 (如果平均值为0，则避免除以0)
            mean_q = np.mean(quadrants)
            if mean_q == 0:
                return 0.0
            cv = np.std(quadrants) / mean_q
            return cv
    except Exception as e:
        print(f"读取 TIF 计算非对称性时出错: {e}")
        return np.nan

# ==================== 2. 核心融合逻辑 ====================

def fuse_and_derive_features(txt_csv, tif_csv, tif_root_dir, output_csv):
    print("1. 正在加载 1D 轨迹与 2D 降水基础数据...")
    df_txt = pd.read_csv(txt_csv)
    df_tif = pd.read_csv(tif_csv)
    
    # 时间格式统一
    df_txt['Time'] = pd.to_datetime(df_txt['Time'])
    df_tif['Time'] = pd.to_datetime(df_tif['Time'])
    
    # ID格式统一为字符串
    df_txt['Typhoon_ID'] = df_txt['Typhoon_ID'].astype(str).str.zfill(4)
    df_tif['Folder_ID'] = df_tif['Folder_ID'].astype(str)
    
    # 【修复核心】：从 2014100 等文件夹名中截取出第3~6位（即1410）作为匹配暗号
    df_tif['Match_ID'] = df_tif['Folder_ID'].str[2:6]
    
    print("2. 正在计算 1D 时序派生特征 (Delta P & Delta V)...")
    # 计算 6小时变率 (因为间隔是0.5h，所以 6小时 = 12 个 period)
    # 必须按 Typhoon_ID 分组计算，防止跨台风做差分
    df_txt['Delta_P_6h'] = df_txt.groupby('Typhoon_ID')['Pressure'].diff(periods=12)
    df_txt['Delta_V_6h'] = df_txt.groupby('Typhoon_ID')['Wind_Speed'].diff(periods=12)
    
    print("3. 正在执行多模态时空对齐 (Inner Merge)...")
    # 只有 txt 和 tif 同时存在的时刻，才会保留
    df_full = pd.merge(
        df_txt, 
        df_tif, 
        left_on=['Typhoon_ID', 'Time'], 
        right_on=['Match_ID', 'Time'], # 这里换成了提取出来的 Match_ID
        how='inner'
    )
    print(f"   => 成功对齐 {len(df_full)} 行综合数据！")
    
    print("4. 正在计算 2D 交叉特征 (质心偏向 D_offset & 非对称性 I_asy)...")
    offset_distances = []
    asymmetry_indices = []
    
    total_rows = len(df_full)
    for idx, row in df_full.iterrows():
        # A. 计算降水中心偏向距离 D_offset
        if row['P_total'] == 0:
            offset_distances.append(0.0)
            asymmetry_indices.append(0.0)
            continue
            
        dist = haversine_distance(
            row['Lat'], row['Lon'], 
            row['P_Centroid_Lat'], row['P_Centroid_Lon']
        )
        offset_distances.append(dist)
        
        # B. 计算降水非对称性 I_asy (需要回溯对应的 TIF 文件)
        # 拼接绝对路径：TIF_ROOT_DIR / Folder_ID / Filename
        tif_absolute_path = os.path.join(tif_root_dir, str(row['Folder_ID']), str(row['Filename']))
        
        if os.path.exists(tif_absolute_path):
            asy = calculate_asymmetry_index(tif_absolute_path, row['Lat'], row['Lon'])
            asymmetry_indices.append(asy)
        else:
            asymmetry_indices.append(np.nan)
            
        if (idx + 1) % 500 == 0:
            print(f"   已处理交叉特征: {idx + 1} / {total_rows}")
            
    df_full['D_offset_km'] = offset_distances
    df_full['I_asy_Index'] = asymmetry_indices
    
    print("5. 正在初始化外部环境变量 (X3) 脚手架 (留白处理)...")
    # ================= 外部数据占位符 =================
    # 在你找到外部数据并写好查询逻辑前，先用 NaN 填充占位
    
    # [1] 海陆状态 (LS): 0=海洋, 1=陆地
    # 逻辑提示: 使用 geopandas 读取全球海岸线 .shp，判断 (Lat, Lon) 是否在多边形内
    df_full['LS_Land_Status'] = np.nan 
    
    # [2] 距海岸线距离 (DTC): km
    # 逻辑提示: 计算 (Lat, Lon) 到最近海岸线对象的欧氏/大圆距离
    df_full['DTC_Dist_To_Coast_km'] = np.nan
    
    # [3] 地形高程 (DEM): m
    # 逻辑提示: 使用 rasterio 或 xarray 读取 SRTM/ETOPO1 高程网格，提取该点的海拔
    df_full['DEM_Elevation_m'] = np.nan
    
    # [4] 强度演变势能 (IEP): 复合指数
    # 逻辑提示: 结合下载的 ECMWF ERA5 的 SST(海温) 和 VWS(风切变) 数据进行公式计算
    df_full['IEP_Potential_Energy'] = np.nan
    
    # ---------------- 清理与导出 ----------------
    if 'Folder_ID' in df_full.columns:
        df_full = df_full.drop(columns=['Folder_ID']) # 删掉冗余ID列
        
    df_full.to_csv(output_csv, index=False)
    print(f"\n✅ 第一问全特征大表已生成！文件保存至: {output_csv}")
    print("下一步：你可以去收集 DEM、海岸线等外部数据，直接填充 CSV 中的留白列。")

# ==================== 3. 运行入口 ====================
if __name__ == '__main__':
    # 文件路径配置 (请确保与前两步生成的文件名一致)
    TXT_FEATURES = 'All_Years_Typhoon_Features.csv' # Step 1.1 生成
    TIF_FEATURES = 'TIF_Features_Base.csv'          # Step 1.2 生成 (全梯度版)
    TIF_ROOT_DIR = r"C:\Users\champ\Desktop\数学建模\数学建模北京\2026年北京高校数学建模校际联赛赛题\B题\2026校赛B题\TIFdata" # 原始图片库，用于算 I_asy
    FINAL_OUTPUT = 'Typhoon_Full_Dataset_Q1.csv'
    
    fuse_and_derive_features(TXT_FEATURES, TIF_FEATURES, TIF_ROOT_DIR, FINAL_OUTPUT)