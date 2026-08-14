import os
import glob
import re
import pandas as pd
import numpy as np
import rasterio
from datetime import datetime

# ==================== 1. TIF 泛型特征提取接口 (全梯度升级版) ====================

def extract_base_precip_features(tif_path):
    """
    [纯粹的图像特征接口]：只解析图像本身的客观属性，不依赖外部台风数据。
    新增：基于 CMA 标准的降水强度分级面积特征 (Gradient Area Extraction)。
    """
    try:
        with rasterio.open(tif_path) as dataset:
            rain_matrix = dataset.read(1)
            transform = dataset.transform
            
            # 处理缺测值 -9999 和异常负值
            rain_matrix = np.where(rain_matrix == -9999, 0, rain_matrix)
            rain_matrix = np.clip(rain_matrix, 0, None)
            
            total_rain = np.sum(rain_matrix)
            
            # 如果该时刻完全没有降水 (防爆兜底)
            if total_rain == 0:
                return {
                    'P_total': 0.0,
                    'P_max': 0.0,
                    'P_Centroid_Lat': dataset.bounds.bottom + (dataset.bounds.top - dataset.bounds.bottom)/2,
                    'P_Centroid_Lon': dataset.bounds.left + (dataset.bounds.right - dataset.bounds.left)/2,
                    'Area_Light_0.1_2': 0.0,
                    'Area_Moderate_2_5': 0.0,
                    'Area_Heavy_5_10': 0.0,
                    'Area_Torrential_10_20': 0.0,
                    'S_ext_Extreme_over_20': 0.0
                }

            # 提取最大降水强度 (对应特征表里的 P_max)
            max_rain = np.max(rain_matrix)
            
            # 计算降水质心 (对应后续算 D_offset 的基础)
            cols, rows = np.meshgrid(np.arange(dataset.width), np.arange(dataset.height))
            lons, lats = rasterio.transform.xy(transform, rows, cols)
            lons = np.array(lons).reshape(rain_matrix.shape)
            lats = np.array(lats).reshape(rain_matrix.shape)
            
            centroid_lon = np.sum(lons * rain_matrix) / total_rain
            centroid_lat = np.sum(lats * rain_matrix) / total_rain
            
            # ================= 核心升级：降水梯度面积测算 =================
            # 像素的面积等效系数 (1个像素约 10km * 10km = 100 平方公里)
            pixel_area = 100.0 
            
            # 分布统计计算 (逻辑与)
            area_light = np.sum((rain_matrix >= 0.1) & (rain_matrix < 2.0)) * pixel_area
            area_mod = np.sum((rain_matrix >= 2.0) & (rain_matrix < 5.0)) * pixel_area
            area_heavy = np.sum((rain_matrix >= 5.0) & (rain_matrix < 10.0)) * pixel_area
            area_torr = np.sum((rain_matrix >= 10.0) & (rain_matrix < 20.0)) * pixel_area
            area_ext = np.sum(rain_matrix >= 20.0) * pixel_area  # S_ext

            # 返回特征字典 (保持了你要求的泛型接口设计)
            return {
                'P_total': total_rain,
                'P_max': max_rain,
                'P_Centroid_Lat': centroid_lat,
                'P_Centroid_Lon': centroid_lon,
                'Area_Light_0.1_2': area_light,
                'Area_Moderate_2_5': area_mod,
                'Area_Heavy_5_10': area_heavy,
                'Area_Torrential_10_20': area_torr,
                'S_ext_Extreme_over_20': area_ext
            }
            
    except Exception as e:
        print(f"读取 {os.path.basename(tif_path)} 时出错: {e}")
        return None
    
# ==================== 2. 批量处理与文件遍历引擎 ====================

def process_all_tif_folders(tif_root_dir, output_csv_path):
    print(f"正在扫描 TIF 根目录: {tif_root_dir}")
    
    # 查找所有子文件夹下的 .tif 文件
    search_pattern = os.path.join(tif_root_dir, "*", "*.tif")
    tif_files = glob.glob(search_pattern)
    
    total_files = len(tif_files)
    if total_files == 0:
        print("未找到任何 .tif 文件，请检查路径是否正确！")
        return
        
    print(f"共发现 {total_files} 个降水图像文件，开始批量特征提取...")
    
    results = []
    
    for idx, tif_path in enumerate(tif_files):
        # 提取文件名中的时间戳 (正则匹配: 8位数字-S6位数字)
        filename = os.path.basename(tif_path)
        time_match = re.search(r'(\d{8}-S\d{6})', filename)
        
        if not time_match:
            continue # 如果文件名格式不符，跳过
            
        time_str = time_match.group(1)
        # 将 "20241222-S120000" 转换为标准的 Datetime 对象
        parsed_time = datetime.strptime(time_str, '%Y%m%d-S%H%M%S')
        
        # 提取父文件夹名称 (事件编号)
        event_folder = os.path.basename(os.path.dirname(tif_path))
        
        # 调用接口提取图像特征
        features = extract_base_precip_features(tif_path)
        
        if features is not None:
            # 构建单行数据记录
            row_data = {
                'Folder_ID': event_folder,
                'Time': parsed_time,
                'Filename': filename
            }
            # 泛型拼接：将提取出的字典直接合并到记录中
            row_data.update(features)
            results.append(row_data)
            
        # 打印进度提示
        if (idx + 1) % 500 == 0 or (idx + 1) == total_files:
            print(f"进度: 已处理 {idx + 1} / {total_files} 个文件...")
            
    # 将提取结果转为 DataFrame 并保存
    print("\n正在生成 CSV 文件，请稍候...")
    df_results = pd.DataFrame(results)
    
    # 按文件夹和时间进行排序，保证数据工整
    df_results = df_results.sort_values(by=['Folder_ID', 'Time'])
    df_results.to_csv(output_csv_path, index=False)
    
    print(f"✅ TIF 图像特征提取完毕！基础数据已保存至: {output_csv_path}")


# ==================== 3. 主程序入口 ====================

if __name__ == '__main__':
    # 你的 TIFdata 文件夹路径 (请根据实际情况修改)
    TIF_DIRECTORY = r"C:\Users\champ\Desktop\数学建模\数学建模北京\2026年北京高校数学建模校际联赛赛题\B题\2026校赛B题\TIFdata"
    OUTPUT_FILE = 'TIF_Features_Base.csv'
    
    if not os.path.exists(TIF_DIRECTORY):
         print(f"错误: 找不到文件夹 {TIF_DIRECTORY}")
    else:
         process_all_tif_folders(TIF_DIRECTORY, OUTPUT_FILE)