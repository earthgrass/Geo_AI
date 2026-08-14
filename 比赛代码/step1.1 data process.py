import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
import math
import os

# ==================== 1. 基础数学与地理公式 ====================

def haversine(lat1, lon1, lat2, lon2):
    """计算两经纬度点之间的地球大圆距离 (单位: km)"""
    R = 6371.0 # 地球平均半径
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

def calculate_bearing(lat1, lon1, lat2, lon2):
    """计算从点1移动到点2的方位角 (0-360度，正北为0，正东为90)"""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(dlon))
    initial_bearing = math.atan2(x, y)
    return (math.degrees(initial_bearing) + 360) % 360

def calculate_curvature(bearing1, bearing2, distance):
    """
    计算路径曲率 (度/km)
    处理了360度跨越的问题 (例如从 350度转到 10度，实际只转了20度)
    """
    if distance == 0:
        return 0.0
    delta_bearing = (bearing2 - bearing1 + 180) % 360 - 180
    return delta_bearing / distance


# ==================== 2. 核心解析与处理逻辑 ====================

def process_typhoon_data(file_path):
    records = []
    current_typhoon_id = None
    
    print("正在读取原始 txt 文件...")
    # 2.1 逐行解析 txt 文件
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.split()
            if not parts: continue
            
            # 判断是否为头记录 (以 66666 开头)
            if parts[0] == '66666':
                current_typhoon_id = parts[4] # 提取台风编号
            else:
                # 解析数据行: 时间, 强度, 纬度(0.1度), 经度(0.1度), 气压(hPa), 风速(m/s)
                time_str = parts[0]
                lat = float(parts[2]) / 10.0
                lon = float(parts[3]) / 10.0
                pres = float(parts[4])
                wind = float(parts[5])
                
                records.append({
                    'Typhoon_ID': current_typhoon_id,
                    'Time': pd.to_datetime(time_str, format='%Y%m%d%H'),
                    'Lat': lat,
                    'Lon': lon,
                    'Pressure': pres,
                    'Wind_Speed': wind
                })
                
    df = pd.DataFrame(records)
    interpolated_dfs = []

    print("正在进行三次样条插值与特征计算...")
    # 2.2 对每场台风单独进行插值和特征计算
    for t_id, group in df.groupby('Typhoon_ID'):
        group = group.sort_values('Time').reset_index(drop=True)
        # 如果台风记录少于3个点，无法进行三次样条插值，跳过
        if len(group) < 3: 
            continue 
            
        # 将时间转换为相对小时数，方便多项式插值
        start_time = group['Time'].iloc[0]
        hours_from_start = (group['Time'] - start_time).dt.total_seconds() / 3600.0
        
        # 设定目标频次：每 0.5 小时采样一次
        new_hours = np.arange(0, hours_from_start.iloc[-1] + 0.5, 0.5)
        new_times = start_time + pd.to_timedelta(new_hours, unit='h')
        
        # 三次样条插值拟合
        cs_lat = CubicSpline(hours_from_start, group['Lat'])
        cs_lon = CubicSpline(hours_from_start, group['Lon'])
        cs_pres = CubicSpline(hours_from_start, group['Pressure'])
        cs_wind = CubicSpline(hours_from_start, group['Wind_Speed'])
        
        # 组装基础插值数据
        interp_df = pd.DataFrame({
            'Typhoon_ID': t_id,
            'Time': new_times,
            'Lat': cs_lat(new_hours),  # 【中心位置 - 纬度】
            'Lon': cs_lon(new_hours),  # 【中心位置 - 经度】
            'Pressure': cs_pres(new_hours),
            'Wind_Speed': np.clip(cs_wind(new_hours), a_min=0, a_max=None) # 风速不能为负
        })
        
        # 2.3 计算高阶运动学特征
        distances = [0.0]
        bearings = [0.0]
        curvatures = [0.0]
        
        for i in range(1, len(interp_df)):
            lat1, lon1 = interp_df.loc[i-1, ['Lat', 'Lon']]
            lat2, lon2 = interp_df.loc[i, ['Lat', 'Lon']]
            
            # 计算距离和方位角
            dist = haversine(lat1, lon1, lat2, lon2)
            b = calculate_bearing(lat1, lon1, lat2, lon2)
            
            distances.append(dist)
            bearings.append(b)
            
            # 计算曲率 (当前方位角与上一个方位角的差值 / 距离)
            if i > 1:
                curv = calculate_curvature(bearings[i-1], b, dist)
                curvatures.append(curv)
            else:
                # 第二个点时没有上一个方位角差，记为0
                curvatures.append(0.0) 
            
        interp_df['Distance_km'] = distances
        interp_df['Moving_Speed_kmh'] = interp_df['Distance_km'] / 0.5 # 【移动速度】 v = s/t
        interp_df['Moving_Direction'] = bearings                       # 【移动方位角】
        interp_df['Curvature_deg_per_km'] = curvatures                 # 【台风曲率】
        
        # 2.4 计算台风半径 (利用 Willoughby 2006 经验公式估计最大风速半径 RMW)
        # 公式: Rmax = 46.4 * exp(-0.0155 * Vmax + 0.0169 * Lat)
        interp_df['Radius_max_wind_km'] = 46.4 * np.exp(-0.0155 * interp_df['Wind_Speed'] + 0.0169 * np.abs(interp_df['Lat'])) # 【台风半径】
        
        interpolated_dfs.append(interp_df)

    # 聚合所有结果
    final_df = pd.concat(interpolated_dfs, ignore_index=True)
    return final_df

import glob

# ==================== 3. 主程序入口 (批量处理版) ====================

if __name__ == '__main__':
    # 填入你电脑上 CMABSTdata 文件夹的绝对路径 (注意末尾不要加反斜杠)
    folder_path = r"C:\Users\champ\Desktop\数学建模\数学建模北京\2026年北京高校数学建模校际联赛赛题\B题\2026校赛B题\CMABSTdata"
    output_path = 'All_Years_Typhoon_Features.csv'
    
    if not os.path.exists(folder_path):
        print(f"找不到文件夹: {folder_path}，请检查路径！")
    else:
        # 使用 glob 自动抓取文件夹下所有的 .txt 文件
        txt_files = glob.glob(os.path.join(folder_path, "*.txt"))
        print(f"共发现 {len(txt_files)} 个年份的台风数据文件，开始批量处理...\n")
        
        all_years_dfs = []
        for file in txt_files:
            print(f"正在处理 -> {os.path.basename(file)}")
            try:
                df = process_typhoon_data(file)
                all_years_dfs.append(df)
            except Exception as e:
                print(f"处理 {os.path.basename(file)} 时跳过，原因: {e}")
                
        # 将所有年份的数据拼接到一个超级大表里
        if all_years_dfs:
            mega_df = pd.concat(all_years_dfs, ignore_index=True)
            mega_df.to_csv(output_path, index=False)
            print(f"\n✅ 大功告成！十年的全部台风特征已合并保存至: {output_path}")
            print(f"总计生成了 {len(mega_df)} 行 0.5小时精度的台风状态记录！")
        else:
            print("未能成功处理任何文件，请检查 txt 内容格式。")