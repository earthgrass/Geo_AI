import numpy as np
import matplotlib.pyplot as plt
import os
import math
# 直接从你之前的引擎脚本中导入核心类
from step2_3_generate_metrix import PINN_InferenceEngine, parse_cma_with_heading

# ==========================================
# 1. 气象顶级画图配置
# ==========================================
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['font.family'] = 'STIXGeneral'

# ==========================================
# 2. 轨迹篡改函数 (生成实验组)
# ==========================================
def modify_track(base_track, delta_p=0, delta_lat=0, speed_ratio=1.0):
    """基于基准轨迹，施加物理扰动"""
    new_track = []
    curr_lat, curr_lon = base_track[0][0], base_track[0][1]
    
    for i, pt in enumerate(base_track):
        # pt: [lat, lon, pres, wind, time_str, heading]
        if i == 0:
            lat = pt[0] + delta_lat
            lon = pt[1] - delta_lat * 0.5 # 北抬同时略微西移
            new_track.append([lat, lon, max(880, pt[2] - delta_p), pt[3] * (1 + delta_p*0.01), pt[4], pt[5]])
            curr_lat, curr_lon = lat, lon
        else:
            prev_pt = base_track[i-1]
            d_lat = (pt[0] - prev_pt[0]) * speed_ratio
            d_lon = (pt[1] - prev_pt[1]) * speed_ratio
            curr_lat += d_lat
            curr_lon += d_lon
            new_track.append([curr_lat, curr_lon, max(880, pt[2] - delta_p), pt[3] * (1 + delta_p*0.01), pt[4], pt[5]])
    return new_track

# ==========================================
# 3. 核心遍历与画图逻辑
# ==========================================
def run_hardcore_sensitivity():
    print("🚀 启动 100% 真实 AI 模型敏感性遍历...")
    if not os.path.exists("typhoon_convlstm_best.pth"):
        print("🚨 找不到模型权重 typhoon_convlstm_best.pth！")
        return
        
    engine = PINN_InferenceEngine("typhoon_convlstm_best.pth")
    base_track = parse_cma_with_heading("CH2024BST.txt", "KONG-REY")
    
    # 实验组设定
    press_drops = [0, 5, 10, 15, 20, 25]       # 气压深切 (热力)
    lat_shifts = [0, 1.5, 3.0, 4.5, 6.0, 7.5]  # 纬度北抬 (动力)
    speed_ratios = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5] # 移速比例 (滞留)
    
    metrics = {'P': {'x': press_drops, 'p_max': [], 's_ext': []},
               'L': {'x': lat_shifts, 'p_max': [], 's_ext': []},
               'S': {'x': [100*(1-s) for s in speed_ratios], 'p_max': [], 's_ext': []}}

    # --- 实验 1: 热力强迫 (气压) ---
    print("\n>>> 开始实验 1/3: 遍历热力强迫 (气压深切) ...")
    for dp in press_drops:
        track = modify_track(base_track, delta_p=dp)
        name = f"EXP_P_{dp}"
        engine.run(name, track, enable_topo=True)
        data = np.load(f"{name}_DataPackage.npz")['matrices']
        # 提取真实指标: 全局最高降水, 超过 20mm/h 的极端面积 (像素数*100km^2)
        metrics['P']['p_max'].append(np.max(data))
        metrics['P']['s_ext'].append(np.max(np.sum(data >= 20.0, axis=(1,2))) * 100 / 10000) 
        os.remove(f"{name}_DataPackage.npz") # 阅后即焚，节省硬盘

    # --- 实验 2: 动力强迫 (纬度) ---
    print("\n>>> 开始实验 2/3: 遍历动力强迫 (纬度北抬) ...")
    for dl in lat_shifts:
        track = modify_track(base_track, delta_lat=dl)
        name = f"EXP_L_{dl}"
        engine.run(name, track, enable_topo=True)
        data = np.load(f"{name}_DataPackage.npz")['matrices']
        metrics['L']['p_max'].append(np.max(data))
        metrics['L']['s_ext'].append(np.max(np.sum(data >= 20.0, axis=(1,2))) * 100 / 10000)
        os.remove(f"{name}_DataPackage.npz")

    # --- 实验 3: 滞留效应 (移速) ---
    print("\n>>> 开始实验 3/3: 遍历滞留效应 (移速减慢) ...")
    for sr in speed_ratios:
        track = modify_track(base_track, speed_ratio=sr)
        name = f"EXP_S_{sr}"
        engine.run(name, track, enable_topo=True)
        data = np.load(f"{name}_DataPackage.npz")['matrices']
        metrics['S']['p_max'].append(np.max(data))
        # 持续时间近似计算: 高强度降水帧数 * 6小时
        duration = np.sum(np.max(data, axis=(1,2)) > 30.0) * 6 
        metrics['S']['s_ext'].append(duration) 
        os.remove(f"{name}_DataPackage.npz")

    # ==========================================
    # 4. 绘制带有真实散点的折线图 (已修复符号解析 Bug)
    # ==========================================
    print("\n[*] 真实推演完成，正在绘制图表...")
    
    # 图 1: 热力强迫
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(metrics['P']['x'], metrics['P']['p_max'], 'o-', color='#cb181d', linewidth=2.5, markersize=8, label='Peak Intensity ($P_{max}$)')
    ax1.set_xlabel(r'Central Pressure Deepening $\Delta P$ (-hPa) [Thermal Forcing]', fontweight='bold')
    ax1.set_ylabel('Real AI Predicted Peak Intensity (mm/h)', color='#cb181d', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#cb181d')
    ax2 = ax1.twinx()
    ax2.plot(metrics['P']['x'], metrics['P']['s_ext'], 's--', color='#2171b5', linewidth=2.5, markersize=8, label='Extreme Area ($S_{ext}$)')
    # 【修复点】：使用 r 原始字符串和 \geq
    ax2.set_ylabel(r'Real AI Predicted Area $\geq 20$ mm/h ($10^4$ km$^2$)', color='#2171b5', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#2171b5')
    plt.title('True Sensitivity: Thermal Forcing vs. Precipitation Outputs', fontweight='bold', pad=15)
    fig.legend(loc='upper left', bbox_to_anchor=(0.15, 0.85))
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig('Fig4_6a_Real_Sensitivity_Pressure.png', bbox_inches='tight')

    # 图 2: 动力强迫
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(metrics['L']['x'], metrics['L']['p_max'], 'o-', color='#cb181d', linewidth=2.5, markersize=8, label='Peak Intensity ($P_{max}$)')
    ax1.set_xlabel(r'Poleward Latitudinal Shift $\Delta \phi$ (+Degree) [Dynamic Forcing]', fontweight='bold')
    ax1.set_ylabel('Real AI Predicted Peak Intensity (mm/h)', color='#cb181d', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#cb181d')
    ax2 = ax1.twinx()
    ax2.plot(metrics['L']['x'], metrics['L']['s_ext'], 's--', color='#2171b5', linewidth=2.5, markersize=8, label='Extreme Area ($S_{ext}$)')
    # 【修复点】：使用 r 原始字符串和 \geq
    ax2.set_ylabel(r'Real AI Predicted Area $\geq 20$ mm/h ($10^4$ km$^2$)', color='#2171b5', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#2171b5')
    plt.title('True Sensitivity: Dynamic Forcing vs. Precipitation Outputs', fontweight='bold', pad=15)
    fig.legend(loc='upper left', bbox_to_anchor=(0.15, 0.85))
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig('Fig4_6b_Real_Sensitivity_Latitude.png', bbox_inches='tight')

    # 图 3: 移速与持续时间
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(metrics['S']['x'], metrics['S']['p_max'], 'o-', color='#cb181d', linewidth=2.5, markersize=8, label='Peak Intensity ($P_{max}$)')
    ax1.set_xlabel('Translation Speed Reduction (%) [Accumulation Effect]', fontweight='bold')
    ax1.set_ylabel('Real AI Predicted Peak Intensity (mm/h)', color='#cb181d', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#cb181d')
    ax2 = ax1.twinx()
    ax2.plot(metrics['S']['x'], metrics['S']['s_ext'], '^-.', color='#41ab5d', linewidth=2.5, markersize=8, label='Hazard Duration ($T_{risk}$)')
    # 【修复点】：使用 r 原始字符串和 \geq
    ax2.set_ylabel(r'Hazard Duration $P_{max} \geq 30$ mm/h (Hours)', color='#41ab5d', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#41ab5d')
    plt.title('True Sensitivity: Translation Slowdown vs. Hazard Duration', fontweight='bold', pad=15)
    fig.legend(loc='upper left', bbox_to_anchor=(0.15, 0.85))
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig('Fig4_6c_Real_Sensitivity_Speed.png', bbox_inches='tight')

    print(" [★] 3 张极其硬核的 100% 真实敏感性分析图生成完毕！")
if __name__ == "__main__":
    run_hardcore_sensitivity()