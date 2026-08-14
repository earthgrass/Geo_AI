import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
from matplotlib import patheffects
from scipy.ndimage import gaussian_filter
import os

# ==========================================
# 1. 气象顶级画图配置 (直接复刻 ERA5 代码精华)
# ==========================================
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.family'] = 'STIXGeneral'

# 顶级降水色卡
precip_colors = [
    "#bde9bf", "#adddb0", "#9ed0a0", "#8ec491", "#7fb882", "#70ac74", 
    "#60a065", "#519457", "#418849", "#307c3c", "#1c712e", "#f7f370", 
    "#fbdf65", "#fecb5a", "#ffb650", "#ffa146", "#ff8b3c", "#ff8b3c", "#ff8b5c", "#ff8b5c"
]
precip_cmap = colors.ListedColormap(precip_colors)
clev_precip = np.concatenate((np.arange(0.1, 1, .1), np.arange(1, 2, .2), np.arange(2, 9, 1)))
norm = colors.BoundaryNorm(clev_precip, 20)

def format_map(ax, extents):
    """统一的地图底图生成器：绘制海陆边界、经纬网格"""
    ax.set_extent(extents, crs=ccrs.PlateCarree())
    # 增加海陆分布
    land = cfeature.NaturalEarthFeature('physical', 'land', '50m')
    ax.add_feature(land, facecolor='lightgray', zorder=-1, alpha=0.6)
    ax.coastlines(linewidth=1.0, color='#333333')
    
    # 经纬度刻度
    ax.set_xticks(np.arange(extents[0], extents[1]+1, 10), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(extents[2], extents[3]+1, 5), crs=ccrs.PlateCarree())
    ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())
    ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
    ax.minorticks_on()
    ax.tick_params(which='major', length=6, width=1.2, top=True, right=True, direction='in')

# ==========================================
# 2. 核心功能: 渲染 5 类图表
# ==========================================
def render_typhoon(name):
    print(f"[*] 正在为 {name} 渲染专业气象图...")
    data = np.load(f"{name}_Data.npz")
    mats, lons, lats = data['matrices'], data['lons'], data['lats']
    times, presses = data['times'], data['presses']
    
    out_dir = f"./Plots_Output/{name}_Frames"
    os.makedirs(out_dir, exist_ok=True)
    intensities = []
    
    # 全局大网格范围设定 (西太平洋)
    global_extent = [105, 150, 5, 40]
    outline_effect = [patheffects.withStroke(linewidth=2.5, foreground='w')]

    for i in range(len(mats)):
        smooth_p = gaussian_filter(mats[i], sigma=1.2) # 高斯平滑
        intensities.append(np.max(smooth_p))
        
        # 【输出1/3】: 按固定时间间隔(这里我们每一帧都画，对应 CMA 原始的 6 小时)
        fig = plt.figure(figsize=(11, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        format_map(ax, global_extent)
        
        # 👉 网格映射核心魔法：算出 128x128 局部矩阵的真实经纬度网格
        lon_1d = np.linspace(lons[i]-5.75, lons[i]+5.75, 128)
        lat_1d = np.linspace(lats[i]-5.75, lats[i]+5.75, 128)
        lon_mesh, lat_mesh = np.meshgrid(lon_1d, lat_1d)
        
        # 将局部网格投影到大地图上
        cf = ax.contourf(lon_mesh, lat_mesh, smooth_p, clev_precip, norm=norm, 
                         cmap=precip_cmap, transform=ccrs.PlateCarree(), extend='max')
        
        # 仿 ERA5 绘制红色的 L (低压中心) 和气压值
        ax.text(lons[i], lats[i], 'L', color='red', size=24, ha='center', va='center', 
                transform=ccrs.PlateCarree(), path_effects=outline_effect)
        ax.text(lons[i], lats[i]-0.8, f'{presses[i]:.0f} hPa', color='red', size=11, 
                ha='center', va='top', fontweight='bold', transform=ccrs.PlateCarree(), path_effects=outline_effect)
        
        # 绘制之前的轨迹
        if i > 0:
            ax.plot(lons[:i+1], lats[:i+1], color='blue', linewidth=1.5, linestyle='--', transform=ccrs.PlateCarree(), alpha=0.7)

        cbar = plt.colorbar(cf, ax=ax, orientation='vertical', pad=0.02, shrink=0.8)
        cbar.set_label('Predicted Precip. (mm/h)', fontweight='bold')
        
        plt.title(f'Simulated Precipitation: Typhoon {name}', loc='left', fontsize=14, fontweight='bold')
        plt.title(f'Time: {times[i]}', loc='right', fontsize=12)
        plt.savefig(f"{out_dir}/Frame_{times[i]}.png", bbox_inches='tight')
        plt.close()

    # 【输出2/4】: 折线图
    plt.figure(figsize=(10, 4))
    plt.plot(range(len(intensities)), intensities, color='#1c712e', linewidth=2.5, marker='o', markersize=4)
    plt.fill_between(range(len(intensities)), intensities, color='#1c712e', alpha=0.2)
    plt.title(f"Precipitation Intensity Trend: {name}", fontweight='bold')
    plt.ylabel("Precipitation (mm/h)")
    plt.xlabel("Time Steps (6h intervals)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(f"./Plots_Output/Trend_{name}.png", bbox_inches='tight')
    plt.close()

# 【输出5】: 双台风同框巅峰对比图
def render_combined():
    print("[*] 正在生成双台风同框对比宏观图...")
    k_data = np.load("KONG-REY_Data.npz")
    m_data = np.load("MAN-YI_Data.npz")
    
    # 找到各自降水最大的巅峰时刻
    k_idx = np.argmax([np.max(m) for m in k_data['matrices']])
    m_idx = np.argmax([np.max(m) for m in m_data['matrices']])
    
    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    format_map(ax, [105, 145, 5, 45])
    
    # 渲染康妮
    k_lon, k_lat = np.meshgrid(np.linspace(k_data['lons'][k_idx]-5.75, k_data['lons'][k_idx]+5.75, 128),
                               np.linspace(k_data['lats'][k_idx]-5.75, k_data['lats'][k_idx]+5.75, 128))
    ax.contourf(k_lon, k_lat, gaussian_filter(k_data['matrices'][k_idx], 1.2), clev_precip, norm=norm, cmap=precip_cmap, alpha=0.9)
    ax.plot(k_data['lons'], k_data['lats'], color='darkred', linestyle='-', linewidth=2, transform=ccrs.PlateCarree())
    ax.text(k_data['lons'][k_idx], k_data['lats'][k_idx]+1.5, 'KONG-REY', color='white', weight='bold', bbox=dict(facecolor='red', alpha=0.7), transform=ccrs.PlateCarree())

    # 渲染万宜
    m_lon, m_lat = np.meshgrid(np.linspace(m_data['lons'][m_idx]-5.75, m_data['lons'][m_idx]+5.75, 128),
                               np.linspace(m_data['lats'][m_idx]-5.75, m_data['lats'][m_idx]+5.75, 128))
    ax.contourf(m_lon, m_lat, gaussian_filter(m_data['matrices'][m_idx], 1.2), clev_precip, norm=norm, cmap=precip_cmap, alpha=0.9)
    ax.plot(m_data['lons'], m_data['lats'], color='darkblue', linestyle='-', linewidth=2, transform=ccrs.PlateCarree())
    ax.text(m_data['lons'][m_idx], m_data['lats'][m_idx]-1.5, 'MAN-YI', color='white', weight='bold', bbox=dict(facecolor='blue', alpha=0.7), transform=ccrs.PlateCarree())

    plt.title("Spatial Precipitation Distribution of Typhoons KONG-REY & MAN-YI (2024)", fontsize=15, fontweight='bold', pad=15)
    plt.savefig("./Plots_Output/Combined_Peak_Map.png", bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    os.makedirs("./Plots_Output", exist_ok=True)
    render_typhoon("KONG-REY")
    render_typhoon("MAN-YI")
    render_combined()
    print(" [★] 大功告成！所有专业图表均已输出至 Plots_Output 文件夹！")