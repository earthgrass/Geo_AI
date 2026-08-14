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
# 1. 气象顶级画图全局配置
# ==========================================
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['font.family'] = 'STIXGeneral'

# 修复色卡：最低一级透明，凸显底图
precip_colors = [
    "#e4f3d8", "#bde9bf", "#9ed0a0", "#7fb882", "#418849", "#1c712e", 
    "#f7f370", "#fecb5a", "#ffa146", "#ff8b3c", "#e31a1c", "#b10026", "#800026"
]
precip_cmap = colors.ListedColormap(precip_colors)
clev_precip = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 80.0]
norm = colors.BoundaryNorm(clev_precip, len(precip_colors))
outline_effect = [patheffects.withStroke(linewidth=2.5, foreground='w')]

# 放大空间尺度 
CHINA_COAST_EXTENT = [112, 130, 18, 34]

# ==========================================
# 2. 提取 V-COMPOUND 的 T2 和 T3 时刻
# ==========================================
def render_china_landfall_evolution():
    file_path = "V-COMPOUND_DataPackage.npz"
    if not os.path.exists(file_path):
        print(f"🚨 找不到 {file_path}！")
        return
        
    print("[*] 正在生成大尺度陆地致灾演变图 (T2 & T3)...")
    data = np.load(file_path)
    matrices, lons, lats = data['matrices'], data['lons'], data['lats']
    times, presses = data['times'], data['presses']
    
    N = len(matrices)
    # 定义我们要提取的两个核心节点
    phases = [
        ('T2', N // 3, 'Pre-landfall Intensification Phase'),
        ('T3', 2 * N // 3, 'Peak Landfall & Topo-Forcing Phase')
    ]
    
    for phase_name, idx, desc in phases:
        matrix = gaussian_filter(matrices[idx], sigma=1.0)
        center_lon, center_lat = lons[idx], lats[idx]
        
        fig = plt.figure(figsize=(11, 9))
        ax = plt.axes(projection=ccrs.PlateCarree())
        
        # 绘制高精度中国底图
        ax.set_extent(CHINA_COAST_EXTENT, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '10m'), facecolor='#f5f5f5', edgecolor='none', zorder=0)
        ax.add_feature(cfeature.OCEAN.with_scale('10m'), facecolor='#e0f3f8', zorder=0)
        ax.add_feature(cfeature.BORDERS.with_scale('10m'), linestyle=':', linewidth=1.5, edgecolor='gray', zorder=1)
        ax.coastlines(resolution='10m', linewidth=1.2, color='#333333', zorder=1)
        
        ax.set_xticks(np.arange(112, 131, 3), crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(18, 35, 3), crs=ccrs.PlateCarree())
        ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())
        ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
        
        # 降水矩阵叠加
        lon_mesh, lat_mesh = np.meshgrid(np.linspace(center_lon-5.75, center_lon+5.75, 128),
                                         np.linspace(center_lat-5.75, center_lat+5.75, 128))
        cf = ax.contourf(lon_mesh, lat_mesh, matrix, clev_precip, norm=norm, 
                         cmap=precip_cmap, transform=ccrs.PlateCarree(), extend='max', alpha=0.85, zorder=2)
        
        # 城市标注
        cities = {'Fuzhou': (119.3, 26.0), 'Wenzhou': (120.6, 28.0), 'Taipei': (121.5, 25.0), 'Xiamen': (118.1, 24.5)}
        for city, (clon, clat) in cities.items():
            ax.plot(clon, clat, 'ko', markersize=4, transform=ccrs.PlateCarree(), zorder=3)
            ax.text(clon+0.1, clat+0.1, city, fontsize=10, fontweight='bold', transform=ccrs.PlateCarree(), path_effects=outline_effect, zorder=3)
            
        # L 标注 (修复为圆点+文字)
        ax.plot(center_lon, center_lat, marker='o', color='black', markersize=6, transform=ccrs.PlateCarree(), zorder=4)
        ax.text(center_lon, center_lat, 'L', color='red', size=24, ha='center', va='center', fontweight='bold', transform=ccrs.PlateCarree(), path_effects=outline_effect, zorder=5)
        ax.text(center_lon, center_lat-0.5, f'{presses[idx]:.0f} hPa', color='red', size=12, ha='center', va='top', fontweight='bold', transform=ccrs.PlateCarree(), path_effects=outline_effect, zorder=5)
        
        # 历史轨迹
        ax.plot(lons[:idx+1], lats[:idx+1], color='darkred', linewidth=2.5, linestyle='--', transform=ccrs.PlateCarree(), zorder=3)

        cbar = plt.colorbar(cf, ax=ax, orientation='horizontal', pad=0.06, shrink=0.85)
        cbar.set_label('Simulated Extreme Precipitation Intensity (mm/h)', fontsize=12, fontweight='bold')
        
        plt.title(f'Future Compound Typhoon Landfall Evolution ({phase_name})\nTime: {times[idx]} | {desc}', 
                  fontsize=14, fontweight='bold', pad=15)
        
        save_name = f'Fig4_5_China_Landfall_Risk_Expanded_{phase_name}.png'
        plt.savefig(save_name, bbox_inches='tight')
        plt.close()
        print(f" [★] 成功生成: {save_name}")

if __name__ == "__main__":
    render_china_landfall_evolution()