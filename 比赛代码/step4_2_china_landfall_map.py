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

# 修复色卡：去掉纯白色，第一级用极浅且半透明的绿色
# 这样低于 0.1 的区域会完全透明，露出底下的陆地和海洋！
precip_colors = [
    "#e4f3d8", "#bde9bf", "#9ed0a0", "#7fb882", "#418849", "#1c712e", 
    "#f7f370", "#fecb5a", "#ffa146", "#ff8b3c", "#e31a1c", "#b10026", "#800026"
]
precip_cmap = colors.ListedColormap(precip_colors)
clev_precip = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 80.0]
norm = colors.BoundaryNorm(clev_precip, len(precip_colors))
outline_effect = [patheffects.withStroke(linewidth=2.5, foreground='w')]

# 放大空间尺度 (南至吕宋海峡，北至江苏，西至内陆江西/湖南)
CHINA_COAST_EXTENT = [112, 130, 18, 34]

# ==========================================
# 2. 提取 V-COMPOUND 的巅峰时刻并渲染
# ==========================================
def render_china_landfall():
    file_path = "V-COMPOUND_DataPackage.npz"
    if not os.path.exists(file_path):
        print(f"🚨 找不到 {file_path}！请确认 step2_3 是否跑出了该文件。")
        return
        
    print("[*] 正在生成大尺度陆地+海洋高精度致灾图...")
    data = np.load(file_path)
    matrices, lons, lats = data['matrices'], data['lons'], data['lats']
    times, presses = data['times'], data['presses']
    
    # 提取 T3 (巅峰碰撞期) 时刻
    idx = int(len(matrices) * 0.66)
    matrix = gaussian_filter(matrices[idx], sigma=1.0)
    center_lon, center_lat = lons[idx], lats[idx]
    
    fig = plt.figure(figsize=(11, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # 绘制高精度中国底图 (zorder=0 锁定在最底层)
    ax.set_extent(CHINA_COAST_EXTENT, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '10m'), facecolor='#f5f5f5', edgecolor='none', zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale('10m'), facecolor='#e0f3f8', zorder=0)
    
    # 海岸线和国界线 (zorder=1)
    ax.add_feature(cfeature.BORDERS.with_scale('10m'), linestyle=':', linewidth=1.5, edgecolor='gray', zorder=1)
    ax.coastlines(resolution='10m', linewidth=1.2, color='#333333', zorder=1)
    
    # 经纬网格刻度放大到 3 度一格
    ax.set_xticks(np.arange(112, 131, 3), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(18, 35, 3), crs=ccrs.PlateCarree())
    ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())
    ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
    
    # 映射降水矩阵
    lon_mesh, lat_mesh = np.meshgrid(np.linspace(center_lon-5.75, center_lon+5.75, 128),
                                     np.linspace(center_lat-5.75, center_lat+5.75, 128))
    
    # 绘制等值面图 (zorder=2，透明度 0.85，叠加在陆地之上)
    cf = ax.contourf(lon_mesh, lat_mesh, matrix, clev_precip, norm=norm, 
                     cmap=precip_cmap, transform=ccrs.PlateCarree(), extend='max', alpha=0.85, zorder=2)
    
    # 标注重要城市 (增加了厦门、杭州等扩大视野后的参照物)
    cities = {'Fuzhou': (119.3, 26.0), 'Wenzhou': (120.6, 28.0), 'Taipei': (121.5, 25.0), 'Xiamen': (118.1, 24.5)}
    for city, (clon, clat) in cities.items():
        ax.plot(clon, clat, 'ko', markersize=4, transform=ccrs.PlateCarree(), zorder=3)
        ax.text(clon+0.1, clat+0.1, city, fontsize=10, fontweight='bold', transform=ccrs.PlateCarree(), path_effects=outline_effect, zorder=3)
        
    # 绘制大写的红色 L 中心
    ax.plot(center_lon, center_lat, marker='o', color='black', markersize=6, transform=ccrs.PlateCarree(), zorder=4)
    ax.text(center_lon, center_lat, 'L', color='red', size=24, ha='center', va='center', fontweight='bold', transform=ccrs.PlateCarree(), path_effects=outline_effect, zorder=5)
    ax.text(center_lon, center_lat-0.5, f'{presses[idx]:.0f} hPa', color='red', size=12, ha='center', va='top', fontweight='bold', transform=ccrs.PlateCarree(), path_effects=outline_effect, zorder=5)
    
    # 绘制轨迹线
    ax.plot(lons[:idx+1], lats[:idx+1], color='darkred', linewidth=2.5, linestyle='--', transform=ccrs.PlateCarree(), zorder=3)

    # 添加色带
    cbar = plt.colorbar(cf, ax=ax, orientation='horizontal', pad=0.06, shrink=0.85)
    cbar.set_label('Simulated Extreme Precipitation Intensity (mm/h)', fontsize=12, fontweight='bold')
    
    plt.title(f'Simulated Landfall of Future Compound Typhoon in China\nTime: {times[idx]} | Expanded Spatial View', 
              fontsize=14, fontweight='bold', pad=15)
    
    plt.savefig('Fig4_5_China_Landfall_Risk_Expanded.png', bbox_inches='tight')
    print(" [★] 成功生成大尺度致灾图: Fig4_5_China_Landfall_Risk_Expanded.png")

if __name__ == "__main__":
    render_china_landfall()