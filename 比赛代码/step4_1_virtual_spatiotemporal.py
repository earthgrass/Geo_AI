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
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.family'] = 'STIXGeneral'

# 降水专属色卡 (ERA5 风格)
precip_colors = [
    "#bde9bf", "#adddb0", "#9ed0a0", "#8ec491", "#7fb882", "#70ac74", 
    "#60a065", "#519457", "#418849", "#307c3c", "#1c712e", "#f7f370", 
    "#fbdf65", "#fecb5a", "#ffb650", "#ffa146", "#ff8b3c", "#ff8b3c", "#ff8b5c", "#ff8b5c"
]
precip_cmap = colors.ListedColormap(precip_colors)
clev_precip = np.concatenate((np.arange(0.1, 1, .1), np.arange(1, 2, .2), np.arange(2, 9, 1)))
norm = colors.BoundaryNorm(clev_precip, 20)
outline_effect = [patheffects.withStroke(linewidth=2.5, foreground='w')]

# 宏观合成图的统一物理边界 (保证 4 种情况大尺度绝对一致)
GLOBAL_EXTENT = [110, 150, 10, 55] 

def format_map(ax, extents, draw_grid_labels=True):
    """通用底图生成器 (微观与宏观共用)"""
    ax.set_extent(extents, crs=ccrs.PlateCarree())
    land = cfeature.NaturalEarthFeature('physical', 'land', '50m')
    ax.add_feature(land, facecolor='#f0f0f0', zorder=-1, alpha=0.8)
    ax.add_feature(cfeature.OCEAN, facecolor='#e0f3f8', zorder=-1)
    ax.coastlines(linewidth=1.2, color='#333333')
    
    if draw_grid_labels:
        # 微观图使用 5度间隔
        ax.set_xticks(np.arange(extents[0], extents[1]+1, 5), crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(extents[2], extents[3]+1, 5), crs=ccrs.PlateCarree())
        ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())
        ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
        ax.minorticks_on()
        ax.tick_params(which='major', length=6, width=1.2, top=True, right=True, direction='in')
    else:
        # 宏观图使用 10度间隔，并用网格线
        ax.set_xticks(np.arange(extents[0], extents[1]+1, 10), crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(extents[2], extents[3]+1, 10), crs=ccrs.PlateCarree())
        ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())
        ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
        ax.minorticks_on()
        ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')

# ==========================================
# 2. 生成微观 4 宫格切片 (追踪台风中心)
# ==========================================
def render_micro_4panel(name, matrices, lons, lats, times, presses, desc):
    N = len(matrices)
    idx_list = [0, N//3, 2*N//3, N-1]
    titles = ['T1: Open Ocean (Initial)', 'T2: Intensification Phase', 
              'T3: Peak Interaction', 'T4: Decay / Late Stage']

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), subplot_kw={'projection': ccrs.PlateCarree()})
    
    for i, (ax, idx) in enumerate(zip(axes.flat, idx_list)):
        smooth_p = gaussian_filter(matrices[idx], sigma=1.2)
        # 视口跟随台风中心，外扩 8 度
        extent = [lons[idx]-8, lons[idx]+8, lats[idx]-8, lats[idx]+8]
        format_map(ax, extent, draw_grid_labels=True)
        
        # 真实降水矩阵映射
        lon_mesh, lat_mesh = np.meshgrid(np.linspace(lons[idx]-5.75, lons[idx]+5.75, 128),
                                         np.linspace(lats[idx]-5.75, lats[idx]+5.75, 128))
        cf = ax.contourf(lon_mesh, lat_mesh, smooth_p, clev_precip, norm=norm, 
                         cmap=precip_cmap, transform=ccrs.PlateCarree(), extend='max')
        
        # 标注
        ax.text(lons[idx], lats[idx], 'L', color='red', size=22, ha='center', va='center', 
                transform=ccrs.PlateCarree(), path_effects=outline_effect, fontweight='bold')
        ax.text(lons[idx], lats[idx]-0.8, f'{presses[idx]:.0f} hPa', color='red', size=12, 
                ha='center', va='top', fontweight='bold', transform=ccrs.PlateCarree(), path_effects=outline_effect)
        
        if idx > 0:
            ax.plot(lons[:idx+1], lats[:idx+1], color='blue', linewidth=1.5, linestyle='--', transform=ccrs.PlateCarree(), alpha=0.7)
        ax.set_title(f'{titles[i]}\nTime: {times[idx]}', fontsize=12, fontweight='bold')

    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])
    cbar = fig.colorbar(cf, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Predicted Precipitation (mm/h)', fontsize=13, fontweight='bold')
    plt.suptitle(f'Micro-Spatiotemporal Evolution: {name}\n[{desc}]', fontsize=18, fontweight='bold', y=0.96)
    
    save_name = f'Fig4_1_Micro_{name}.png'
    plt.savefig(save_name, bbox_inches='tight')
    plt.close()

# ==========================================
# 3. 生成宏观大尺度合成图 (统一边界)
# ==========================================
def render_macro_composite(name, matrices, lons, lats, times, presses, desc):
    fig = plt.figure(figsize=(10, 11))
    ax = plt.axes(projection=ccrs.PlateCarree())
    format_map(ax, GLOBAL_EXTENT, draw_grid_labels=False)
    
    # 画底部的全路径虚线
    ax.plot(lons, lats, color='dimgray', linewidth=2, linestyle='--', transform=ccrs.PlateCarree(), zorder=2)
    
    N = len(matrices)
    idx_list = [0, N//3, 2*N//3, N-1]
    labels = ['T1', 'T2', 'T3', 'T4']
    cf = None
    
    for i, idx in enumerate(idx_list):
        smooth_p = gaussian_filter(matrices[idx], sigma=1.2)
        lon_mesh, lat_mesh = np.meshgrid(np.linspace(lons[idx]-5.75, lons[idx]+5.75, 128),
                                         np.linspace(lats[idx]-5.75, lats[idx]+5.75, 128))
        cf = ax.contourf(lon_mesh, lat_mesh, smooth_p, clev_precip, norm=norm, 
                         cmap=precip_cmap, transform=ccrs.PlateCarree(), extend='max', alpha=0.9, zorder=3)
        
        ax.plot(lons[idx], lats[idx], marker='o', color='black', markersize=6, transform=ccrs.PlateCarree(), zorder=4)
        ax.text(lons[idx]+1.0, lats[idx]-1.0, labels[i], color='red', fontsize=14, 
                fontweight='bold', transform=ccrs.PlateCarree(), path_effects=outline_effect, zorder=5)

    cbar = plt.colorbar(cf, ax=ax, orientation='horizontal', pad=0.06, shrink=0.85)
    cbar.set_label('Precipitation Footprint Composite (mm/h)', fontsize=12, fontweight='bold')
    plt.title(f'Macro Full Trajectory & Footprint: {name}', fontsize=16, fontweight='bold', pad=20)
    plt.text(0.5, 1.01, f'[{desc}]', ha='center', va='bottom', transform=ax.transAxes, fontsize=12, color='gray')
    
    save_name = f'Fig4_4_Macro_{name}.png'
    plt.savefig(save_name, bbox_inches='tight')
    plt.close()

# ==========================================
# 4. 自动化批处理执行入口
# ==========================================
if __name__ == "__main__":
    # 完美匹配严谨版 IPCC 气候变暖四大推演情景
    scenario_info = {
        "V-SHIFT": "Pure Path Shift (Dynamics Anomaly)",
        "V-INTENSE": "Rapid Intensification (Thermal Forcing)",
        "V-COMPOUND": "Compound Extreme (Shift + Intensification)",
        "V-SLOW": "Translation Slowdown (Accumulation Effect)"
    }
    
    print("🚀 开始一键批量生成所有微观与宏观对比图 (共计 8 张)...")
    
    for name, desc in scenario_info.items():
        file_path = f"{name}_DataPackage.npz"
        if not os.path.exists(file_path):
            print(f"🚨 跳过 {name}: 找不到数据文件 {file_path}，请确认 step2_3 是否跑出了该文件。")
            continue
            
        print(f"[*] 正在处理: {name} ...")
        # 仅加载一次数据
        data = np.load(file_path)
        matrices, lons, lats = data['matrices'], data['lons'], data['lats']
        times, presses = data['times'], data['presses']
        
        # 任务 1：生成 4 宫格微观图
        render_micro_4panel(name, matrices, lons, lats, times, presses, desc)
        
        # 任务 2：生成大尺度宏观全图
        render_macro_composite(name, matrices, lons, lats, times, presses, desc)
        
    print("\n🎉 大功告成！8 张严谨版论文级神图已全部生成至当前目录！")