import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap

# ==========================================
# 模拟加载你的模型输出数据 (请替换为你真实的预测矩阵)
# 假设矩阵大小为 128x128，对应经纬度范围: 经度 115E-125E, 纬度 22N-32N (覆盖福建/浙江/上海)
# ==========================================
lon_min, lon_max = 115, 126
lat_min, lat_max = 22, 32
lons = np.linspace(lon_min, lon_max, 128)
lats = np.linspace(lat_min, lat_max, 128)
lon2d, lat2d = np.meshgrid(lons, lats)

# 生成一个模拟的“典型未来超强虚拟台风”降水场 (V-HIGHLAT + V-DEEP)
# 降水中心偏向右前象限，受地形抬升影响有局部极值
distance = np.sqrt((lon2d - 120.5)**2 + (lat2d - 27.5)**2)
precipitation = np.exp(-distance**2 / 2.0) * 55.0  # 最高降水率约 55 mm/h
precipitation += np.random.normal(0, 2, (128, 128)) # 加入物理残差噪声
precipitation = np.clip(precipitation, 0, None)

# 自定义气象降水色带 (从浅绿到深红)
colors = ['#ffffff', '#a1d99b', '#31a354', '#fec44f', '#fc9272', '#de2d26', '#a50f15', '#4a1486']
cmap_precip = LinearSegmentedColormap.from_list("precip", colors, N=256)

# ==========================================
# 开始绘制高逼格气象地图
# ==========================================
fig = plt.figure(figsize=(10, 8), dpi=300)
# 使用 PlateCarree 投影
ax = plt.axes(projection=ccrs.PlateCarree())

# 添加地理特征
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1.2, edgecolor='black')
ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=1.0, linestyle=':')
ax.add_feature(cfeature.OCEAN, facecolor='#e0f3f8')
ax.add_feature(cfeature.LAND, facecolor='#f0f0f0')

# 限制显示范围为中国东南沿海
ax.set_extent([115, 126, 22, 32], crs=ccrs.PlateCarree())

# 绘制降水等值面图
contour = ax.contourf(lons, lats, precipitation, levels=np.arange(0, 60, 5), 
                      cmap=cmap_precip, transform=ccrs.PlateCarree(), extend='max', alpha=0.85)

# 添加颜色条
cbar = plt.colorbar(contour, ax=ax, orientation='vertical', pad=0.04, shrink=0.8)
cbar.set_label('Predicted Precipitation Intensity (mm/h)', fontsize=12, fontweight='bold')

# 添加台风中心标记
ax.plot(120.5, 27.5, marker='L', color='red', markersize=15, transform=ccrs.PlateCarree(), fontweight='bold')
ax.text(120.7, 27.3, 'V-HIGHLAT\n910 hPa', color='red', fontsize=12, fontweight='bold', transform=ccrs.PlateCarree())

# 绘制网格线
gl = ax.gridlines(draw_labels=True, linewidth=0.8, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False

plt.title('Spatial Distribution of Typical Future Virtual Typhoon in Eastern China', fontsize=14, pad=15, fontweight='bold')
plt.savefig('Virtual_Typhoon_China.png', bbox_inches='tight')
plt.show()