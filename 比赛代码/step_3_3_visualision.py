import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 1. 整理四组虚拟情景数据 (实事求是拆分)
data = {
    'Scenario': ['BASE', 'V-HIGHLAT', 'V-DEEP', 'V-STRONG', 'V-SLOW'],
    'Peak_Intensity': [37.20, 49.41, 41.50, 43.25, 38.10],  # mm/h
    'Extreme_Area': [145.88, 151.20, 173.88, 165.42, 148.50] # 10^4 km^2
}
df = pd.DataFrame(data)

# 绘图设置
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax1 = plt.subplots(figsize=(11, 6), dpi=300)
x = np.arange(len(df['Scenario']))
width = 0.35

# 绘制左轴：降水极值
bars1 = ax1.bar(x - width/2, df['Peak_Intensity'], width, label='Peak Intensity (mm/h)', color='#c0392b', alpha=0.85)
ax1.set_ylabel('Peak Precipitation Intensity (mm/h)', fontsize=12, fontweight='bold')
ax1.set_ylim(0, 65)

# 绘制右轴：极端面积
ax2 = ax1.twinx()
bars2 = ax2.bar(x + width/2, df['Extreme_Area'], width, label='Extreme Area (10^4 km²)', color='#2980b9', alpha=0.85)
ax2.set_ylabel('Extreme Precipitation Area (10^4 km²)', fontsize=12, fontweight='bold')
ax2.set_ylim(0, 200)

# 标注与美化
ax1.set_xticks(x)
ax1.set_xticklabels(df['Scenario'], fontsize=11, fontweight='bold')
plt.title('Multi-Scenario Climate Sensitivity Analysis: 4 Extreme Variations', fontsize=14, pad=20)

def autolabel(bars, ax):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

autolabel(bars1, ax1)
autolabel(bars2, ax2)
fig.legend(loc='upper right', bbox_to_anchor=(0.88, 0.88))
plt.tight_layout()
plt.savefig('Climate_Scenario_Final.png')
plt.show()