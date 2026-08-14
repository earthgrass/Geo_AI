import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
import warnings
from sklearn.preprocessing import MinMaxScaler

# 屏蔽版本警告，保持控制台整洁
warnings.filterwarnings('ignore')

# ================= 1. 全局中英映射字典 (用于优化论文显示) =================
CN_MAP = {
    'Lat': '纬度 (°N)',
    'Lon': '经度 (°E)',
    'Wind_Speed': '最大风速 (m/s)',
    'Pressure': '中心气压 (hPa)',
    'Radius_max_wind_km': '最大风速半径 (km)',
    'Moving_Speed_kmh': '移动速度 (km/h)',
    'Moving_Direction': '移动方向 (deg)',
    'Curvature_deg_per_km': '路径曲率 (deg/km)',
    'Delta_P_6h': '6h气压变化 (hPa)',
    'Delta_V_6h': '6h风速变化 (m/s)',
    'P_total': '总降水量 (mm)',
    'P_max': '最大降水强度 (mm/h)',
    'S_ext_Extreme_over_20': '极端降水面积 (km²)',
    'D_offset_km': '降水中心偏移 (km)',
    'I_asy_Index': '非对称性指数',
    'Intensity_Level': '台风强度等级'
}

def translate(key):
    return CN_MAP.get(key, key)

# ================= 配置与数据加载 =================
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False
DPI = 300
SAVE_DIR = 'Results_Figures_Final'
if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

def load_data():
    df_full = pd.read_csv('Typhoon_Full_Dataset_Q1.csv')
    corr_matrix = pd.read_csv('1.4_Data_Full_Spearman_Matrix.csv', index_col=0)
    rf_importance = pd.read_csv('1.4_Data_RF_Raw_Importance.csv')
    shap_values = np.load('1.4_Data_SHAP_Matrix_S_ext.npy')
    shap_x = pd.read_csv('1.4_Data_SHAP_Base_Features.csv')
    return df_full, corr_matrix, rf_importance, shap_values, shap_x

# ================= Q1: 路径、强度与降水的宏观关系 (增强版) =================

def plot_q1_enhanced(corr_matrix, df_full):
    print("正在生成 Q1 增强版关系图表 (已加入中英映射)...")
    X_cols = ['Lat', 'Lon', 'Wind_Speed', 'Pressure', 'Radius_max_wind_km', 
              'Moving_Speed_kmh', 'Moving_Direction', 'Curvature_deg_per_km', 'Delta_P_6h', 'Delta_V_6h']
    Y_cols = ['P_total', 'P_max', 'S_ext_Extreme_over_20', 'D_offset_km', 'I_asy_Index']
    
    # 1.1 裁剪热力图 (中文坐标轴)
    plt.figure(figsize=(12, 9))
    sub_corr = corr_matrix.loc[X_cols, Y_cols]
    sub_corr.index = [translate(x) for x in sub_corr.index]
    sub_corr.columns = [translate(y) for y in sub_corr.columns]
    sns.heatmap(sub_corr, annot=True, cmap='RdBu_r', center=0, fmt=".2f")
    plt.title("台风动力参量与降水特征 Spearman 相关性热力图", fontsize=14)
    plt.savefig(f'{SAVE_DIR}/Fig1_1_Heatmap_CN.png', dpi=DPI, bbox_inches='tight')
    plt.close()

    # 1.2 联合分布密度图
    plt.figure(figsize=(8, 8))
    sns.jointplot(data=df_full, x='Wind_Speed', y='S_ext_Extreme_over_20', kind='reg', color='#4CB391')
    plt.xlabel(translate('Wind_Speed'))
    plt.ylabel(translate('S_ext_Extreme_over_20'))
    plt.savefig(f'{SAVE_DIR}/Fig1_2_Joint_RegPlot.png', dpi=DPI)
    plt.close()

    # 1.3 降水特征雷达图
    df_radar = df_full.copy()
    df_radar['Intensity_Level'] = pd.qcut(df_radar['Wind_Speed'], 3, labels=['弱台风', '中等台风', '强台风'])
    radar_data = df_radar.groupby('Intensity_Level')[Y_cols].mean()
    scaler = MinMaxScaler()
    radar_norm = pd.DataFrame(scaler.fit_transform(radar_data), columns=[translate(y) for y in Y_cols], index=radar_data.index)
    
    angles = np.linspace(0, 2 * np.pi, len(Y_cols), endpoint=False).tolist()
    angles += angles[:1] 

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for index, row in radar_norm.iterrows():
        values = row.tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=index)
        ax.fill(angles, values, alpha=0.25)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([translate(y) for y in Y_cols])
    plt.title("不同强度等级台风的降水特征多维画像对比", pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    plt.savefig(f'{SAVE_DIR}/Fig1_3_Radar_Chart_CN.png', dpi=DPI, bbox_inches='tight')
    plt.close()

# ================= Q2: 全特征依赖扫描 (高阶定制 + 中文映射) =================

def plot_q2_full_shap(rf_importance, shap_values, shap_x):
    print("正在生成 Q2 全特征 SHAP 阈值扫描 (中文响应曲线版)...")
    target = 'S_ext_Extreme_over_20'
    df_target_rf = rf_importance[rf_importance['Target_Variable'] == target].sort_values('Raw_Importance_Score', ascending=False)
    
    # 2.1 RF 贡献度排名
    plt.figure(figsize=(10, 6))
    df_plot_rf = df_target_rf.copy()
    df_plot_rf['Feature_Name_CN'] = df_plot_rf['Feature_Name'].apply(translate)
    sns.barplot(data=df_plot_rf, x='Raw_Importance_Score', y='Feature_Name_CN', 
                hue='Feature_Name_CN', palette='viridis', legend=False)
    plt.title(f"各特征对 [{translate(target)}] 的非线性贡献度排名")
    plt.savefig(f'{SAVE_DIR}/Fig2_1_RF_Importance_CN.png', dpi=DPI, bbox_inches='tight')
    plt.close()

    # 2.2 SHAP 摘要图 (由于shap库内部限制，此图保持英文特征名，但在论文中可对照字典解释)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, shap_x, show=False)
    plt.title("SHAP 全特征影响方向与分布密度图 (Global Explainer)")
    plt.savefig(f'{SAVE_DIR}/Fig2_2_SHAP_Summary.png', dpi=DPI, bbox_inches='tight')
    plt.close()

    # 2.3 自定义重构：全变量扫描依赖图 (前 8 名 + 自动中文标题)
    top_8_features = df_target_rf['Feature_Name'].head(8).tolist()
    
    for i, feat in enumerate(top_8_features):
        plt.figure(figsize=(8, 5))
        feat_idx = shap_x.columns.get_loc(feat)
        x_vals = shap_x[feat].values
        y_vals = shap_values[:, feat_idx]
        
        sns.scatterplot(x=x_vals, y=y_vals, alpha=0.5, color='#3498db', edgecolor=None)
        
        # 尝试使用 lowess 平滑，如果没装 statsmodels 则回退到 order=2
        try:
            sns.regplot(x=x_vals, y=y_vals, scatter=False, lowess=True, color='#e74c3c', line_kws={'linewidth': 2.5})
        except:
            sns.regplot(x=x_vals, y=y_vals, scatter=False, order=2, color='#e74c3c')
            
        plt.axhline(0, color='gray', linestyle='--', linewidth=1.5)
        plt.xlabel(translate(feat), fontsize=12)
        plt.ylabel(f"对 {translate(target)} 的贡献值", fontsize=12)
        plt.title(f"定量解剖：{translate(feat)} 的非线性物理阈值响应", fontsize=14, pad=15)
        plt.grid(alpha=0.3)
        plt.savefig(f'{SAVE_DIR}/Fig2_3_SHAP_Dep_{i+1}_{feat}_CN.png', dpi=DPI, bbox_inches='tight')
        plt.close()
    print("   => 已完成 Top 8 特征的高定版中文响应曲线扫描。")

# ================= Q3: 时空协同演变 (折线图 + 创新地图切片) =================

def plot_q3_combined(df_full):
    print("正在生成 Q3 时空演变系列 (折线演变 + 地图切片)...")
    
    # 选取数据最丰富的台风 ID
    top_typhoon = df_full['Typhoon_ID'].value_counts().index[0]
    df_case = df_full[df_full['Typhoon_ID'] == top_typhoon].sort_values('Time')
    df_case['Time'] = pd.to_datetime(df_case['Time'])

    # 3.1 原有的双轴时序演变图 (保留并优化)
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_xlabel('台风生命周期 (时间序列)')
    ax1.set_ylabel(translate('Wind_Speed'), color='tab:red')
    ax1.plot(df_case['Time'], df_case['Wind_Speed'], color='tab:red', linewidth=2, label='强度演变')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    ax2 = ax1.twinx()
    ax2.set_ylabel(translate('S_ext_Extreme_over_20'), color='tab:blue')
    ax2.fill_between(df_case['Time'], df_case['S_ext_Extreme_over_20'], color='tab:blue', alpha=0.3, label='降水面积')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    plt.title(f"台风 {top_typhoon} 演变复盘：动力核心与降水空间分布的时空耦合")
    plt.savefig(f'{SAVE_DIR}/Fig3_1_Evolution_Timeline.png', dpi=DPI)
    plt.close()

    # 3.2 创新的时空离散切片地图 (Snapshot Grid)
    indices = [int(len(df_case)*p) for p in [0.2, 0.4, 0.6, 0.8]]
    snapshots = df_case.iloc[indices]

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    for idx, (i, row) in enumerate(snapshots.iterrows()):
        ax = axes[idx]
        # 绘制中心
        ax.scatter(row['Lon'], row['Lat'], s=200, color='red', marker='p', label='台风中心')
        # 绘制降水范围边界
        radius = np.sqrt(row['S_ext_Extreme_over_20']) / 111.0 
        circle = plt.Circle((row['Lon'], row['Lat']), radius, color='blue', fill=False, linestyle='--', label='极端降水边界')
        ax.add_patch(circle)
        # 绘制移动箭头
        ax.quiver(row['Lon'], row['Lat'], np.sin(np.deg2rad(row['Moving_Direction'])), 
                  np.cos(np.deg2rad(row['Moving_Direction'])), scale=10, color='orange')

        ax.set_title(f"时刻: {row['Time']}\n强度: {row['Wind_Speed']}m/s | 范围: {int(row['S_ext_Extreme_over_20'])}km²")
        ax.set_xlabel(translate('Lon'))
        ax.set_ylabel(translate('Lat'))
        ax.grid(alpha=0.3)
        if idx == 0: ax.legend()

    plt.suptitle(f"台风 {top_typhoon} 生命周期关键节点时空演变切片", fontsize=18, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/Fig3_2_Spatial_Snapshots.png', dpi=DPI, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    try:
        data_full, corr, rf, sv, sx = load_data()
        plot_q1_enhanced(corr, data_full)
        plot_q2_full_shap(rf, sv, sx)
        plot_q3_combined(data_full)
        print(f"\n✨ 终极融合版可视化完成！图片已生成至: {SAVE_DIR}")
        print("💡 建议：重点查看 Fig2_3 系列 (定量阈值) 和 Fig3_2 (时空切片)。")
    except Exception as e:
        print(f"绘图执行失败: {e}")