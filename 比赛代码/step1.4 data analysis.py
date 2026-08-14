import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import shap
import os
import warnings
warnings.filterwarnings('ignore')

def run_pure_quantitative_engine(input_csv):
    print("==================================================")
    print("⚙️ Step 1.4 纯净版数学分析引擎已启动 (Pure Math Engine)")
    print("==================================================")
    
    # 1. 载入与预处理
    df = pd.read_csv(input_csv)
    
    # 特征空间定义
    X_cols = [
        'Lat', 'Lon', 'Wind_Speed', 'Pressure', 'Radius_max_wind_km', 
        'Moving_Speed_kmh', 'Moving_Direction', 'Curvature_deg_per_km', 
        'Delta_P_6h', 'Delta_V_6h'
    ]
    Y_cols = ['P_total', 'P_max', 'S_ext_Extreme_over_20', 'D_offset_km', 'I_asy_Index']
    all_cols = X_cols + Y_cols
    
    # 清洗：构建绝对干净的数据底座
    df_clean = df.dropna(subset=all_cols).copy()
    print(f"[数据加载] 有效样本总量: {len(df_clean)}")

    # ==========================================
    # 模块 A: 全矩阵 Spearman 秩相关系数 (客观事实)
    # ==========================================
    print("\n[计算中] 模块 A: 全维度 Spearman 相关性矩阵...")
    # 输出完整的 N x N 矩阵，不进行任何画图裁剪
    full_corr_matrix = df_clean[all_cols].corr(method='spearman')
    full_corr_matrix.to_csv('1.4_Data_Full_Spearman_Matrix.csv')
    print("  -> 落盘完毕: 1.4_Data_Full_Spearman_Matrix.csv")

    # ==========================================
    # 模块 B: 随机森林全量特征重要性 (客观事实)
    # ==========================================
    print("\n[计算中] 模块 B: 随机森林全量特征重要性 (Gini Impurity)...")
    rf_data_records = []
    
    # 为所有因变量建立独立映射，客观记录每个特征的贡献度
    for target in Y_cols:
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(df_clean[X_cols], df_clean[target])
        
        importances = rf.feature_importances_
        for feature, imp in zip(X_cols, importances):
            rf_data_records.append({
                'Target_Variable': target,
                'Feature_Name': feature,
                'Raw_Importance_Score': imp
            })
            
    df_rf_raw = pd.DataFrame(rf_data_records)
    # 纯数据导出，不进行面向画图的 sort_values 排序
    df_rf_raw.to_csv('1.4_Data_RF_Raw_Importance.csv', index=False)
    print("  -> 落盘完毕: 1.4_Data_RF_Raw_Importance.csv")

    # ==========================================
    # 模块 C: SHAP 基础值矩阵生成 (底层归因数据)
    # ==========================================
    print("\n[计算中] 模块 C: SHAP 归因底座生成 (Target: S_ext)...")
    target_shap = 'S_ext_Extreme_over_20'
    X_shap = df_clean[X_cols]
    y_shap = df_clean[target_shap]
    
    rf_shap = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
    rf_shap.fit(X_shap, y_shap)
    
    explainer = shap.TreeExplainer(rf_shap)
    # 计算并直接吐出原生 numpy 矩阵
    shap_raw_values = explainer.shap_values(X_shap)
    
    np.save('1.4_Data_SHAP_Matrix_S_ext.npy', shap_raw_values)
    X_shap.to_csv('1.4_Data_SHAP_Base_Features.csv', index=False)
    print("  -> 落盘完毕: 1.4_Data_SHAP_Matrix_S_ext.npy")
    print("  -> 落盘完毕: 1.4_Data_SHAP_Base_Features.csv")

    print("\n✅ Step 1.4 计算层已全部完成，所有数据已封存！")

if __name__ == '__main__':
    run_pure_quantitative_engine('Typhoon_Full_Dataset_Q1.csv')