import numpy as np
import glob
import pandas as pd  # 引入 pandas 用于优雅地导出数据

def calculate_metrics(npz_path):
    data = np.load(npz_path)
    matrices = data['matrices']  # [Seq_len, 128, 128]
    
    # 1. 峰值降水强度 (P_max)
    p_max = np.max(matrices)
    
    # 2. 极端降水总影响面积 (> 5.0 mm/h 的像素 * 100km^2)
    s_ext_extreme = np.sum(matrices > 5.0) * 100 
    
    # 3. 强降水滞留时间 (持续多少帧极值 > 3.0 mm/h)
    peak_per_frame = [np.max(m) for m in matrices]
    duration_steps = sum(1 for p in peak_per_frame if p > 3.0)
    
    # 4. 平均降水强度
    p_mean = np.mean(matrices[matrices > 0.1])
    
    return p_max, s_ext_extreme, duration_steps, p_mean

if __name__ == "__main__":
    print("================ 正在测算量化指标并导出文件 ================")
    
    files = glob.glob("*_DataPackage.npz")
    results_list = []

    for f in sorted(files):
        name = f.replace('_DataPackage.npz', '')
        p_max, s_ext, dur, p_mean = calculate_metrics(f)
        
        results_list.append({
            'Scenario': name,
            'P_max_mm_h': round(p_max, 2),
            'S_ext_km2': int(s_ext),
            'Duration_steps': dur,
            'P_mean_mm_h': round(p_mean, 2)
        })
    
    # 1. 导出为 CSV 文件
    df = pd.DataFrame(results_list)
    output_file = "Sensitivity_Analysis_Results.csv"
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    # 2. 同时在命令行输出 Markdown 格式方便查看
    # 把这行：
    # print("\n" + df.to_markdown(index=False))
    
    # 替换成这行：
    print("\n" + df.to_string(index=False))
    print(f"\n[★] 成功！量化指标已写入文件: {output_file}")