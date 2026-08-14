import numpy as np
import matplotlib.pyplot as plt
import os

def render_typhoon_heatmap(npy_file, typhoon_name):
    """
    将预测的降水矩阵渲染为高清气象热力图
    """
    if not os.path.exists(npy_file):
        print(f" [!] 找不到文件: {npy_file}")
        return

    print(f"[*] 正在渲染 {typhoon_name} 的降水分布图...")
    
    # 1. 加载预测矩阵
    # 经过 ReLU 激活，数据理论上是非负的，但为了画图更纯粹，我们可以加一层 np.clip 保底
    precip_matrix = np.load(npy_file)
    precip_matrix = np.clip(precip_matrix, a_min=0.0, a_max=None)
    
    # 2. 设置画布 (300 DPI 保证论文打印绝对清晰)
    plt.figure(figsize=(10, 8), dpi=300)
    
    # 3. 核心绘图：使用 'turbo' 色带 (非常适合展示降水的极端梯度，比 jet 更平滑现代)
    # origin='lower' 是因为矩阵通常原点在左上角，而地理坐标系原点在左下角
    im = plt.imshow(precip_matrix, cmap='turbo', origin='lower', interpolation='bilinear')
    
    # 4. 专业气象图表修饰
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label('Precipitation Rate (mm/h)', fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=12)
    
    plt.title(f'Predicted Spatial Precipitation Distribution\nTyphoon {typhoon_name} (2024)', 
              fontsize=16, fontweight='bold', pad=15)
    
    # 隐藏坐标轴刻度数字 (因为我们映射的是 0-127 网格，直接显示数字没有地理意义)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Spatial Grid (Longitude)', fontsize=14)
    plt.ylabel('Spatial Grid (Latitude)', fontsize=14)
    
    # 5. 输出保存
    output_png = f"Figure_Q2_Precipitation_{typhoon_name}.png"
    plt.savefig(output_png, bbox_inches='tight', transparent=False, facecolor='white')
    print(f" [★] 渲染完成！图片已保存为: {output_png}")
    
    # 自动在屏幕上弹窗展示一下
    plt.show()

if __name__ == "__main__":
    print("========== 启动 Q2 视觉渲染引擎 ==========")
    
    render_typhoon_heatmap("KONG-REY_pred.npy", "KONG-REY")
    render_typhoon_heatmap("MAN-YI_pred.npy", "MAN-YI")
    
    print("========== 所有图片渲染完毕，可以插入论文！ ==========")