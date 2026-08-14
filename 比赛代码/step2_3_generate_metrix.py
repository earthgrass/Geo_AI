import torch
import numpy as np
import os
import time
import math
import rasterio
from rasterio.windows import from_bounds
from convLSTM_model import SpatialResidualConvLSTM

# 地球自转角速度 (rad/s)
OMEGA_EARTH = 7.2921e-5 

def calculate_rmax(wind_speed, lat):
    vmax_knots = wind_speed * 1.94384
    rmax_nm = 66.785 - 0.09102 * vmax_knots + 0.0105 * abs(lat)
    return max(15.0, min(150.0, rmax_nm * 1.852))

# ==========================================
# 模块 A: 真实地理引擎 (读取官方 DEM TIF)
# ==========================================
class RealGeographyEngine:
    def __init__(self, dem_path="Global_DEM.tif"):
        self.dem_path = dem_path
        if not os.path.exists(self.dem_path):
            raise FileNotFoundError(f"找不到真实地形文件 {self.dem_path}！请先下载并放在同级目录。")
        print(f"[*] 成功加载真实地球高程数据: {dem_path}")

    def get_real_dem_and_mask(self, center_lon, center_lat, radius_deg=5.75, shape=(128, 128)):
        min_lon, max_lon = center_lon - radius_deg, center_lon + radius_deg
        min_lat, max_lat = center_lat - radius_deg, center_lat + radius_deg
        
        with rasterio.open(self.dem_path) as src:
            window = from_bounds(min_lon, min_lat, max_lon, max_lat, src.transform)
            dem_matrix = src.read(1, window=window, out_shape=shape, resampling=rasterio.enums.Resampling.bilinear)
            land_mask = np.where(dem_matrix > 0, 1.0, 0.0)
            dem_matrix = np.clip(dem_matrix, 0, None)
            return dem_matrix, land_mask

# ==========================================
# 模块 B: 包含地转偏向力的物理引擎
# ==========================================
class PhysicsInformedEngine:
    def __init__(self, grid_size=128, res_km=10.0):
        self.grid_size = grid_size
        self.res_km = res_km
        y, x = np.ogrid[-grid_size//2:grid_size//2, -grid_size//2:grid_size//2]
        self.dist_matrix = np.sqrt(x**2 + y**2) * res_km
        self.angle_matrix = np.arctan2(x, y) 

    def render_true_physics_rain(self, v_max, r_max, p_center, heading_angle, center_lon, center_lat, dem, land_mask, enable_topo=True):
        # 1. 基础对称风压场
        wind = np.where(self.dist_matrix <= r_max, 
                        v_max * (self.dist_matrix / r_max), 
                        v_max * np.exp(-(self.dist_matrix - r_max) / 50.0))
        press = 1010.0 - max(1010.0 - p_center, 0) * np.exp(-(self.dist_matrix**2) / (2 * 300**2))
        
        # 2. 地转偏向力非对称性
        f_coriolis = 2 * OMEGA_EARTH * np.sin(math.radians(abs(center_lat)))
        heading_rad = math.radians(heading_angle)
        phase_diff = self.angle_matrix - (heading_rad + math.pi/4)
        coriolis_effect = 1.0 + (f_coriolis * 1e4) * np.cos(phase_diff) 
        
        # 3. 🚨 地形控制开关 (Ablation Study) 🚨
        if enable_topo:
            # 开启地形：迎风坡强迫抬升与陆地摩擦
            orographic_multiplier = 1.0 + (dem / 1000.0) * 0.35
            friction_multiplier = np.where(land_mask == 1.0, 1.20, 1.0)
        else:
            # 关闭地形：纯粹的海洋平面环境
            orographic_multiplier = 1.0
            friction_multiplier = 1.0
        
        # 4. 生成最终融合物理场
        base_rain = (wind / 5.0) ** 1.2 
        phys_rain = base_rain * coriolis_effect * orographic_multiplier * friction_multiplier
        phys_rain = np.clip(phys_rain, 0, None)
        
        return wind, press, phys_rain

def parse_cma_with_heading(txt_path, name):
    tracks = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('66666'):
            n = int(line.split()[2])
            if name.upper() in line.upper():
                for j in range(1, n+1):
                    p = lines[i+j].split()
                    lat, lon = float(p[2])/10.0, float(p[3])/10.0
                    tracks.append([lat, lon, float(p[4]), float(p[5]), p[0], 0.0]) 
                break
            i += n + 1
        else: i += 1
        
    for k in range(1, len(tracks)):
        lat1, lon1 = tracks[k-1][0], tracks[k-1][1]
        lat2, lon2 = tracks[k][0], tracks[k][1]
        dy = lat2 - lat1
        dx = np.cos(math.radians(lat1)) * (lon2 - lon1)
        angle = (math.degrees(math.atan2(dx, dy)) + 360) % 360
        tracks[k][5] = angle
    if len(tracks) > 0: tracks[0][5] = tracks[1][5] if len(tracks) > 1 else 0.0
    return tracks

class PINN_InferenceEngine:
    def __init__(self, weight):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SpatialResidualConvLSTM().to(self.device)
        self.model.load_state_dict(torch.load(weight, map_location=self.device))
        self.model.eval()
        self.phys_engine = PhysicsInformedEngine()
        self.geo_engine = RealGeographyEngine("Global_DEM.tif")

    def run(self, name, track, enable_topo=True):
        # 如果关闭了地形，文件名加上 _NoTopo 后缀
        suffix = "" if enable_topo else "_NoTopo"
        save_name = f"{name}{suffix}"
        
        print(f"\n[*] 正在为 {save_name} 进行物理推演 (地形模块开启状态: {enable_topo})...")
        all_feat = []
        base_rains = []
        
        for lat, lon, pres, wind, _, heading in track:
            rmax = calculate_rmax(wind, lat)
            dem, land_mask = self.geo_engine.get_real_dem_and_mask(lon, lat)
            
            # 传入 enable_topo 参数
            w_f, p_f, phys_rain = self.phys_engine.render_true_physics_rain(
                wind, rmax, pres, heading, lon, lat, dem, land_mask, enable_topo=enable_topo)
            
            all_feat.append(np.stack([phys_rain/10.0, w_f, p_f, self.phys_engine.dist_matrix]))
            base_rains.append(phys_rain) 
            
        all_feat = np.array(all_feat, dtype=np.float32) 
        results, lons, lats, times_str, winds, presses = [], [], [], [], [], []
        
        print(f"    -> 启动 数据驱动+自回归 融合推理...")
        for i in range(11, len(track)):
            x = torch.tensor(all_feat[i-11:i], dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                nn_residual = self.model(x).squeeze().cpu().numpy()
            
            final_pred = base_rains[i] + nn_residual * 1.5 
            final_pred = np.clip(final_pred, 0, None)
            
            # 自回归同化闭环
            if i < len(track) - 1:
                all_feat[i, 0, :, :] = final_pred / 10.0 
            
            results.append(final_pred)
            lat_i, lon_i, pres_i, wind_i, time_i, _ = track[i]
            lats.append(lat_i); lons.append(lon_i); times_str.append(time_i)
            winds.append(wind_i); presses.append(pres_i)
            
        # 统一保存为 DataPackage.npz 方便后续评估脚本读取
        np.savez_compressed(f"{save_name}_DataPackage.npz", matrices=results, lons=lons, lats=lats, 
                            times=times_str, winds=winds, presses=presses)
        print(f" [★] {save_name} 推演完毕！")

if __name__ == "__main__":
    import torch
    torch.set_num_threads(16) 
    
    if not os.path.exists("Global_DEM.tif"):
        print("🚨 警告: 找不到 Global_DEM.tif！请前往 NOAA 下载 ETOPO1 数据。")
    else:
        engine = PINN_InferenceEngine("typhoon_convlstm_best.pth")
        
        # ==================================================
        # 针对第二问：正常推演 + 地形消融实验
        # ==================================================
        print("\n\n============= 第二问：基准台风与地形消融实验 =============")
        engine.run("KONG-REY", parse_cma_with_heading("CH2024BST.txt", "KONG-REY"), enable_topo=True)
        # 这就是关闭地形模块的对照组，跑完后去减上面的结果就是地形贡献！
        engine.run("KONG-REY", parse_cma_with_heading("CH2024BST.txt", "KONG-REY"), enable_topo=False) 
        
        engine.run("MAN-YI", parse_cma_with_heading("CH2024BST.txt", "MAN-YI"), enable_topo=True)

        # ==================================================
        # 针对第三问：未来气候极端严谨情景推演 (更新版)
        # ==================================================
        print("\n\n============= 第三问：未来气候极端严谨情景推演 =============")
        if os.path.exists("Virtual_Typhoons_2026.txt"):
            engine.run("V-SHIFT", parse_cma_with_heading("Virtual_Typhoons_2026.txt", "V-SHIFT"), enable_topo=True)
            engine.run("V-INTENSE", parse_cma_with_heading("Virtual_Typhoons_2026.txt", "V-INTENSE"), enable_topo=True)
            engine.run("V-COMPOUND", parse_cma_with_heading("Virtual_Typhoons_2026.txt", "V-COMPOUND"), enable_topo=True)
            engine.run("V-SLOW", parse_cma_with_heading("Virtual_Typhoons_2026.txt", "V-SLOW"), enable_topo=True)
        else:
            print("⚠️ 未找到 Virtual_Typhoons_2026.txt，请确保你已经运行了 step3_1 生成脚本！")